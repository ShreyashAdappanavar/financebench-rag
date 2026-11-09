# src/index.py

import os
import json
from openai import OpenAI
import numpy as np
from pathlib import Path
import re
import time
from dotenv import load_dotenv
import config
from pinecone import Pinecone
from tqdm import tqdm
import random
from collections import Counter

load_dotenv()



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_SAVE_PATH = config.BATCH_DIR
EMBEDDING_MODEL = config.EMBEDDING_MODEL
CHUNKS_PATH = config.CHUNKS_PATH
EMBEDDINGS_PATH = config.EMBEDDINGS_PATH
IDS_PATH = config.IDS_PATH
INDEX_NAME = config.INDEX_NAME
NAMESPACE = config.NAMESPACE

FIELDS = ["doc_id", "company", "fiscal_year", "form_type", "section",
          "page_start", "page_end", "chunk_index", "prefix", "text"]

def embed():

    batch_size = 512
    start_index = 0

    os.makedirs(EMBEDDING_SAVE_PATH, exist_ok=True)

    openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=10)

    def pad_int(x: int): return f"{x:0{6}d}"

    all_chunks = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_chunks.append(obj)

    print(f"all chunks loaded. Number of chunks: {len(all_chunks)}")

    if os.path.exists(EMBEDDING_SAVE_PATH):
        files = os.listdir(EMBEDDING_SAVE_PATH)
        files.sort(reverse=True)
        if len(files) > 0:
            pattern = re.compile(r"^ids_till_.+$")
            rel_files = [s for s in files if pattern.match(s)]
            last_index = int(rel_files[0].split(".")[0].split("_")[-1])
            start_index = last_index + 1
        else:
            start_index = 0

    print(f"Start index has been set as: {start_index}")

    total_batches = (len(all_chunks) + batch_size - 1) // batch_size  
    done_before = start_index // batch_size    
    run_seconds, run_batches = 0.0, 0    

    for i in range(start_index, len(all_chunks), batch_size):

        time0 = time.time()

        curr_slice = all_chunks[i:i+batch_size]
        
        inputs = [chunk['prefix'] + "\n" + chunk['text'] for chunk in curr_slice]
        ids = [chunk['chunk_id'] for chunk in curr_slice]

        resp = openai_client.embeddings.create(input=inputs, model= EMBEDDING_MODEL)
        embedding = np.array([data.embedding for data in resp.data])
        assert len(embedding) == len(inputs)
        assert [j.index for j in resp.data] == list(range(len(inputs)))

        last_index = min(i + batch_size, len(all_chunks)) - 1
        final_embedding_file_name = f"embeddings_till_{pad_int(last_index)}.npy"
        final_ids_file_name = f"ids_till_{pad_int(last_index)}.npy"

        np.save(EMBEDDING_SAVE_PATH / f"tmp_{final_embedding_file_name}", embedding.astype(np.float32))
        np.save(EMBEDDING_SAVE_PATH / f"tmp_{final_ids_file_name}", np.array(ids))

        os.replace(EMBEDDING_SAVE_PATH / f"tmp_{final_embedding_file_name}", 
                EMBEDDING_SAVE_PATH / f"{final_embedding_file_name}")
        
        os.replace(EMBEDDING_SAVE_PATH / f"tmp_{final_ids_file_name}", 
                EMBEDDING_SAVE_PATH / f"{final_ids_file_name}")

        elapsed = time.time() - time0
        run_seconds += elapsed
        run_batches += 1
        avg = run_seconds / run_batches                    
        done = done_before + run_batches
        remaining = total_batches - done
        print(f"Batch {done}/{total_batches} (chunks {i}-{last_index}) in {elapsed:.2f}s | "
              f"avg {avg:.2f}s/batch | {remaining} left | ETA {avg * remaining / 60:.1f} min")




def extract_int(x: str): return x.split(".")[0].split("_")[-1]


def merge():

    all_chunks = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_chunks.append(obj)

    if not os.path.exists(EMBEDDING_SAVE_PATH):
        raise FileNotFoundError(f"No batch folder at {EMBEDDING_SAVE_PATH}")

    files = os.listdir(EMBEDDING_SAVE_PATH)

    id_pattern = re.compile(r"^ids_till_.+$")
    id_files = [s for s in files if id_pattern.match(s)]
    id_files.sort()

    embedding_pattern = re.compile(r"^embeddings_till_.+$")
    embedding_files = [s for s in files if embedding_pattern.match(s)]
    embedding_files.sort()

    all_embeddings, all_ids = [], []

    assert len(id_files) == len(embedding_files), f"{len(id_files)} ids files vs {len(embedding_files)} embeddings files"  # zip would drop extras silently

    for curr_embeddings_file_name, curr_ids_file_name in zip(embedding_files, id_files):
        assert extract_int(curr_embeddings_file_name) == extract_int(curr_ids_file_name)
        curr_ids = np.load(EMBEDDING_SAVE_PATH / curr_ids_file_name)
        curr_embeddings = np.load(EMBEDDING_SAVE_PATH / curr_embeddings_file_name)
        
        all_embeddings.append(curr_embeddings)
        all_ids.append(curr_ids)

    all_embeddings = np.concatenate(all_embeddings)
    all_ids = np.concatenate(all_ids)

    assert all_ids.tolist() == [chunk['chunk_id'] for chunk in all_chunks]

    assert all_embeddings.shape == (len(all_ids), 1536), all_embeddings.shape   # rows match ids, right dimension
    assert not np.isnan(all_embeddings).any(), "NaN in embeddings"
    norms = np.linalg.norm(all_embeddings, axis=1)
    assert norms.min() > 0.99 and norms.max() < 1.01, (norms.min(), norms.max())  # also catches all-zero rows

    np.save(EMBEDDINGS_PATH, all_embeddings.astype(np.float32))
    np.save(IDS_PATH, np.array(all_ids))

    print(f"saved {all_embeddings.shape} | norm min {norms.min():.4f} max {norms.max():.4f} | first {all_ids[0]} last {all_ids[-1]}")




def upload():

    all_chunks = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_chunks.append(obj)

    all_ids = np.load(IDS_PATH)
    all_embeddings = np.load(EMBEDDINGS_PATH)

    for curr_id, curr_chunk, curr_embedding in zip(all_ids, all_chunks, all_embeddings):
        assert curr_id == curr_chunk["chunk_id"]
        curr_chunk["_id"] = curr_chunk["chunk_id"]
        curr_chunk['text'] = curr_chunk['prefix'] + "\n" + curr_chunk['text']
        curr_chunk['embedding'] = curr_embedding.tolist()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.index(INDEX_NAME)
    start_index = 0
    batch_size = 50

    with tqdm(total=len(all_chunks), initial=start_index, unit="chunk", dynamic_ncols=True, desc="Upserting") as pbar:
        while start_index < len(all_chunks):
            end_index = min(start_index + batch_size, len(all_chunks))
            res = index.documents.upsert(namespace=NAMESPACE, documents=all_chunks[start_index:end_index])
            assert res.upserted_count == end_index - start_index
            pbar.update(end_index - start_index)
            pbar.set_postfix_str(f"last {start_index}-{end_index}")
            start_index = end_index

    res = index.describe_namespace(name=NAMESPACE)
    assert res.record_count == len(all_chunks)
    print(res.record_count)


def verify():
    all_chunks = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_chunks.append(obj)

    all_ids = np.load(IDS_PATH)
    all_embeddings = np.load(EMBEDDINGS_PATH)

    for curr_id, curr_chunk, curr_embedding in zip(all_ids, all_chunks, all_embeddings):
        assert curr_id == curr_chunk["chunk_id"]
        curr_chunk["_id"] = curr_chunk["chunk_id"]
        curr_chunk['text'] = curr_chunk['prefix'] + "\n" + curr_chunk['text']
        curr_chunk['embedding'] = curr_embedding

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.index(INDEX_NAME)

    random.seed(42)
    sample = random.sample(range(len(all_chunks)), 50)

    passed = Counter()
    failures = []
    dense_scores = []
    bm25_fallbacks = 0


    def score_of(match):
        s = getattr(match, "score", None)
        return s if s is not None else match.get("_score")


    def bm25_search(query, top_k):
        return index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            score_by=[{"type": "text", "fields": ["text"], "query": query}],
            include_fields=["*"],
        )


    for i in sample:
        chunk = all_chunks[i]
        cid = chunk["chunk_id"]
        docs = index.documents.fetch(namespace=NAMESPACE, ids=[cid]).documents
        try:
            doc = docs[cid]
        except KeyError:
            failures.append(("fetch", cid, "not found in index"))
        else:
            bad = {f: (repr(doc.get(f))[:60], repr(chunk[f])[:60]) for f in FIELDS if doc.get(f) != chunk[f]}
            if bad:
                failures.append(("fetch", cid, bad))
            else:
                passed["fetch"] += 1


        res = index.documents.search(
            namespace=NAMESPACE,
            top_k=5,
            score_by=[{"type": "dense_vector", "field": "embedding",
                    "values": np.asarray(chunk["embedding"]).tolist()}],
            include_fields=["*"],
        )
        top = res.matches[0]
        s = score_of(top)
        dense_scores.append(s)
        if top.get("_id") == cid and s > 0.999:
            passed["dense_self"] += 1
        else:
            failures.append(("dense_self", cid, (top.get("_id"), s)))


        try:
            res = bm25_search(chunk["text"], top_k=5)
        except Exception:
            bm25_fallbacks += 1
            res = bm25_search(" ".join(chunk["text"].split()[:100]), top_k=5)
        top = res.matches[0]
        if top.get("_id") == cid:
            passed["bm25_self"] += 1
        else:
            failures.append(("bm25_self", cid, (top.get("_id"), score_of(top))))


    print(f"Sample: {len(sample)} chunks")
    for check in ["fetch", "dense_self", "bm25_self"]:
        print(f"{check:<11} passed {passed[check]}/{len(sample)}")
    print(f"lowest dense self-score: {min(dense_scores):.6f}")
    print(f"BM25 queries that needed the 100-word fallback: {bm25_fallbacks}")

    print(f"\nFailures ({len(failures)}):")
    for check, cid, detail in failures:
        print(f"[{check}] {cid}: {detail}")


    print("\nBM25 top 10 for 'purchases of property plant and equipment':")
    res = bm25_search("purchases of property plant and equipment", top_k=10)
    for rank, m in enumerate(res.matches, 1):
        print(f"  {rank:>2}. {m.get('_id'):<32} {str(m.get('section')):<34} {score_of(m):.3f}")


def main():

    n_chunks = 0
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                n_chunks += 1

    # Embed + merge: only if the merged embeddings don't exist yet
    if not EMBEDDINGS_PATH.exists():
        embed()
        merge()

    # Upload: only if the namespace doesn't already hold every chunk
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.index(INDEX_NAME)
    try:
        n_uploaded = index.describe_namespace(name=NAMESPACE).record_count
    except Exception:
        n_uploaded = 0      # namespace doesn't exist yet
    if n_uploaded != n_chunks:
        upload()

    verify()

if __name__ == "__main__":
    main()