# 5_upload_chunks_to_pinecone.py

import os
import json
from pathlib import Path
import numpy as np
from pinecone import Pinecone
from tqdm import tqdm
from dotenv import load_dotenv
import random
from collections import Counter


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BASE_PATH = Path(os.getenv("BASE_PATH"))
NAMESPACE = os.getenv("NAMESPACE")

INDEX_NAME = "finbench-rag-index"
ALL_DATA_PATH = BASE_PATH / "data"
CHUNKS = ALL_DATA_PATH / "chunks.jsonl"
ALL_IDS = ALL_DATA_PATH / "all_ids.npy"
ALL_EMBEDDINGS = ALL_DATA_PATH / "all_embeddings.npy"


FIELDS = ["doc_id", "company", "fiscal_year", "form_type", "section",
          "page_start", "page_end", "chunk_index", "prefix", "text"]

all_chunks = []

with open(CHUNKS, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            obj = json.loads(line)
            all_chunks.append(obj)

all_ids = np.load(ALL_IDS)
all_embeddings = np.load(ALL_EMBEDDINGS)

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