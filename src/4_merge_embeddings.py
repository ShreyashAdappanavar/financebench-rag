# src/4_merge_embeddings.py

import os
import json
from pathlib import Path
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()

def extract_int(x: str): return x.split(".")[0].split("_")[-1]

BASE_PATH = Path(os.getenv("BASE_PATH"))

CHUNK_DATA_PATH = BASE_PATH / "data"
EMBEDDING_SAVE_PATH = CHUNK_DATA_PATH / "batch_storage"


def main():

    all_chunks = []

    with open(CHUNK_DATA_PATH / "chunks.jsonl", "r", encoding="utf-8") as file:
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

    np.save(CHUNK_DATA_PATH / f"all_embeddings.npy", all_embeddings.astype(np.float32))
    np.save(CHUNK_DATA_PATH / f"all_ids.npy", np.array(all_ids))

    print(f"saved {all_embeddings.shape} | norm min {norms.min():.4f} max {norms.max():.4f} | first {all_ids[0]} last {all_ids[-1]}")

if __name__ == "__main__":
    main()