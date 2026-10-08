# src/create_embeddings.py

import os
import json
from openai import OpenAI
from pathlib import Path
import numpy as np
import re
import time
from dotenv import load_dotenv

load_dotenv()

def main():

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_PATH = Path(os.getenv("BASE_PATH"))
    batch_size = 512
    start_index = 0

    CHUNK_DATA_PATH = BASE_PATH / "data"
    EMBEDDING_SAVE_PATH = CHUNK_DATA_PATH / "batch_storage"

    os.makedirs(EMBEDDING_SAVE_PATH, exist_ok=True)

    openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=10)

    def pad_int(x: int): return f"{x:0{6}d}"

    all_chunks = []

    with open(CHUNK_DATA_PATH / "chunks.jsonl", "r", encoding="utf-8") as file:
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

        resp = openai_client.embeddings.create(input=inputs, model='text-embedding-3-small')
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

if __name__ == "__main__":
    main()