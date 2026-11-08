# src/10_run_dense_dev.py

import os
from openai import OpenAI
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv
from retrieval import dense_retrieval
import json
from typing import List, Dict

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BASE_PATH = Path(os.getenv("BASE_PATH"))
INDEX_NAME = os.getenv("INDEX_NAME")

CHUNKS = BASE_PATH / "data" / "chunks.jsonl"
ALL_QUESTIONS_PATH = BASE_PATH / "finbench_github" / "financebench" / "data" / "financebench_open_source.jsonl"
SPLIT = BASE_PATH / "data" / "split.json"
DEV_QUESTION_CHUNKS = BASE_PATH / "data" / "dev_questions" / "dev_questions_chunks.json"
DEV_QUESTION_CHUNKS.parent.mkdir(parents=True, exist_ok=True)


openai_client = OpenAI(api_key=OPENAI_API_KEY)

with open(SPLIT, "r", encoding="utf-8") as f:
    splits = json.load(f)


dev_questions = []

with open(ALL_QUESTIONS_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            obj = json.loads(line)
            if obj['financebench_id'] in splits['dev']:
                dev_questions.append(obj)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.index(name=INDEX_NAME)

all_dev_chunks = {}

assert len(dev_questions) == 38

for dev_qs in dev_questions:

    dense_chunks = dense_retrieval(dev_qs['question'], openai_client, index, include_fields=["chunk_id", "doc_id", "page_start"])
    
    assert len(dense_chunks) == 50
    
    chunk_ids = []
    min_score = 1.1
    curr_rank = 0
    
    for chunk in dense_chunks:
        assert chunk['chunk_id'] not in chunk_ids
        chunk_ids.append(chunk['chunk_id'])
        assert chunk['score'] <= min_score
        min_score = chunk['score']
        
        assert chunk['rank'] > curr_rank
        curr_rank = chunk['rank']
    
    all_dev_chunks[dev_qs['financebench_id']] = dense_chunks



with open(DEV_QUESTION_CHUNKS, "w", encoding="utf-8") as f:
    json.dump(all_dev_chunks, f, indent=2)