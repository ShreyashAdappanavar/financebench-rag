# src/config.py
# To try a new chunking experiment - set VERSION, add its namespace to NAMESPACES, change the chunk settings.
# Every chunking change needs a new VERSION: index.py skips steps when files already exist.

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = Path(os.getenv("BASE_PATH"))

FINBENCH_DIR = BASE_PATH / "finbench_github" / "financebench"
QUESTIONS_PATH = FINBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDF_DIR = FINBENCH_DIR / "pdfs"

PAGES_DIR = BASE_PATH / "data" / "pages"     
SPLIT_PATH = BASE_PATH / "data" / "split.json"   # frozen dev/test split, shared by all versions


################ CHUNK SETTINGS - CHANGE! ################
VERSION = "v1"
CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

NAMESPACES = {
    "v1": "__default__",  
}
NAMESPACE = NAMESPACES[VERSION]

VERSION_DIR = BASE_PATH / "data" / VERSION
CHUNKS_PATH = VERSION_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = VERSION_DIR / "all_embeddings.npy"
IDS_PATH = VERSION_DIR / "all_ids.npy"
BATCH_DIR = VERSION_DIR / "batch_storage"
RESULTS_DIR = VERSION_DIR / "results"

# ---------- Settings ----------
INDEX_NAME = "finbench-rag-index"
EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 50