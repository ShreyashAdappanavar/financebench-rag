# src/app.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

import config
from answer import run_pipeline

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).index(name=config.INDEX_NAME)

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    chunks, answer_text = run_pipeline(query.question, openai_client, index)
    return {
        "answer": answer_text,
        "chunks": [
            {"chunk_id": c['chunk_id'], "doc_id": c['doc_id'], "page_start": c['page_start'], "reranker_score": c['reranker_score']}
            for c in chunks
        ],
    }