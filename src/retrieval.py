# src/retrieval.py

import os
from openai import OpenAI
from pinecone.index import Index
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

NAMESPACE = os.getenv("NAMESPACE")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def dense_retrieval(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                    top_k: int = 50) -> List[Dict]:


    qs_embeddings = openai_client.embeddings.create(input=qs, model=EMBEDDING_MODEL).data[0].embedding
    res = index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            score_by=[{
                "type": "dense_vector", 
                "field": "embedding",
                "values": qs_embeddings
                }],
            include_fields=include_fields,
        )

    dense_chunks = []

    for i, doc in enumerate(res.matches):
        curr_score = doc.score
        curr_doc = doc.to_dict()
        curr_doc['score'] = curr_score
        curr_doc['rank'] = i + 1
        dense_chunks.append(curr_doc)

    return dense_chunks

