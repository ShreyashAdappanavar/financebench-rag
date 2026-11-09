# src/retrieval.py

from openai import OpenAI
import config
from pinecone.index import Index
from typing import List, Dict


NAMESPACE = config.NAMESPACE
INDEX_NAME = config.INDEX_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL
TOP_K = config.TOP_K

def dense_retrieval(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                    top_k: int = TOP_K) -> List[Dict]:


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