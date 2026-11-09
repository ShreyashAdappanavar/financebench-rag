# src/retrieval.py

from openai import OpenAI
import config
from pinecone.index import Index
from typing import List, Dict


NAMESPACE = config.NAMESPACE
INDEX_NAME = config.INDEX_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL
TOP_K = config.TOP_K
K_RRF = config.K_RRF


def compute_individual_rrf_score(rank: int, k_rrf: int) -> float:
    return 1/(k_rrf + rank)

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

    chunks = []

    for i, doc in enumerate(res.matches):
        curr_score = doc.score
        curr_doc = doc.to_dict()
        curr_doc['score'] = curr_score
        curr_doc['rank'] = i + 1
        chunks.append(curr_doc)

    return chunks

def bm25_retrieval(qs: str, openai_client: OpenAI | None, index: Index, include_fields: List[str],
                    top_k: int = TOP_K) -> List[Dict]:
    
    res = index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            score_by=[{
                "type": "text",
                "fields": ["text"],
                "query": qs,
                }],
            include_fields=include_fields,
        )

    chunks = []

    for i, doc in enumerate(res.matches):
        curr_score = doc.score
        curr_doc = doc.to_dict()
        curr_doc['score'] = curr_score
        curr_doc['rank'] = i + 1
        chunks.append(curr_doc)

    return chunks


def hybrid_retrieval(qs: str, openai_client: OpenAI | None, index: Index, include_fields: List[str],
                    top_k: int = TOP_K) -> List[Dict]:
    
    dense_res = dense_retrieval(qs, openai_client, index, include_fields, TOP_K)
    bm25_res = bm25_retrieval(qs, openai_client, index, include_fields, TOP_K)

    chunk_db = {}
    for chunk in dense_res:
        tmp = {}
        tmp['chunk_id'] = chunk['chunk_id']
        tmp['doc_id'] = chunk['doc_id']
        tmp['page_start'] = chunk['page_start']
        tmp['dense_rank'] = chunk['rank']
        tmp['bm25_rank'] = None
        tmp['rrf_score'] = compute_individual_rrf_score(chunk['rank'], K_RRF)

        chunk_db[chunk['chunk_id']] = tmp

    for chunk in bm25_res:
        if chunk['chunk_id'] not in chunk_db.keys():
            tmp = {}
            tmp['chunk_id'] = chunk['chunk_id']
            tmp['doc_id'] = chunk['doc_id']
            tmp['page_start'] = chunk['page_start']
            tmp['dense_rank'] = None
            tmp['bm25_rank'] = chunk['rank']
            tmp['rrf_score'] = compute_individual_rrf_score(chunk['rank'], K_RRF)
            
            chunk_db[chunk['chunk_id']] = tmp

        else:
            chunk_db[chunk['chunk_id']]['bm25_rank'] = chunk['rank']
            chunk_db[chunk['chunk_id']]['rrf_score'] += compute_individual_rrf_score(chunk['rank'], K_RRF)

    # Highest RRF score first; ties broken by chunk ID (A to Z) so reruns give the same order
    sorted_chunks = sorted(chunk_db.values(), key=lambda record: (-record['rrf_score'], record['chunk_id']))

    for i, record in enumerate(sorted_chunks):
        record['rank'] = i + 1
        record['score'] = record['rrf_score']

    return sorted_chunks