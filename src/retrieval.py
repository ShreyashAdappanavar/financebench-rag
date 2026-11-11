# src/retrieval.py

from openai import OpenAI
import config
from pinecone.index import Index
from typing import List, Dict
import json
import os
from pinecone import Pinecone

NAMESPACE = config.NAMESPACE
INDEX_NAME = config.INDEX_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL
TOP_K = config.TOP_K
K_RRF = config.K_RRF

# Chunk text (prefix included) by chunk_id, for the reranker
CHUNKS_LOOKUP = {}
with open(config.CHUNKS_PATH, encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        CHUNKS_LOOKUP[chunk['chunk_id']] = chunk

def compute_individual_rrf_score(rank: int, k_rrf: int) -> float:
    return 1/(k_rrf + rank)

def extract_companies_years(qs: str, openai_client: OpenAI) -> Dict[str, List]:
    # 31 companies from filings.json; the model may only pick from these
    with open(config.FILINGS_PATH) as f:
        filings = json.load(f)
    companies = list(filings.keys())

    prompt = f"""Extract the companies and fiscal years mentioned in this question. 
Return every company mentioned, using only names from the allowed list.
Return every fiscal year written in the question, as numbers. Turn two-digit years into four digits (“FY22” → 2022)
Never guess a year that is not written (e.g. "last year" means no year).
Return empty lists if nothing is mentioned.

Question: {qs}"""

    schema = {
        "type": "object",
        "properties": {
            "companies": {"type": "array", "items": {"type": "string", "enum": companies}},
            "years": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["companies", "years"],
        "additionalProperties": False
    }

    res = openai_client.responses.create(
        model=config.FILTER_EXTRACTOR_MODEL,
        input=prompt,
        reasoning={"effort": "none"},
        text={"format": {"type": "json_schema", "name": "companies_years", "schema": schema, "strict": True}}
    )
    return json.loads(res.output_text)

def build_filter(extracted: Dict[str, List]) -> tuple[Dict | None, str]:
    with open(config.FILINGS_PATH) as f:
        filings = json.load(f)

    clauses = []
    for company in extracted['companies']:
        if company not in filings:
            continue
        valid_years = [float(year) for year in extracted['years'] if year in filings[company]]
        if valid_years:
            clauses.append({"company": {"$eq": company}, "fiscal_year": {"$in": valid_years}})
        else:
            clauses.append({"company": {"$eq": company}})

    if not clauses:
        return None, "none"

    year_clauses = [clause for clause in clauses if "fiscal_year" in clause]
    if len(year_clauses) == len(clauses):
        filter_level = "filing"
    elif len(year_clauses) == 0:
        filter_level = "company"
    else:
        filter_level = "mixed"

    return {"$or": clauses}, filter_level

def dense_retrieval(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                    top_k: int = TOP_K, metadata_filter: Dict | None = None) -> List[Dict]:

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
            filter=metadata_filter,
        )

    chunks = []

    for i, doc in enumerate(res.matches):
        curr_score = doc.score
        curr_doc = doc.to_dict()
        curr_doc['dense_score'] = curr_score
        curr_doc['dense_rank'] = i + 1
        curr_doc['score'] = curr_score
        curr_doc['rank'] = i + 1
        chunks.append(curr_doc)

    return chunks

def bm25_retrieval(qs: str, openai_client: OpenAI | None, index: Index, include_fields: List[str],
                   top_k: int = TOP_K, metadata_filter: Dict | None = None) -> List[Dict]:
    
    res = index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            score_by=[{
                "type": "text",
                "fields": ["text"],
                "query": qs,
                }],
            include_fields=include_fields,
            filter=metadata_filter,
        )

    chunks = []

    for i, doc in enumerate(res.matches):
        curr_score = doc.score
        curr_doc = doc.to_dict()
        curr_doc['bm25_score'] = curr_score
        curr_doc['bm25_rank'] = i + 1
        curr_doc['score'] = curr_score
        curr_doc['rank'] = i + 1
        chunks.append(curr_doc)

    return chunks


def hybrid_retrieval(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                     top_k: int = TOP_K, metadata_filter: Dict | None = None) -> List[Dict]:

    dense_res = dense_retrieval(qs, openai_client, index, include_fields, top_k, metadata_filter)
    bm25_res = bm25_retrieval(qs, openai_client, index, include_fields, top_k, metadata_filter)

    chunk_db = {}
    for chunk in dense_res:
        tmp = {}
        tmp['chunk_id'] = chunk['chunk_id']
        tmp['doc_id'] = chunk['doc_id']
        tmp['page_start'] = chunk['page_start']
        tmp['dense_score'] = chunk['score']
        tmp['dense_rank'] = chunk['rank']
        tmp['bm25_score'] = None
        tmp['bm25_rank'] = None
        tmp['hybrid_score'] = compute_individual_rrf_score(chunk['rank'], K_RRF)

        chunk_db[chunk['chunk_id']] = tmp

    for chunk in bm25_res:
        if chunk['chunk_id'] not in chunk_db.keys():
            tmp = {}
            tmp['chunk_id'] = chunk['chunk_id']
            tmp['doc_id'] = chunk['doc_id']
            tmp['page_start'] = chunk['page_start']
            tmp['dense_score'] = None
            tmp['dense_rank'] = None
            tmp['bm25_score'] = chunk['score']
            tmp['bm25_rank'] = chunk['rank']
            tmp['hybrid_score'] = compute_individual_rrf_score(chunk['rank'], K_RRF)

            chunk_db[chunk['chunk_id']] = tmp

        else:
            chunk_db[chunk['chunk_id']]['bm25_score'] = chunk['score']
            chunk_db[chunk['chunk_id']]['bm25_rank'] = chunk['rank']
            chunk_db[chunk['chunk_id']]['hybrid_score'] += compute_individual_rrf_score(chunk['rank'], K_RRF)

    # Highest RRF score first; ties broken by chunk ID (A to Z) so reruns give the same order
    sorted_chunks = sorted(chunk_db.values(), key=lambda record: (-record['hybrid_score'], record['chunk_id']))

    for i, record in enumerate(sorted_chunks):
        record['hybrid_rank'] = i + 1
        record['rank'] = i + 1
        record['score'] = record['hybrid_score']

    return sorted_chunks

def filtered_dense(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                   top_k: int = TOP_K) -> List[Dict]:
    
    extracted = extract_companies_years(qs, openai_client)
    metadata_filter, filter_level = build_filter(extracted)
    chunks = dense_retrieval(qs, openai_client, index, include_fields, top_k, metadata_filter)
    for chunk in chunks:
        chunk['filter_level'] = filter_level
        chunk['companies'] = extracted['companies']
        chunk['years'] = extracted['years']
    return chunks

def filtered_bm25(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                  top_k: int = TOP_K) -> List[Dict]:
    
    extracted = extract_companies_years(qs, openai_client)
    metadata_filter, filter_level = build_filter(extracted)
    chunks = bm25_retrieval(qs, openai_client, index, include_fields, top_k, metadata_filter)
    for chunk in chunks:
        chunk['filter_level'] = filter_level
        chunk['companies'] = extracted['companies']
        chunk['years'] = extracted['years']
    return chunks

def filtered_hybrid(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                    top_k: int = TOP_K) -> List[Dict]:
    
    extracted = extract_companies_years(qs, openai_client)
    metadata_filter, filter_level = build_filter(extracted)
    chunks = hybrid_retrieval(qs, openai_client, index, include_fields, top_k, metadata_filter)
    for chunk in chunks:
        chunk['filter_level'] = filter_level
        chunk['companies'] = extracted['companies']
        chunk['years'] = extracted['years'] 
    return chunks

def rerank(qs: str, chunks: List[Dict]) -> List[Dict]:
    assert len(chunks) <= 100   # Pinecone's reranker limit per call

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    documents = [{"text": CHUNKS_LOOKUP[chunk['chunk_id']]['text']} for chunk in chunks]

    res = pc.inference.rerank(
        model=config.RERANKER_MODEL,
        query=qs,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        parameters={"truncate": "END"},
    )

    reranked = []
    for i, item in enumerate(res.data):
        chunk = chunks[item.index]   # item.index = position in the list we sent
        chunk['reranker_score'] = item.score
        chunk['reranker_rank'] = i + 1
        chunk['score'] = item.score
        chunk['rank'] = i + 1
        reranked.append(chunk)

    return reranked

def rerank_dense(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                 top_k: int = TOP_K) -> List[Dict]:
    chunks = filtered_dense(qs, openai_client, index, include_fields, top_k)
    return rerank(qs, chunks)

def rerank_hybrid(qs: str, openai_client: OpenAI, index: Index, include_fields: List[str],
                  top_k: int = TOP_K) -> List[Dict]:
    chunks = filtered_hybrid(qs, openai_client, index, include_fields, top_k)
    return rerank(qs, chunks)