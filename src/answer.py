# src/answer.py

import os
import time
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

import config
from labeller import load_and_process_questions
from retrieval import CHUNKS_LOOKUP, filtered_dense, rerank_dense

load_dotenv()

TOP_N = 5   # chunks given to the answer model

# Setup name -> retriever (None = no context, the baseline)
SETUPS = {
    "no_context": None,
    "filtered_dense": filtered_dense,
    "rerank_dense": rerank_dense,
}


def answer(qs, chunks, openai_client):
    if chunks:
        context = ""
        for chunk in chunks:
            record = CHUNKS_LOOKUP[chunk['chunk_id']]
            context += f"[{chunk['chunk_id']}] {record['company']} FY{int(record['fiscal_year'])}, page {int(record['page_start'])}\n{record['text']}\n\n"
        prompt = f"""Answer the question using only the chunks below.
Cite the chunk IDs you used in square brackets.
If the answer is not in the chunks, say "I don't know".

{context}Question: {qs}"""
    else:
        prompt = f"Answer the question.\n\nQuestion: {qs}"

    res = openai_client.responses.create(
        model=config.ANSWER_MODEL,
        input=prompt,
        reasoning={"effort": "low"},
    )
    return res.output_text


def main():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=10)
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).index(name=config.INDEX_NAME)
    all_questions, dev_questions_lookup, test_questions_lookup = load_and_process_questions()

    path = config.RESULTS_DIR / "answers.json"
    answers = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            answers = json.load(f)

    start = time.time()
    for i, (qid, question) in enumerate(dev_questions_lookup.items()):

        if qid in answers:
            continue

        qs = question['question']
        answers[qid] = {"question": qs, "gold": question['answer']}

        for name, retriever in SETUPS.items():
            if retriever is None:
                chunks = []
            else:
                chunks = retriever(qs, openai_client, index, include_fields=["chunk_id", "doc_id", "page_start"])[:TOP_N]

            answers[qid][name] = {
                "chunk_ids": [chunk['chunk_id'] for chunk in chunks],
                "answer": answer(qs, chunks, openai_client),
            }


        with open(config.RESULTS_DIR / "answers.json", "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2)

        elapsed = time.time() - start
        remaining = elapsed / (i + 1) * (len(dev_questions_lookup) - i - 1)
        print(f"done {i + 1}/{len(dev_questions_lookup)} {qid} | elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min")


if __name__ == "__main__":
    main()