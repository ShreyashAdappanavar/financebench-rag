# src/answer.py

import os
import time
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

import config
from labeller import load_and_process_questions
from retrieval import CHUNKS_LOOKUP, dense_retrieval, hybrid_retrieval, filtered_dense, filtered_hybrid, rerank_dense, rerank_hybrid
import matplotlib.pyplot as plt

load_dotenv()

REJUDGE = False   # True = overwrite existing judge scores; set back to False after this run

# Setup name -> (retriever, number of chunks given to the answer model); None = no context
SETUPS = {
    "no_context": (None, 0),
    "dense_top10": (dense_retrieval, 10),
    "dense_top20": (dense_retrieval, 20),
    "hybrid_top10": (hybrid_retrieval, 10),
    "filtered_dense_top10": (filtered_dense, 10),
    "filtered_hybrid_top10": (filtered_hybrid, 10),
    "rerank_dense_top10": (rerank_dense, 10),
    "rerank_hybrid_top10": (rerank_hybrid, 10),
    "rerank_dense": (rerank_dense, 5),          # top 5, kept under its old name to reuse answers
    "rerank_dense_top20": (rerank_dense, 20),
}

# Ladder: one change per step
LADDER = ["no_context", "dense_top10", "hybrid_top10", "filtered_dense_top10",
          "filtered_hybrid_top10", "rerank_dense_top10", "rerank_hybrid_top10"]

# Chunk-count sweep on the chosen pipeline: (setup name, chunks)
SWEEP = [("rerank_dense", 5), ("rerank_dense_top10", 10), ("rerank_dense_top20", 20)]

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

def judge_answer(question, gold, answer, openai_client):
    # "I don't know" is decided by rule, not by the judge
    if answer.strip().lower().replace("’", "'").startswith("i don't know"):
        return {"reason": "Answer starts with I don't know (rule).", "score": 1}

    prompt = f"""You are grading an answer to a financial question about a company's 10-K filing.

Question: {question}

Gold answer: {gold}

Answer to grade: {answer}

Score the answer:
- 2 (correct): the final result matches the gold answer: the same figures (equal after rounding or a change of units, e.g. $1,577 million = $1.58 billion), the same direction (higher/lower, yes/no), and the same main drivers or items the gold lists. Working shown and citations are fine.
- 1 (I don't know): the answer says it cannot answer and gives no final result.
- 0 (wrong): anything else. If the main figures, the direction or the main drivers differ from the gold answer, or the answer gives only part of what the question asks for, score 0.

Give a one-line reason, then the score.
Judge only the final result against the gold answer. Do not use your own knowledge of the company."""

    schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "score": {"type": "integer", "enum": [0, 1, 2]},
        },
        "required": ["reason", "score"],
        "additionalProperties": False
    }

    res = openai_client.responses.create(
        model=config.ANSWER_MODEL,
        input=prompt,
        reasoning={"effort": "low"},
        text={"format": {"type": "json_schema", "name": "judge_answer", "schema": schema, "strict": True}}
    )
    return json.loads(res.output_text)


def judge_all(answers, openai_client):
    for qid, value in answers.items():
        for name in SETUPS:
            if 'judge_score' in value[name] and not REJUDGE:   # already judged
                continue
            value[name]['judge_score'] = judge_answer(value['question'], value['gold'], value[name]['answer'], openai_client)
    return answers


def print_scores(answers):
    scores = {}
    for name in SETUPS:
        counts = {0: 0, 1: 0, 2: 0}
        for value in answers.values():
            counts[value[name]['judge_score']['score']] += 1
        scores[name] = {"correct": counts[2], "idk": counts[1], "wrong": counts[0]}
        n = len(answers)
        print(f"{name}: correct {counts[2]}/{n} | I don't know {counts[1]}/{n} | wrong {counts[0]}/{n}")
    return scores

def plot_charts(scores):
    # Chart 1: ablation ladder, stacked correct / I don't know / wrong
    correct = [scores[name]['correct'] for name in LADDER]
    idk = [scores[name]['idk'] for name in LADDER]
    wrong = [scores[name]['wrong'] for name in LADDER]
    bottom_wrong = [c + i for c, i in zip(correct, idk)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(LADDER, correct, label="Correct", color="#2e7d32")
    ax.bar(LADDER, idk, bottom=correct, label="I don't know", color="#9e9e9e")
    ax.bar(LADDER, wrong, bottom=bottom_wrong, label="Wrong", color="#c62828")
    for i, c in enumerate(correct):
        ax.text(i, c / 2, str(c), ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylabel("Questions (of 38 dev)")
    ax.set_title("Ablation ladder: answer quality by retrieval setup (top 10 chunks)")
    ax.legend(loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "ladder.png", dpi=150)
    plt.close()

    # Chart 2: chunk-count sweep on reranked dense
    ks = [k for name, k in SWEEP]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [scores[name]['correct'] for name, k in SWEEP], marker="o", label="Correct", color="#2e7d32")
    ax.plot(ks, [scores[name]['idk'] for name, k in SWEEP], marker="o", label="I don't know", color="#9e9e9e")
    ax.plot(ks, [scores[name]['wrong'] for name, k in SWEEP], marker="o", label="Wrong", color="#c62828")
    ax.set_xticks(ks)
    ax.set_xlabel("Chunks given to the answer model")
    ax.set_ylabel("Questions (of 38 dev)")
    ax.set_title("Reranked dense: answers vs number of chunks")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "chunk_sweep.png", dpi=150)
    plt.close()

def main():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).index(name=config.INDEX_NAME)
    all_questions, dev_questions_lookup, test_questions_lookup = load_and_process_questions()
    path = config.RESULTS_DIR / "answers.json"

    # 1. Load existing answers (if any)
    answers = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            answers = json.load(f)

    # 2. Generate only the setups missing for each question
    for qid, question in dev_questions_lookup.items():
        qs = question['question']
        if qid not in answers:
            answers[qid] = {"question": qs, "gold": question['answer']}

        for name, (retriever, n_chunks) in SETUPS.items():
            if name in answers[qid]:   # already generated
                continue
            if retriever is None:
                chunks = []
            else:
                chunks = retriever(qs, openai_client, index, include_fields=["chunk_id", "doc_id", "page_start"])[:n_chunks]

            answers[qid][name] = {
                "chunk_ids": [chunk['chunk_id'] for chunk in chunks],
                "answer": answer(qs, chunks, openai_client),
            }
        print(f"done {qid}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)

    # 3. Judge (skips answers already judged unless REJUDGE), save, print
    answers = judge_all(answers, openai_client)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)

    scores = print_scores(answers)
    with open(config.RESULTS_DIR / "answer_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    plot_charts(scores)


if __name__ == "__main__":
    main()