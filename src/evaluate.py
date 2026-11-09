# src/evaluate.py


import os
import json
import datetime
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

import config
from labeller import load_and_process_chunks, load_and_process_questions, chunk_question_grader, grader
from retrieval import dense_retrieval

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

RETRIEVERS = {
    "dense": dense_retrieval,
}

def find_unreachable(all_chunks, questions_lookup):
    """Evidence items that no chunk of the current version can grade 2 (the evidence-level ceiling)."""
    unreachable = []
    for qid, question in questions_lookup.items():
        for evidence in question['evidence']:
            page = evidence['evidence_page_num']
            found = False
            for chunk in all_chunks:
                if chunk['doc_id'] == evidence['doc_name'] and chunk['page_start'] <= page <= chunk['page_end']:
                    if grader(chunk, evidence)[0] == 2:
                        found = True
                        break
            if not found:
                unreachable.append((qid, evidence['doc_name'], page))
    return unreachable

# Dev evidence items that no chunk can grade 2


KS = [5, 10, 20, 50]
EXHIBITS = "Financial Statements & Exhibits"


def run_retriever(name, dev_questions_lookup):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.index(name=config.INDEX_NAME)
    retriever = RETRIEVERS[name]

    results = {}
    for qid, question in dev_questions_lookup.items():
        chunks = retriever(question['question'], openai_client, index,
                           include_fields=["chunk_id", "doc_id", "page_start"], top_k=config.TOP_K)

        # Checks: right count, no duplicates, scores never increase, ranks always increase
        assert len(chunks) == config.TOP_K
        chunk_ids = []
        prev_score = None
        prev_rank = 0
        for chunk in chunks:
            assert chunk['chunk_id'] not in chunk_ids
            chunk_ids.append(chunk['chunk_id'])
            if prev_score is not None:
                assert chunk['score'] <= prev_score
            prev_score = chunk['score']
            assert chunk['rank'] > prev_rank
            prev_rank = chunk['rank']

        results[qid] = chunks

    settings = {
        "version": config.VERSION,
        "retriever": name,
        "top_k": config.TOP_K,
        "namespace": config.NAMESPACE,
        "embedding_model": config.EMBEDDING_MODEL,
        "date": datetime.date.today().isoformat(),
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "results": results}, f, indent=2)

    return settings, results


def grade(results, chunks_lookup, questions_lookup):
    for qid, chunks in results.items():
        for chunk in chunks:
            chunk['grades'] = chunk_question_grader(chunk['chunk_id'], qid, chunks_lookup, questions_lookup)
    return results


def report(settings, results, questions_lookup, chunks_lookup, all_chunks, unreachable):
    print(f"Settings: {settings}\n")

    # ---------- 1. For every evidence item: the first rank where it is found ----------
    all_items = []

    for qid, chunks in results.items():
        question = questions_lookup[qid]

        for i, evidence in enumerate(question['evidence']):
            first_rank_evidence = None   # first rank with grade 2
            first_rank_page = None       # first rank with grade 1 or 2

            for chunk in chunks:         # chunks are already in rank order
                grade_value = chunk['grades'][i][0]
                if grade_value == 2 and first_rank_evidence is None:
                    first_rank_evidence = chunk['rank']
                if grade_value >= 1 and first_rank_page is None:
                    first_rank_page = chunk['rank']

            reachable = (qid, evidence['doc_name'], evidence['evidence_page_num']) not in unreachable

            all_items.append({
                "qid": qid,
                "reachable": reachable,
                "first_rank_evidence": first_rank_evidence,
                "first_rank_page": first_rank_page,
            })

    # ---------- 2. Checks ----------
    assert len(results) == 38
    assert len(all_items) == 49

    for item in all_items:
        if not item['reachable']:
            assert item['first_rank_evidence'] is None, f"unreachable item got grade 2: {item}"

    reachable_items = []
    reachable_qids = []
    for item in all_items:
        if item['reachable']:
            reachable_items.append(item)
            if item['qid'] not in reachable_qids:
                reachable_qids.append(item['qid'])

    print(f"Questions: 38 (evidence-reachable {len(reachable_qids)})")
    print(f"Evidence items: 49 (reachable {len(reachable_items)})\n")

    # ---------- 3. Metrics ----------
    for k in KS:
        recall_evidence = 0
        recall_page = 0
        hit_evidence_qids = []
        hit_page_qids = []

        for item in all_items:
            if item['first_rank_evidence'] is not None and item['first_rank_evidence'] <= k:
                recall_evidence += 1
                if item['qid'] not in hit_evidence_qids:
                    hit_evidence_qids.append(item['qid'])

            if item['first_rank_page'] is not None and item['first_rank_page'] <= k:
                recall_page += 1
                if item['qid'] not in hit_page_qids:
                    hit_page_qids.append(item['qid'])

        print(f"k={k}")
        print(f"  Evidence  hit {len(hit_evidence_qids)}/{len(reachable_qids)} = {len(hit_evidence_qids)/len(reachable_qids):.2f}"
              f" | recall {recall_evidence}/{len(reachable_items)} = {recall_evidence/len(reachable_items):.2f}")
        print(f"  Page      hit {len(hit_page_qids)}/38 = {len(hit_page_qids)/38:.2f}"
              f" | recall {recall_page}/49 = {recall_page/49:.2f}")

    # ---------- 4. Exhibits share ----------
    total_results = 0
    exhibit_results = 0
    for qid, chunks in results.items():
        for chunk in chunks:
            total_results += 1
            if chunks_lookup[chunk['chunk_id']]['section'] == EXHIBITS:
                exhibit_results += 1

    exhibit_corpus = 0
    for chunk in all_chunks:
        if chunk['section'] == EXHIBITS:
            exhibit_corpus += 1

    print(f"\nExhibits: {exhibit_results/total_results:.3f} of top-{settings['top_k']} results vs {exhibit_corpus/len(all_chunks):.3f} of the corpus")

    # ---------- 5. Top-10 filing mix ----------
    same_filing = 0
    same_company_other_year = 0
    other_company = 0

    for qid, chunks in results.items():
        gold_doc = questions_lookup[qid]['doc_name']
        gold_company = gold_doc.rsplit("_", 2)[0]      # "JOHNSON_JOHNSON_2022_10K" -> "JOHNSON_JOHNSON"

        for chunk in chunks[:10]:
            doc = chunks_lookup[chunk['chunk_id']]['doc_id']
            if doc == gold_doc:
                same_filing += 1
            elif doc.rsplit("_", 2)[0] == gold_company:
                same_company_other_year += 1
            else:
                other_company += 1

    total = same_filing + same_company_other_year + other_company
    print(f"Top-10 by filing: same filing {same_filing/total:.2f} | "
          f"same company, other year {same_company_other_year/total:.2f} | "
          f"other company {other_company/total:.2f}")


def main():
    all_chunks, all_chunks_lookup = load_and_process_chunks()
    all_questions, dev_questions_lookup, test_questions_lookup = load_and_process_questions()
    assert len(dev_questions_lookup) == 38

    unreachable = find_unreachable(all_chunks, dev_questions_lookup)
    print(f"Unreachable evidence items for {config.VERSION}: {len(unreachable)}")
    for item in unreachable:
        print(f"  {item}")

    for name in RETRIEVERS:
        print(f"\n==================== {name} ====================")
        path = config.RESULTS_DIR / f"{name}.json"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            settings = saved['settings']
            results = saved['results']
        else:
            settings, results = run_retriever(name, dev_questions_lookup)

        results = grade(results, all_chunks_lookup, dev_questions_lookup)
        report(settings, results, dev_questions_lookup, all_chunks_lookup, all_chunks, unreachable)


if __name__ == "__main__":
    main()