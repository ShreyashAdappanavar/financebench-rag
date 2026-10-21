# 6_make_question_splits.py

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import random


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BASE_PATH = Path(os.getenv("BASE_PATH"))

ALL_QUESTIONS_PATH = BASE_PATH / "finbench_github" / "financebench" / "data" / "financebench_open_source.jsonl"

SEED = 42
random.seed(SEED)

SPLIT_PATH = BASE_PATH / "data" / "split.json"

all_questions = []

with open(ALL_QUESTIONS_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            obj = json.loads(line)
            all_questions.append(obj)


_10k_questions = [el for el in all_questions if el['doc_name'].split("_")[-1] == "10K"]
metrics_generated_list = sorted([el['financebench_id'] for el in _10k_questions if el['question_type'] == 'metrics-generated'])
domain_relevant_list = sorted([el['financebench_id'] for el in _10k_questions if el['question_type'] == 'domain-relevant'])
novel_generated_list = sorted([el['financebench_id'] for el in _10k_questions if el['question_type'] == 'novel-generated'])

random.shuffle(metrics_generated_list)
random.shuffle(domain_relevant_list)
random.shuffle(novel_generated_list)

dev_metrics_generated_length = round((1/3)*len(metrics_generated_list))
dev_domain_relevant_length = round((1/3)*len(domain_relevant_list))
dev_novel_generated_length = round((1/3)*len(novel_generated_list))



dev_question_ids_by_type = {
    "metrics-generated": metrics_generated_list[:dev_metrics_generated_length],
    "domain-relevant": domain_relevant_list[:dev_domain_relevant_length],
    "novel-generated": novel_generated_list[:dev_novel_generated_length],
}

test_question_ids_by_type = {
    "metrics-generated": metrics_generated_list[dev_metrics_generated_length:],
    "domain-relevant": domain_relevant_list[dev_domain_relevant_length:],
    "novel-generated": novel_generated_list[dev_novel_generated_length:],
}

if SPLIT_PATH.exists():
    raise SystemExit(f"{SPLIT_PATH} already exists - the split is frozen.")

dev_ids = [qid for ids in dev_question_ids_by_type.values() for qid in ids]
test_ids = [qid for ids in test_question_ids_by_type.values() for qid in ids]

all_10k_ids = {q["financebench_id"] for q in _10k_questions}
assert len(all_10k_ids) == len(_10k_questions) == 112, "expected 112 unique 10-K questions"
assert len(set(dev_ids)) == len(dev_ids) and len(set(test_ids)) == len(test_ids), "duplicate ID within a side"
assert not set(dev_ids) & set(test_ids), "an ID is in both dev and test"
assert set(dev_ids) | set(test_ids) == all_10k_ids, "dev + test must be exactly the 112 10-K questions"

counts = {
    "dev": {t: len(ids) for t, ids in dev_question_ids_by_type.items()},
    "test": {t: len(ids) for t, ids in test_question_ids_by_type.items()},
}

company_of = {q["financebench_id"]: q["company"] for q in _10k_questions}
dev_companies = {company_of[qid] for qid in dev_ids}
test_companies = {company_of[qid] for qid in test_ids}
test_only_companies = sorted(test_companies - dev_companies)

split = {
    "seed": SEED,
    "method": "10-K questions sorted by financebench_id, grouped by question_type, "
              "shuffled per type, first third (rounded) of each type -> dev",
    "counts": counts,
    "dev": sorted(dev_ids),
    "test": sorted(test_ids),
}

SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SPLIT_PATH, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)

print(f"wrote {SPLIT_PATH}")
print(f"dev  {len(dev_ids):>3}  {counts['dev']}")
print(f"test {len(test_ids):>3}  {counts['test']}")
print(f"companies only in test: {len(test_only_companies)} of {len(test_companies)}  {test_only_companies}")