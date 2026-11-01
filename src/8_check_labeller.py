# src/8_check_labeller.py
# Step 5c: does the labeller give every dev evidence item at least one grade-2 chunk?

import random
from collections import defaultdict
import numpy as np
import labeller

all_chunks, all_chunks_lookup = labeller.load_and_process_chunks()
all_questions, dev_lookup, _ = labeller.load_and_process_questions()

# Group chunks by filing so we only scan one filing per evidence item
chunks_by_doc = defaultdict(list)
for chunk in all_chunks:
    chunks_by_doc[chunk["doc_id"]].append(chunk)


def page_chunks(evidence):
    """Chunks of the evidence's filing that touch the evidence page."""
    p = evidence["evidence_page_num"]
    return [c for c in chunks_by_doc[evidence["doc_name"]] if c["page_start"] <= p <= c["page_end"]]


def overlap(chunk, evidence):
    """Overlap with thr=0, so grader returns it whenever the texts are big enough (else None)."""
    return labeller.grader(chunk, evidence, thr=0.0)[1]


# ---------- 1. Coverage ----------
best = []  # (question_id, doc, page, best_overlap) per evidence item
for qid, q in dev_lookup.items():
    for ev in q["evidence"]:
        scores = [s for s in (overlap(c, ev) for c in page_chunks(ev)) if s is not None]
        best.append((qid, ev["doc_name"], ev["evidence_page_num"], max(scores) if scores else 0.0))

values = np.array([b[3] for b in best])
print(f"Dev evidence items: {len(best)}")
print(f"Best-overlap quantiles (25/50/75%): {np.quantile(values, [0.25, 0.5, 0.75]).round(3)}")
for thr in (0.7, 0.8, 0.9):
    print(f"  threshold {thr}: {int((values >= thr).sum())}/{len(best)} items have a grade-2 chunk")

print(f"\nItems below {labeller.OVERLAP_THRESHOLD}:")
for qid, doc, page, b in sorted(best, key=lambda x: x[3]):
    if b < labeller.OVERLAP_THRESHOLD:
        print(f"  {qid}  {doc} p{page}  best overlap {b:.3f}")

# ---------- 2. Spot check by eye ----------
random.seed(0)
print("\n========== Spot check (judge each grade yourself) ==========")
for qid in random.sample(sorted(dev_lookup), 3):
    q = dev_lookup[qid]
    print(f"\n{qid} | {q['doc_name']} | {q['question'][:120]}")
    for ev in q["evidence"]:
        print(f"  Evidence page {ev['evidence_page_num']}: {ev['evidence_text'][:150]!r}")
        for c in page_chunks(ev):
            grade, ov = labeller.grader(c, ev)
            ov_str = f"{ov:.3f}" if ov is not None else "n/a"
            print(f"    {c['chunk_id']}  p{c['page_start']}-{c['page_end']}  overlap {ov_str}  grade {grade}")
            print(f"      {c['text'][:200]!r}")