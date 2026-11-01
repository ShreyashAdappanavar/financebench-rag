# labeller.py 
# (chunk - question grader) 


import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple, List

load_dotenv()

MIN_WORD_COUNT = 20
OVERLAP_THRESHOLD = 0.8

BASE_PATH = Path(os.getenv("BASE_PATH"))
ALL_QUESTIONS_PATH = BASE_PATH / "finbench_github" / "financebench" / "data" / "financebench_open_source.jsonl"
SPLIT_PATH = BASE_PATH / "data" / "split.json"
ALL_DATA_PATH = BASE_PATH / "data"
CHUNKS = ALL_DATA_PATH / "chunks.jsonl"


def load_questions()  -> List[dict]:
    questions = []

    with open(ALL_QUESTIONS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                obj = json.loads(line)
                questions.append(obj)
    all_questions = []
    for qs in questions:
        if qs['doc_name'].split("_")[-1] == "10K" :
            all_questions.append(qs)

    return all_questions


def load_chunks() -> List[dict]:

    all_chunks = []

    with open(CHUNKS, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_chunks.append(obj)

    return all_chunks


def text_replacements(text: str) -> str:
    text = text.lower()
    text = text.replace("−", "-")
    text = text.replace("$", "")
    return text

def process_and_normalize_text(text: str) -> set:
    text = text_replacements(text)
    all_words = text.split()
    processed_words = []
    
    for word in all_words:

        negative = False
        word = word.strip(",.;:%' ’ “ ” \"")

        if len(word) > 0:

            if (word[0] == "(") and (word[-1] == ")"):
                negative = True

            if word[0] =="(": word = word[1:]
            if len(word) > 0 and word[-1] == ")": word = word[:-1]

            try:
                float(word.replace(",", ""))
                word = word.replace(",", "")
                if negative: word = "-" + word

            except ValueError:
                pass

            if len(word) > 0: processed_words.append(word)

    return set(processed_words)



def grader(chunk: dict, evidence: dict, thr: float = OVERLAP_THRESHOLD, min_word_count: int = MIN_WORD_COUNT) -> Tuple[int, float | None]:

    # Does chunk have same doc? If TRUE, return 0. If FALSE, don't check equality condition again 
    if chunk['doc_id'] != evidence['doc_name']:
        return 0, None

    # If chunk has the same doc, does evidence page number and chunk page number match?
    if not (chunk['page_start'] <= evidence['evidence_page_num'] <= chunk['page_end']):
        return 0, None

    # If function has not returned yet, that means chunk and evidence are from the same page of the same document. Now minimum grade will be 1.

    chunk_set = chunk['chunk_set_of_words']
    evidence_set = evidence['evidence_set_of_words']
    min_len = min(len(chunk_set), len(evidence_set))

    if min_len < min_word_count: 
        return 1, None

    curr_overlap = len(chunk_set.intersection(evidence_set)) / min_len

    if curr_overlap < thr:
        return 1, curr_overlap

    return 2, curr_overlap
    

def chunk_question_grader(chunk_id: str, question_id: str, all_chunks_lookup: dict[str, dict], 
                          all_questions_lookup: dict[str, dict], thr: float = OVERLAP_THRESHOLD,
                          min_word_count: int = MIN_WORD_COUNT) -> List[Tuple[int, float | None]]:

    chunk = all_chunks_lookup[chunk_id]
    question = all_questions_lookup[question_id]

    all_grades = []

    for evidence in question['evidence']:
        all_grades.append(grader(chunk=chunk, evidence=evidence, thr=thr, min_word_count=min_word_count))

    return all_grades

        
def load_and_process_chunks() -> Tuple[List[dict], dict]:
    all_chunks = load_chunks()

    for chunk in all_chunks:
        chunk['chunk_set_of_words'] = process_and_normalize_text(chunk['text'])

    all_chunks_lookup = {chunk['chunk_id']: chunk for chunk in all_chunks}

    return all_chunks, all_chunks_lookup


def load_and_process_questions()  -> Tuple[List[dict], dict, dict]:

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split = json.load(file)

    dev_ids = split["dev"]
    test_ids = split["test"]

    all_questions = load_questions()

    for question in all_questions:
        for evidence in question['evidence']:
            evidence['evidence_set_of_words'] = process_and_normalize_text(evidence['evidence_text'])

    dev_questions = [el for el in all_questions if el['financebench_id'] in dev_ids]
    dev_questions_lookup = {el["financebench_id"]: el for el in dev_questions}

    test_questions = [el for el in all_questions if el['financebench_id'] in test_ids]
    test_questions_lookup = {el["financebench_id"]: el for el in test_questions}


    return all_questions, dev_questions_lookup, test_questions_lookup


################## Running It: ##################
# all_chunks, all_chunks_lookup = load_and_process_chunks()
# all_questions, dev_questions_lookup, test_questions_lookup = load_and_process_questions()
# all_grades = chunk_question_grader(chunk_id=chunk_id, question_id=question_id, all_chunks_lookup=all_chunks_lookup, all_questions_lookup=dev_questions_lookup)
# grade = max([k[0] for k in all_grades])