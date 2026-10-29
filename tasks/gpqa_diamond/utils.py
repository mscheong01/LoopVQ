"""GPQA-Diamond: deterministic option shuffle, simple-evals prompt, last-answer letter scoring."""
import hashlib
import random
import re

LETTERS = "ABCD"
_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think "
    "step by step before answering.\n\n{q}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def process_docs(dataset):
    def _one(doc):
        opts = [_clean(doc["Correct Answer"]), _clean(doc["Incorrect Answer 1"]),
                _clean(doc["Incorrect Answer 2"]), _clean(doc["Incorrect Answer 3"])]
        order = list(range(4))
        # seeded by the question itself, so every arm / seed / run sees the same option order
        random.Random(int(hashlib.sha256(doc["Record ID"].encode()).hexdigest(), 16)).shuffle(order)
        out = {L: opts[i] for L, i in zip(LETTERS, order)}
        out["question"] = doc["Question"].strip()
        out["answer"] = LETTERS[order.index(0)]
        return out
    return dataset.map(_one, remove_columns=dataset.column_names)


def doc_to_text(doc):
    return _TEMPLATE.format(q=doc["question"], **{L: doc[L] for L in LETTERS})


# Most-specific marker first; within each, the LAST hit wins (a thinking model restates and
# sometimes corrects itself, and the final statement is its answer).
_MD = r"[\*_`\s\$\(\[\{]*"
_PATTERNS = [
    re.compile(r"Answer\s*[:：]" + _MD + r"(?:\\text\{)?" + r"([A-D])\b"),
    re.compile(r"\\boxed\{" + _MD + r"(?:\\text\{)?([A-D])\b"),
    re.compile(r"answer is" + _MD + r"(?:option\s*)?" + r"([A-D])\b", re.I),
]


def extract_answer(text):
    text = text or ""
    # score only what follows the reasoning when the template's think block is in the reply
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for pat in _PATTERNS:
        hits = pat.findall(text)
        if hits:
            return hits[-1].upper()
    return None


def process_results(doc, results):
    pred = extract_answer(results[0])
    return {"exact_match": float(pred is not None and pred == doc["answer"])}
