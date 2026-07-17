from __future__ import annotations

from collections import Counter
import re


DISTRACTOR_ANSWERS = (
    "感染",
    "炎症",
    "观察",
    "进一步检查",
    "保守治疗",
    "无法确定",
    "功能性问题",
)

_PUNCTUATION_PATTERN = re.compile(r"[\s\-_/:;,.!?()\[\]{}\"'`~@#$%^&*+=<>|\\]+")


def normalize_text(text: str) -> str:
    compact = _PUNCTUATION_PATTERN.sub("", text.strip().lower())
    return compact


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def mixed_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    latin_buffer: list[str] = []

    def flush_latin() -> None:
        if latin_buffer:
            tokens.append("".join(latin_buffer).lower())
            latin_buffer.clear()

    for char in text:
        if _is_cjk(char):
            flush_latin()
            tokens.append(char)
            continue
        if char.isalnum():
            latin_buffer.append(char)
            continue
        flush_latin()

    flush_latin()
    return tokens


def weighted_jaccard(left: str, right: str) -> float:
    left_counts = Counter(mixed_tokenize(left))
    right_counts = Counter(mixed_tokenize(right))
    if not left_counts and not right_counts:
        return 1.0
    intersection = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    return intersection / union if union else 0.0


def split_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r"[。！？!?]\s*|\n+", text)
    return [sentence.strip() for sentence in raw_sentences if sentence.strip()]


def answer_score(reference: str, candidate: str) -> float:
    ref = normalize_text(reference)
    pred = normalize_text(candidate)

    if not ref or not pred:
        return 0.0
    if ref == pred:
        return 1.0
    if ref in pred or pred in ref:
        shorter = min(len(ref), len(pred))
        longer = max(len(ref), len(pred))
        return shorter / longer if longer else 0.0

    ref_counter = Counter(ref)
    pred_counter = Counter(pred)
    common = sum((ref_counter & pred_counter).values())
    if not common:
        return 0.0

    precision = common / len(pred)
    recall = common / len(ref)
    return (2 * precision * recall) / (precision + recall)


def choose_distractor(gold_answer: str, salt: int = 0) -> str:
    normalized_gold = normalize_text(gold_answer)
    candidates = [answer for answer in DISTRACTOR_ANSWERS if normalize_text(answer) != normalized_gold]
    if not candidates:
        return "无法确定"
    return candidates[salt % len(candidates)]
