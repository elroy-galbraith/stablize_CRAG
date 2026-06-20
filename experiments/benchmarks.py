"""Per-benchmark loaders that emit the crag.CragExample contract, so the whole
stabilization/trigger pipeline runs unchanged. See
docs/superpowers/specs/2026-06-20-second-benchmark-generalization-design.md.
"""
from __future__ import annotations

from typing import Iterator, Optional

from crag import CragExample

_HOTPOT_SPLIT = {0: "validation", 1: "train"}


def hotpot_to_example(row: dict, split: int) -> CragExample:
    """Map one hotpotqa/hotpot_qa (distractor) row to the contract.
    1 context paragraph = 1 passage (paragraphs are already passage-sized), so
    passage index == paragraph index and gold = supporting-fact paragraph indices."""
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    passages = [" ".join(sents) for sents in sentences]
    gold_titles = set(row["supporting_facts"]["title"])
    gold = {i for i, t in enumerate(titles) if t in gold_titles}
    return CragExample(
        interaction_id=row["id"],
        query=row["question"],
        answer=row["answer"],
        alt_ans=[],
        domain="",
        question_type=row.get("type", ""),       # comparison | bridge
        static_or_dynamic=row.get("level", ""),   # easy | medium | hard
        split=split,
        passages=passages,
        gold=gold,
    )


def load_hotpotqa(split: int, limit: Optional[int] = None) -> Iterator[CragExample]:
    from datasets import load_dataset  # lazy: [bench] extra

    hf_split = _HOTPOT_SPLIT.get(split, "validation")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=hf_split, streaming=True)
    n = 0
    for row in ds:
        ex = hotpot_to_example(row, split)
        if not ex.passages or not ex.query.split() or not ex.gold:
            continue
        yield ex
        n += 1
        if limit and n >= limit:
            break


BENCHMARKS = {"hotpotqa": load_hotpotqa}
