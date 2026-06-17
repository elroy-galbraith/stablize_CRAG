"""
CRAG loader, HTML cleaning, passage chunking, and answer-grounding.

Schema (confirmed from facebookresearch/CRAG docs/dataset.md, Task 1 & 2):
  interaction_id, query_time, domain, question_type, static_or_dynamic,
  query, answer, alt_ans (list), split (0=val, 1=public test), popularity,
  search_results: [{page_name, page_url, page_snippet, page_result(HTML),
                    page_last_modified}, ... up to 5 for Task 1]

There is NO gold-passage label. The gold *answer* string(s) are grounded into
the page text to derive which passages are answer-bearing (d*). This is a
methodological choice; see `gold_passage_ids`.
"""
from __future__ import annotations

import bz2
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterator, Optional, Callable


# --------------------------------------------------------------------------- #
# JSONL(.bz2) reading                                                          #
# --------------------------------------------------------------------------- #
def iter_jsonl(path: str) -> Iterator[dict]:
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# HTML -> text (BeautifulSoup if present, stdlib fallback otherwise)           #
# --------------------------------------------------------------------------- #
class _StdlibText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "head", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            t = data.strip()
            if t:
                self.parts.append(t)


def html_to_text(html: str) -> str:
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup  # optional, gives cleaner text

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "head", "template"]):
            tag.decompose()
        return " ".join(soup.get_text(" ").split())
    except Exception:
        p = _StdlibText()
        try:
            p.feed(html)
        except Exception:
            pass
        return " ".join(" ".join(p.parts).split())


# --------------------------------------------------------------------------- #
# Passage chunking                                                             #
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str, chunk_words: int = 120, overlap: int = 20, max_chunks: int = 400
) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    out: list[str] = []
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + chunk_words]))
        if len(out) >= max_chunks:
            break
    return out


# --------------------------------------------------------------------------- #
# Answer grounding -> gold passage ids (derives d*)                            #
# --------------------------------------------------------------------------- #
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
# Answers that signal an unanswerable / false-premise item — not groundable.
_NON_ANSWERS = {"", "i don't know", "i dont know", "invalid question", "unknown"}


def _norm(s) -> str:
    # CRAG answers/alt_ans are sometimes numeric (int/float, e.g. aggregation
    # counts or a false-premise "0"). Coerce explicitly: `s or ""` would wrongly
    # blank out a valid numeric 0.
    s = "" if s is None else str(s)
    return _WS.sub(" ", _NON_ALNUM.sub(" ", s.lower())).strip()


def gold_passage_ids(
    answer: str,
    alt_ans: Optional[list],
    passages: list[str],
    min_tokens_for_substring: int = 2,
    llm_judge: Optional[Callable[[str, str], bool]] = None,
) -> set[int]:
    """Return indices of passages judged to contain the gold answer.

    Multi-token answers: normalized substring match. Single-token answers
    (e.g. a number or a name): word-boundary match to avoid spurious hits.
    `llm_judge(answer, passage) -> bool` is an optional fallback used only when
    string matching finds nothing; it is never called by default.
    """
    candidates = [answer, *(alt_ans or [])]
    norm_answers = [na for na in (_norm(a) for a in candidates) if na and na not in _NON_ANSWERS]
    if not norm_answers:
        return set()  # ungroundable (e.g. false-premise / "I don't know")

    norm_passages = [_norm(p) for p in passages]
    gold: set[int] = set()
    for na in norm_answers:
        toks = na.split()
        if len(toks) >= min_tokens_for_substring:
            for i, npas in enumerate(norm_passages):
                if na in npas:
                    gold.add(i)
        else:
            pat = re.compile(r"\b" + re.escape(na) + r"\b")
            for i, npas in enumerate(norm_passages):
                if pat.search(npas):
                    gold.add(i)

    if not gold and llm_judge is not None:
        for i, p in enumerate(passages):
            if any(llm_judge(a, p) for a in candidates if a):
                gold.add(i)
    return gold


# --------------------------------------------------------------------------- #
# Example container + loader                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class CragExample:
    interaction_id: str
    query: str
    answer: str
    alt_ans: list
    domain: str
    question_type: str
    static_or_dynamic: str
    split: int
    passages: list[str] = field(default_factory=list)
    gold: set[int] = field(default_factory=set)

    @property
    def groundable(self) -> bool:
        return len(self.gold) > 0


def load_crag(
    path: str,
    split: Optional[int] = 0,
    limit: Optional[int] = None,
    chunk_words: int = 120,
    overlap: int = 20,
    max_chunks_per_q: int = 400,
    llm_judge: Optional[Callable[[str, str], bool]] = None,
) -> Iterator[CragExample]:
    """Stream CragExamples. `split=0` keeps validation; None keeps all."""
    n = 0
    for row in iter_jsonl(path):
        if split is not None and int(row.get("split", 0)) != split:
            continue
        passages: list[str] = []
        for sr in row.get("search_results", []) or []:
            text = html_to_text(sr.get("page_result", "") or "")
            if not text:
                snip = sr.get("page_snippet", "") or ""
                text = snip
            passages.extend(chunk_text(text, chunk_words, overlap, max_chunks_per_q))
            if len(passages) >= max_chunks_per_q:
                passages = passages[:max_chunks_per_q]
                break
        gold = gold_passage_ids(row.get("answer", ""), row.get("alt_ans"), passages, llm_judge=llm_judge)
        yield CragExample(
            interaction_id=row.get("interaction_id", str(n)),
            query=row.get("query", ""),
            answer=row.get("answer", ""),
            alt_ans=row.get("alt_ans") or [],
            domain=row.get("domain", ""),
            question_type=row.get("question_type", ""),
            static_or_dynamic=row.get("static_or_dynamic", ""),
            split=int(row.get("split", 0)),
            passages=passages,
            gold=gold,
        )
        n += 1
        if limit and n >= limit:
            break
