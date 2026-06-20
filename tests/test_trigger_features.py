from trigger_features import (
    question_word_type, char_spans_to_word_offsets, first_ne_position,
    per_word_rows,
)


def test_question_word_type():
    assert question_word_type("who makes the playstation") == "who"
    assert question_word_type("Where is the Eiffel Tower") == "where"
    assert question_word_type("name the tallest mountain") == "other"
    assert question_word_type("") == "other"


def test_char_spans_to_word_offsets():
    q = "who makes the playstation console"   # word offsets: who=1 makes=2 the=3 playstation=4 console=5
    # "playstation" starts at char index 14
    assert q.index("playstation") == 14
    assert char_spans_to_word_offsets(q, [(14, 25)]) == [4]
    # two entities, unsorted input, dedup
    spans = [(q.index("console"), q.index("console") + 7), (14, 25)]
    assert char_spans_to_word_offsets(q, spans) == [4, 5]


def test_first_ne_position():
    assert first_ne_position([4, 5]) == 4
    assert first_ne_position([]) is None


import trigger_features as tf


def test_spacy_ner_uses_char_span_mapping(monkeypatch):
    # Stub the spaCy doc.ents so the test needs no model download.
    class _Ent:
        def __init__(self, start_char): self.start_char = start_char

    class _Doc:
        def __init__(self, ents): self.ents = ents

    q = "who founded microsoft corporation"
    def fake_nlp(text):
        return _Doc([_Ent(text.index("microsoft"))])
    monkeypatch.setattr(tf, "_load_nlp", lambda: fake_nlp)

    assert tf.spacy_ner(q) == [3]  # "microsoft" is the 3rd word


def test_per_word_rows_features_and_labels():
    # seq: top1 ids by prefix length 1..4; stabilizes on id 7 from t=3.
    seq = [(2, {2}), (5, {5, 2}), (7, {7, 5}), (7, {7, 9})]
    meta = {"interaction_id": "q1", "question_type": "simple",
            "domain": "music", "query": "who made the playstation"}
    rows = per_word_rows(meta, seq, n=4, t_suf=3, t_sc=3, ne_offsets=[4])
    assert len(rows) == 4
    # streak resets on change, grows while stable
    assert [r["top1_stable_streak"] for r in rows] == [1, 1, 1, 2]
    assert [r["top1_changed"] for r in rows] == [0, 1, 1, 0]
    # label = 1[t >= t_suf]; t_suf=3
    assert [r["label"] for r in rows] == [0, 0, 1, 1]
    assert [r["label_sc"] for r in rows] == [0, 0, 1, 1]
    # entity first appears at word 4 -> detected only at t=4
    assert [r["named_entity_detected"] for r in rows] == [0, 0, 0, 1]
    assert [r["words_since_first_ne"] for r in rows] == [0, 0, 0, 0]
    assert rows[0]["question_word_type"] == "who"
    assert rows[0]["retrieved_gold"] is True


def test_per_word_rows_ungroundable_blank_label():
    seq = [(2, {2}), (2, {2})]
    meta = {"interaction_id": "q2", "question_type": "false_premise",
            "domain": "x", "query": "is the moon cheese"}
    rows = per_word_rows(meta, seq, n=2, t_suf=None, t_sc=1, ne_offsets=[])
    assert [r["label"] for r in rows] == ["", ""]
    assert [r["label_sc"] for r in rows] == [1, 1]
    assert rows[0]["retrieved_gold"] is False


from trigger_features import extract, FEATURE_FIELDS


def test_extract_on_fixture_schema_and_counts():
    rows = extract("data/crag_fixture.jsonl.bz2", split=0, top_k=3,
                   ner_fn=lambda q: [])   # no-NER stub keeps it fast/deterministic
    assert rows, "fixture produced no rows"
    # one row per word position per question
    by_q = {}
    for r in rows:
        by_q.setdefault(r["interaction_id"], []).append(r)
    for q, qr in by_q.items():
        assert [r["t"] for r in qr] == list(range(1, qr[0]["n_words"] + 1))
    # every declared field is present
    for r in rows:
        assert set(FEATURE_FIELDS).issubset(r.keys())
