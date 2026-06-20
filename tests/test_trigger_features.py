from trigger_features import (
    question_word_type, char_spans_to_word_offsets, first_ne_position,
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
