from benchmarks import hotpot_to_example


def _fake_row():
    return {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
        "context": {
            "title": ["Ed Wood", "Scott Derrickson", "Distractor A", "Distractor B"],
            "sentences": [
                ["Ed Wood is a 1994 film.", " It starred Johnny Depp."],
                ["Scott Derrickson is an American director."],
                ["Unrelated text one."],
                ["Unrelated text two."],
            ],
        },
    }


def test_hotpot_to_example_maps_contract():
    ex = hotpot_to_example(_fake_row(), split=0)
    assert ex.interaction_id == "5a8b57f25542995d1e6f1371"
    assert ex.query.startswith("Were Scott")
    assert ex.answer == "yes"
    assert ex.question_type == "comparison"
    assert ex.split == 0
    # one passage per context paragraph, in order
    assert len(ex.passages) == 4
    assert ex.passages[0] == "Ed Wood is a 1994 film.  It starred Johnny Depp."
    # gold = indices of the two supporting-fact paragraphs (by title)
    assert ex.gold == {0, 1}
    # gold is the SHIPPED label, not string-matched: "yes" is not in the gold passages
    assert all("yes" not in ex.passages[i].lower() for i in ex.gold)


def test_hotpot_to_example_groundable_property():
    ex = hotpot_to_example(_fake_row(), split=0)
    assert ex.groundable is True  # non-empty gold
