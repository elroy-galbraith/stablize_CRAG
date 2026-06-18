"""
Generate a tiny synthetic CRAG-format file (bz2 JSONL) matching the real schema,
so the pipeline can be tested without the ~GB download. NOT real CRAG data.
"""
import bz2
import json
import os


def page(name, html):
    return {
        "page_name": name,
        "page_url": f"https://example.com/{name.replace(' ', '_')}",
        "page_snippet": html[:120],
        "page_result": html,
        "page_last_modified": "2025-01-01",
    }


EXAMPLES = [
    {
        # Early-stabilizing: intent ("Sony Interactive / PlayStation maker") is
        # clear from an early prefix; answer "Hiroki Totoki" is in a page.
        "interaction_id": "ex-early",
        "query_time": "01/01/2026, 10:00:00 PT",
        "domain": "open",
        "question_type": "simple",
        "static_or_dynamic": "slow-changing",
        "query": "sony interactive entertainment playstation maker current president and chief executive",
        "answer": "Hiroki Totoki",
        "alt_ans": [],
        "split": 0,
        "popularity": "",
        "search_results": [
            page("SIE leadership", "<html><body><h1>Sony Interactive Entertainment</h1>"
                 "<p>Hiroki Totoki became President and CEO of Sony Interactive Entertainment, "
                 "the maker of the PlayStation console, after his appointment in 2025.</p>"
                 "<script>var x=1;</script></body></html>"),
            page("Nintendo", "<html><body><p>Nintendo develops the Switch and franchises "
                 "such as Mario and Zelda. It is a Japanese company.</p></body></html>"),
            page("Xbox", "<html><body><p>Microsoft develops the Xbox family of consoles, "
                 "including the Series X and Series S.</p></body></html>"),
        ],
    },
    {
        # Late-stabilizing: decisive term ("2022 fifa world cup") arrives mid/late,
        # and the answer requires a hop (winner -> its capital).
        "interaction_id": "ex-late",
        "query_time": "01/01/2026, 10:00:00 PT",
        "domain": "sports",
        "question_type": "multi-hop",
        "static_or_dynamic": "static",
        "query": "what is the capital city of the country that won the 2022 fifa world cup",
        "answer": "Buenos Aires",
        "alt_ans": [],
        "split": 0,
        "popularity": "head",
        "search_results": [
            page("WC 2022", "<html><body><p>Argentina won the 2022 FIFA World Cup, defeating "
                 "France in the final. The capital of Argentina is Buenos Aires.</p></body></html>"),
            page("Capitals", "<html><body><p>Paris is the capital of France. Madrid is the "
                 "capital of Spain. London is the capital of the United Kingdom.</p></body></html>"),
            page("Football", "<html><body><p>The FIFA World Cup is held every four years. "
                 "Brazil has won it five times.</p></body></html>"),
        ],
    },
    {
        # False-premise / ungroundable: answer is "I don't know" -> excluded from grounding.
        "interaction_id": "ex-fp",
        "query_time": "01/01/2026, 10:00:00 PT",
        "domain": "music",
        "question_type": "false_premise",
        "static_or_dynamic": "static",
        "query": "which grammy did the fictional band the moon penguins win in 2023",
        "answer": "I don't know",
        "alt_ans": ["invalid question"],
        "split": 0,
        "popularity": "tail",
        "search_results": [
            page("Grammys", "<html><body><p>The Grammy Awards honor achievements in the "
                 "music industry. Winners vary by category and year.</p></body></html>"),
        ],
    },
    {
        # A public-test row (split=1) to verify split filtering excludes it.
        "interaction_id": "ex-test-split",
        "query_time": "01/01/2026, 10:00:00 PT",
        "domain": "finance",
        "question_type": "simple",
        "static_or_dynamic": "fast-changing",
        "query": "what is the stock ticker for apple inc",
        "answer": "AAPL",
        "alt_ans": [],
        "split": 1,
        "popularity": "head",
        "search_results": [
            page("Apple", "<html><body><p>Apple Inc. trades under the ticker AAPL on Nasdaq.</p></body></html>"),
        ],
    },
]


def main(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "crag_fixture.jsonl.bz2")
    with bz2.open(path, "wt", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(EXAMPLES)} examples -> {path}")


if __name__ == "__main__":
    main()
