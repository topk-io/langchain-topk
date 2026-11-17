from __future__ import annotations

import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from topk_sdk.query import field, match, select

from langchain_topk import TopKTranslator
from langchain_topk._filters import build_filter


def r(value: object) -> str:
    # n-ary expressions hide their children in repr(); a query shows them.
    return repr(select("x").filter(value))  # type: ignore[arg-type]


def test_none_and_empty() -> None:
    assert build_filter(None, "metadata") is None
    assert build_filter({}, "metadata") is None


def test_equality_shorthand() -> None:
    assert r(build_filter({"genre": "sci-fi"}, "metadata")) == r(
        field("metadata.genre") == "sci-fi"
    )


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("$eq", field("metadata.year").eq(2020)),
        ("$ne", field("metadata.year").ne(2020)),
        ("$gt", field("metadata.year").gt(2020)),
        ("$gte", field("metadata.year").gte(2020)),
        ("$lt", field("metadata.year").lt(2020)),
        ("$lte", field("metadata.year").lte(2020)),
    ],
)
def test_comparisons(op: str, expected: object) -> None:
    assert r(build_filter({"year": {op: 2020}}, "metadata")) == r(expected)


def test_in_nin_contains_starts_with_exists() -> None:
    assert "In" in r(build_filter({"g": {"$in": ["a", "b"]}}, "m"))
    nin = r(build_filter({"g": {"$nin": ["a"]}}, "m"))
    assert "Not" in nin and "In" in nin
    assert "Contains" in r(build_filter({"tags": {"$contains": "x"}}, "m"))
    assert "StartsWith" in r(build_filter({"t": {"$starts_with": "x"}}, "m"))
    assert "IsNotNull" in r(build_filter({"t": {"$exists": True}}, "m"))
    assert "IsNull" in r(build_filter({"t": {"$exists": False}}, "m"))


def test_logical_operators() -> None:
    expr = r(
        build_filter(
            {
                "$or": [
                    {"genre": "sci-fi"},
                    {"$and": [{"year": {"$gte": 2000}}, {"rating": {"$gt": 8}}]},
                ],
                "$not": {"genre": "horror"},
            },
            "metadata",
        )
    )
    for part in ("metadata.genre", "metadata.year", "metadata.rating", "Not"):
        assert part in expr


def test_multiple_keys_are_anded() -> None:
    expr = r(build_filter({"a": 1, "b": {"$gt": 1, "$lt": 5}}, "m"))
    assert expr.count("field(m.") == 3


def test_topk_expression_passthrough() -> None:
    expr = field("metadata.year") > 2020
    assert build_filter(expr, "metadata") is expr
    text = match("rust")
    assert build_filter(text, "metadata") is text


@pytest.mark.parametrize(
    "bad",
    [
        {"$xor": []},
        {"a": {"$regex": "x"}},
        {"a": {}},
        {"a": {"$in": "abc"}},
        {"a": {"$in": []}},
        {"$or": []},
        {"$not": [{"a": 1}]},
        {"a": {"$starts_with": 3}},
        "genre == 'x'",
        {"lang": ["a", "b"]},
        {"lang": {"$eq": ["a", "b"]}},
    ],
)
def test_invalid_filters(bad: object) -> None:
    with pytest.raises(ValueError):
        build_filter(bad, "metadata")


def test_translator_comparison_and_operation() -> None:
    t = TopKTranslator()
    query, kwargs = t.visit_structured_query(
        StructuredQuery(
            query="space adventure",
            filter=Operation(
                operator=Operator.AND,
                arguments=[
                    Comparison(comparator=Comparator.GTE, attribute="year", value=1990),
                    Comparison(comparator=Comparator.IN, attribute="genre", value="sf"),
                    Operation(
                        operator=Operator.NOT,
                        arguments=[
                            Comparison(
                                comparator=Comparator.EQ,
                                attribute="director",
                                value="X",
                            )
                        ],
                    ),
                ],
            ),
            limit=None,
        )
    )
    assert query == "space adventure"
    assert kwargs == {
        "filter": {
            "$and": [
                {"year": {"$gte": 1990}},
                {"genre": {"$in": ["sf"]}},
                {"$not": {"director": {"$eq": "X"}}},
            ]
        }
    }
    # The translated filter is valid input for the vector store.
    assert build_filter(kwargs["filter"], "metadata") is not None


def test_translator_dates_and_no_filter() -> None:
    t = TopKTranslator()
    out = t.visit_comparison(
        Comparison(
            comparator=Comparator.LT,
            attribute="released",
            value={"date": "2020-01-01", "type": "date"},
        )
    )
    assert out == {"released": {"$lt": "2020-01-01"}}
    assert t.visit_structured_query(StructuredQuery(query="q", filter=None)) == (
        "q",
        {},
    )


def test_float_in_lists_are_sent_as_f64() -> None:
    # f32 would turn 9.99 into 9.989999771 and silently match nothing.
    for values in ([9.99, 2.5], [9.99, 3]):
        for op in ("$in", "$nin"):
            assert "F64" in r(build_filter({"x": {op: values}}, "m"))
    assert "F32" not in r(build_filter({"x": {"$in": [9.99]}}, "m"))
    assert "F64" not in r(build_filter({"x": {"$in": [1, 2]}}, "m"))
    assert "F64" not in r(build_filter({"x": {"$in": ["a", "b"]}}, "m"))


def test_delete_filters_reject_text_expressions() -> None:
    with pytest.raises(ValueError):
        build_filter(match("rust"), "metadata", allow_text=False)


def test_translator_datetime() -> None:
    out = TopKTranslator().visit_comparison(
        Comparison(
            comparator=Comparator.GTE,
            attribute="updated",
            value={"datetime": "2024-01-01T00:00:00", "type": "datetime"},
        )
    )
    assert out == {"updated": {"$gte": "2024-01-01T00:00:00"}}
    assert build_filter(out, "metadata") is not None


def test_translator_rejects_like() -> None:
    with pytest.raises(ValueError, match="disallowed"):
        TopKTranslator().visit_comparison(
            Comparison(comparator=Comparator.LIKE, attribute="a", value="x")
        )
