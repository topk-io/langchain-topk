"""Translate LangChain-style metadata filters into TopK filter expressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any, ClassVar

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from topk_sdk.data import f64_list
from topk_sdk.query import LogicalExpr, field, match, not_

_COMPARISON_OPERATORS = {
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
    "$contains",
    "$starts_with",
    "$exists",
}
_LOGICAL_OPERATORS = {"$and", "$or", "$not"}


# `TextExpr` (returned by `match()` and `should()`) is not exported by
# `topk_sdk.query`, so look it up from an instance.
_TextExpr: type = next(
    (c for c in type(match("_")).__mro__ if c.__name__ == "TextExpr"),
    type(match("_")),
)


def build_filter(
    filter: Any, metadata_field: str, *, allow_text: bool = True
) -> Any | None:
    """Build a TopK filter expression from a LangChain filter.

    Accepts:

    * `None`: no filter.
    * A TopK expression (for example `field("metadata.year") > 2020` or
      `match("rust")`), which is passed through unchanged.
    * A dict in the MongoDB-like syntax used by many LangChain vector stores,
      for example `{"genre": "sci-fi", "year": {"$gte": 2020}}` or
      `{"$or": [{"genre": "sci-fi"}, {"genre": "fantasy"}]}`.

    Dict keys refer to metadata keys. They are resolved inside the metadata
    struct field, so `{"year": 2020}` becomes `field("metadata.year") == 2020`.

    Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`,
    `$nin`, `$contains`, `$starts_with`, `$exists`, `$and`, `$or`, `$not`.

    Args:
        filter: The filter to translate.
        metadata_field: Name of the struct field that stores document metadata.
        allow_text: Whether to accept text expressions such as `match()`.
            Deletes accept only logical expressions.

    Returns:
        A TopK expression, or `None` if there is no filter.

    Raises:
        ValueError: If the filter uses an unknown operator or malformed value.
    """
    if filter is None:
        return None
    if isinstance(filter, Mapping):
        if not filter:
            return None
        return _translate_dict(filter, metadata_field)
    if isinstance(filter, LogicalExpr) or (
        allow_text and isinstance(filter, _TextExpr)
    ):
        return filter
    msg = (
        "`filter` must be a dict (for example {'genre': 'sci-fi'}) or a TopK "
        f"expression from `topk_sdk.query`, got {type(filter).__name__}."
    )
    raise ValueError(msg)


def _path(key: str, metadata_field: str) -> str:
    if not isinstance(key, str) or not key:
        msg = f"Filter keys must be non-empty strings, got {key!r}."
        raise ValueError(msg)
    return f"{metadata_field}.{key}"


def _combine_all(exprs: Sequence[Any]) -> Any:
    # Chain binary `and_` rather than `all()`, so that `repr()` of the
    # resulting expression shows every clause (useful when debugging).
    return reduce(lambda left, right: left.and_(right), exprs)


def _combine_any(exprs: Sequence[Any]) -> Any:
    return reduce(lambda left, right: left.or_(right), exprs)


def _translate_dict(filter: Mapping[str, Any], metadata_field: str) -> Any:
    clauses: list[Any] = []
    for key, value in filter.items():
        if key in _LOGICAL_OPERATORS:
            clauses.append(_translate_logical(key, value, metadata_field))
        elif key.startswith("$"):
            msg = (
                f"Unsupported filter operator {key!r} at the top level. Logical "
                f"operators are {sorted(_LOGICAL_OPERATORS)}."
            )
            raise ValueError(msg)
        else:
            clauses.append(_translate_field(key, value, metadata_field))
    return _combine_all(clauses)


def _translate_logical(op: str, value: Any, metadata_field: str) -> Any:
    if op == "$not":
        if not isinstance(value, Mapping):
            msg = "`$not` expects a single filter dict."
            raise ValueError(msg)
        return not_(_translate_dict(value, metadata_field))
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        msg = f"`{op}` expects a non-empty list of filter dicts."
        raise ValueError(msg)
    exprs = [_translate_dict(v, metadata_field) for v in value]
    return _combine_all(exprs) if op == "$and" else _combine_any(exprs)


def _translate_field(key: str, value: Any, metadata_field: str) -> Any:
    path = _path(key, metadata_field)
    if isinstance(value, Mapping):
        if not value:
            msg = f"Empty operator dict for filter key {key!r}."
            raise ValueError(msg)
        clauses = []
        for op, operand in value.items():
            if op not in _COMPARISON_OPERATORS:
                msg = (
                    f"Unsupported filter operator {op!r} for key {key!r}. "
                    f"Supported operators: {sorted(_COMPARISON_OPERATORS)}."
                )
                raise ValueError(msg)
            clauses.append(_compare(path, op, operand))
        return _combine_all(clauses)
    if isinstance(value, (list, tuple, set)):
        msg = (
            f"Filter value for {key!r} is a list. Use {{'$in': [...]}} to match "
            "any of several values, or {'$contains': value} to match a list field."
        )
        raise ValueError(msg)
    return _compare(path, "$eq", value)


def _normalize_list(values: Sequence[Any]) -> Any:
    """Prepare an `$in` / `$nin` list for TopK.

    A plain Python list of floats is sent as 32-bit floats, so values such as
    `9.99` no longer equal the stored 64-bit value and silently match nothing.
    Lists that contain any float are therefore sent as 64-bit floats (ints are
    widened to float).
    """
    items = list(values)
    numeric = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in items
    )
    if numeric and any(isinstance(v, float) for v in items):
        return f64_list([float(v) for v in items])
    return items


def _compare(path: str, op: str, value: Any) -> LogicalExpr:
    try:
        return _compare_unchecked(path, op, value)
    except TypeError as e:
        msg = f"Invalid value {value!r} for operator {op!r} on {path!r}: {e}"
        raise ValueError(msg) from e


def _compare_unchecked(path: str, op: str, value: Any) -> LogicalExpr:
    f = field(path)
    if op == "$eq":
        return f.eq(value)
    if op == "$ne":
        return f.ne(value)
    if op == "$gt":
        return f.gt(value)
    if op == "$gte":
        return f.gte(value)
    if op == "$lt":
        return f.lt(value)
    if op == "$lte":
        return f.lte(value)
    if op in ("$in", "$nin"):
        if isinstance(value, str) or not isinstance(value, Sequence) or not value:
            msg = f"`{op}` expects a non-empty list, got {value!r}."
            raise ValueError(msg)
        expr = f.in_(_normalize_list(value))
        return expr if op == "$in" else not_(expr)
    if op == "$contains":
        return f.contains(value)
    if op == "$starts_with":
        if not isinstance(value, str):
            msg = f"`$starts_with` expects a string, got {value!r}."
            raise ValueError(msg)
        return f.starts_with(value)
    if op == "$exists":
        return f.is_not_null() if value else f.is_null()
    msg = f"Unsupported filter operator {op!r}."  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


class TopKTranslator(Visitor):
    """Translate LangChain structured queries into TopK filters.

    Use it with a self-querying retriever so that an LLM can turn a natural
    language question into a semantic query plus a TopK metadata filter:

    ```python
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain_topk import TopKTranslator

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_contents="Brief summary of a movie",
        metadata_field_info=metadata_field_info,
        structured_query_translator=TopKTranslator(),
    )
    ```

    The translator produces the dict filter syntax accepted by
    `TopKVectorStore`, so the result can also be inspected or logged.
    """

    allowed_operators = (Operator.AND, Operator.OR, Operator.NOT)
    allowed_comparators = (
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.CONTAIN,
        Comparator.IN,
        Comparator.NIN,
    )

    _COMPARATOR_MAP: ClassVar[dict[Comparator, str]] = {
        Comparator.EQ: "$eq",
        Comparator.NE: "$ne",
        Comparator.GT: "$gt",
        Comparator.GTE: "$gte",
        Comparator.LT: "$lt",
        Comparator.LTE: "$lte",
        Comparator.CONTAIN: "$contains",
        Comparator.IN: "$in",
        Comparator.NIN: "$nin",
    }

    def visit_operation(self, operation: Operation) -> dict[str, Any]:
        """Translate an `Operation` (and, or, not)."""
        self._validate_func(operation.operator)
        args = [arg.accept(self) for arg in operation.arguments]
        if operation.operator == Operator.NOT:
            if len(args) != 1:
                msg = "NOT expects exactly one argument."
                raise ValueError(msg)
            return {"$not": args[0]}
        if len(args) == 1:
            return args[0]
        key = "$and" if operation.operator == Operator.AND else "$or"
        return {key: args}

    def visit_comparison(self, comparison: Comparison) -> dict[str, Any]:
        """Translate a `Comparison`."""
        self._validate_func(comparison.comparator)
        value = comparison.value
        if isinstance(value, dict) and value.get("type") in ("date", "datetime"):
            # Dates arrive as {"date": "YYYY-MM-DD", "type": "date"} and
            # datetimes as {"datetime": "...", "type": "datetime"}. Compare them
            # as ISO 8601 strings, which sort chronologically.
            value = value[value["type"]]
        op = self._COMPARATOR_MAP[comparison.comparator]
        if op in ("$in", "$nin") and not isinstance(value, list):
            value = [value]
        return {comparison.attribute: {op: value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> tuple[str, dict[str, Any]]:
        """Translate a `StructuredQuery` into a query and search kwargs."""
        if structured_query.filter is None:
            return structured_query.query, {}
        return structured_query.query, {"filter": structured_query.filter.accept(self)}
