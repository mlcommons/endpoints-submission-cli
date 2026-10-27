# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The message catalog and the code that uses it must agree.

Every result names a rule and a message key and passes the values the message needs
(see :mod:`submission_checker.messages`). These tests read the checker's source to
hold both sides together: each key the code uses exists, each call passes exactly
the placeholders its template needs, and the catalog carries nothing the code no
longer produces.

Validation errors are held to the same standard. The file models raise
:class:`~submission_checker.messages.Invalid` rather than ``ValueError``, and every
Pydantic error type their schemas can produce has a ``_pydantic`` message.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel
from pydantic_core._pydantic_core import list_all_errors

import submission_checker
from submission_checker import messages
from submission_checker.messages import PYDANTIC, SHARED, MessageCatalogError, placeholders
from submission_checker.models.file import (
    AccuracyResult,
    PointConfig,
    PointSummary,
    SystemDescription,
    SystemPower,
)
from submission_checker.models.loader import (
    load_accuracy_result,
    load_point_config,
    load_result_summary,
    load_system_description,
    load_system_power,
)

_SRC = Path(submission_checker.__file__).parent
_RESULT_FUNCS = {"ok", "warn", "err", "_ok", "_warn", "_err", "fragment", "Invalid"}
#: Catalog sections that no single rule's code produces.
_SECTIONS = {SHARED, PYDANTIC}

#: Calls whose rule is a variable, and the rules it can take. The coverage loop
#: picks one of three region rules; the loaders and the score helper report under
#: their caller's rule with a `_shared` message.
_DYNAMIC_RULES = {
    "pass": ["low-concurrency-coverage", "med-concurrency-coverage", "high-concurrency-coverage"],
    "fail": ["low-concurrency-coverage", "med-concurrency-coverage", "high-concurrency-coverage"],
}


@dataclass(frozen=True)
class _Call:
    where: str
    rule: str | None
    keys: tuple[str, ...]
    kwargs: frozenset[str] | None  # None when the call splats a dict


def _literal_keys(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.IfExp):
        body, orelse = _literal_keys(node.body), _literal_keys(node.orelse)
        if body is not None and orelse is not None:
            return body + orelse
    return None


def _calls() -> list[_Call]:
    found = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "messages.py" or path.parent.name == "models" and path.name == "results.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _RESULT_FUNCS
                and len(node.args) >= 2
            ):
                continue
            where = f"{path.relative_to(_SRC)}:{node.lineno}"
            rule_arg = node.args[0]
            rule = rule_arg.value if isinstance(rule_arg, ast.Constant) else None
            keys = _literal_keys(node.args[1])
            splat = any(kw.arg is None for kw in node.keywords)
            if keys is None and splat:
                # The loaders' dispatcher: rule, key and values all arrive as data
                # from `_load_json`/`_load_yaml`, whose keys are `_shared` messages.
                continue
            assert keys is not None, f"{where}: message key must be a literal (or a literal choice)"
            kwargs = None if splat else frozenset(kw.arg for kw in node.keywords if kw.arg)
            found.append(_Call(where, rule, keys, kwargs))
    return found


_CALLS = _calls()
_CATALOG = messages.catalog()


def _message(rule: str, key: str) -> messages.Message | None:
    entry = _CATALOG.get(rule)
    if entry is not None and key in entry.messages:
        return entry.messages[key]
    return _CATALOG[SHARED].messages.get(key)


def _needs(message: messages.Message) -> set[str]:
    return placeholders(message.text) | (placeholders(message.fix) if message.fix else set())


@pytest.mark.unit
class TestCatalogMatchesCode:
    def test_the_scan_found_the_call_sites(self) -> None:
        assert len(_CALLS) > 200  # guards against the scan silently matching nothing

    def test_every_key_used_exists(self) -> None:
        missing = []
        for call in _CALLS:
            rules = [call.rule] if call.rule else None
            for key in call.keys:
                candidates = rules or _DYNAMIC_RULES.get(key) or [SHARED]
                for rule in candidates:
                    if _message(rule, key) is None:
                        missing.append(f"{call.where}: {rule}.{key}")
        assert not missing, "keys missing from data/messages.yaml:\n" + "\n".join(missing)

    def test_every_call_passes_exactly_its_placeholders(self) -> None:
        wrong = []
        for call in _CALLS:
            if call.kwargs is None:
                continue  # values splatted from a dict; rendering is covered by strict mode
            rules = [call.rule] if call.rule else (_DYNAMIC_RULES.get(call.keys[0]) or [SHARED])
            for rule in rules:
                needed: set[str] = set()
                for key in call.keys:
                    message = _message(rule, key)
                    if message is None:
                        continue  # reported by test_every_key_used_exists
                    branch = _needs(message)
                    if not branch <= call.kwargs:
                        wrong.append(f"{call.where}: {rule}.{key} needs {sorted(branch)}")
                    needed |= branch
                extra = call.kwargs - needed
                if extra:
                    wrong.append(f"{call.where}: {rule} is passed unused {sorted(extra)}")
        assert not wrong, "\n".join(wrong)

    def test_no_message_is_unused(self) -> None:
        used = {(call.rule, key) for call in _CALLS if call.rule for key in call.keys}
        for key, rules in _DYNAMIC_RULES.items():
            used |= {(rule, key) for rule in rules}
        unused = [
            f"{rule}.{key}"
            for rule, entry in _CATALOG.items()
            if rule not in _SECTIONS
            for key in entry.messages
            if (rule, key) not in used
        ]
        assert not unused, "messages no code produces:\n" + "\n".join(unused)


@pytest.mark.unit
class TestCatalog:
    def test_every_rule_has_a_title_and_spec(self) -> None:
        for rule, entry in _CATALOG.items():
            if rule in _SECTIONS:
                continue
            assert entry.title and entry.title != rule, rule
            assert entry.spec.startswith(("§", "Submission Rules §")), (
                f"{rule}: spec {entry.spec!r}"
            )

    def test_specs_use_one_notation(self) -> None:
        for rule, entry in _CATALOG.items():
            for key, message in entry.messages.items():
                assert "#" not in message.spec, f"{rule}.{key}: {message.spec!r}"

    def test_specs_name_sections_that_exist(self) -> None:
        """Each spec cites a section of the Rules (§1–§9) or Submission Rules (§1–§11).

        A bare § is the Rules; the Submission Rules are named. The early checker
        numbered its rules by their row in the §9.1 table (§10 load pattern, §15
        accuracy, ...), which no reader could look up.
        """
        top_level = {"": 9, "Submission Rules ": 11}
        for rule, entry in _CATALOG.items():
            for key, message in entry.messages.items():
                for ref in filter(None, message.spec.split(", ")):
                    match = re.fullmatch(r"(Submission Rules )?§(\d+)(\.\d+)*", ref)
                    assert match, f"{rule}.{key}: {ref!r} is not a § citation"
                    assert 1 <= int(match[2]) <= top_level[match[1] or ""], (
                        f"{rule}.{key}: {ref!r} names no section"
                    )

    @pytest.mark.parametrize(
        ("rule", "key"),
        [(rule, key) for rule, entry in _CATALOG.items() for key in entry.messages],
    )
    def test_every_template_renders(self, rule: str, key: str) -> None:
        message = _CATALOG[rule].messages[key]
        values = dict.fromkeys(_needs(message), 1.5)
        rendered = messages.render(rule if rule != SHARED else "path-exists", key, values)
        assert rendered.text


@pytest.mark.unit
class TestRendering:
    def test_attribute_access_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "messages.yaml"
        bad.write_text("r:\n  title: R\n  messages:\n    k: '{a.b}'\n")
        with pytest.raises(MessageCatalogError, match="plain names"):
            messages.load(bad)

    def test_a_missing_value_raises_in_strict_mode(self) -> None:
        with pytest.raises(MessageCatalogError):
            messages.render("model-name-valid", "fail", {})

    def test_outside_strict_mode_a_bad_message_degrades(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(messages, "strict", False)
        rendered = messages.render("model-name-valid", "no-such-key", {"name": "x"})
        assert rendered.text == "model-name-valid (no-such-key): name='x'"
        assert rendered.title == "Benchmark model name"


#: The models each loader validates, and the Pydantic error types each kind of core
#: schema node can produce. A node kind missing from both maps fails the test, so a
#: new field type is classified before it reaches a submitter.
_FILE_MODELS: list[type[BaseModel]] = [
    AccuracyResult,
    PointConfig,
    PointSummary,
    SystemDescription,
    SystemPower,
]
_NODE_ERRORS = {
    "model-field": {"missing"},
    "model": {"model_type"},
    "model-fields": {"model_attributes_type"},
    "str": {"string_type"},
    "int": {"int_type", "int_parsing", "int_from_float"},
    "float": {"float_type", "float_parsing"},
    "bool": {"bool_type", "bool_parsing"},
    "literal": {"literal_error"},
    "enum": {"enum"},
    "dict": {"dict_type"},
    "list": {"list_type"},
}
_SILENT_NODES = {
    "any",
    "computed-field",
    "default",
    "definition-ref",
    "definitions",
    "function-after",
    "function-before",
    "no-info",
    "nullable",
    "union",
    "with-info",
}
_CONSTRAINT_ERRORS = {
    "gt": "greater_than",
    "ge": "greater_than_equal",
    "lt": "less_than",
    "le": "less_than_equal",
}


def _reachable_errors(schema: object, found: set[str]) -> None:
    if isinstance(schema, list | tuple):
        for item in schema:
            _reachable_errors(item, found)
        return
    if not isinstance(schema, dict):
        return
    kind = schema.get("type")
    if isinstance(kind, str):
        assert kind in _NODE_ERRORS or kind in _SILENT_NODES, f"unclassified schema node {kind!r}"
        found |= _NODE_ERRORS.get(kind, set())
        found |= {error for name, error in _CONSTRAINT_ERRORS.items() if name in schema}
        if "min_length" in schema:
            found.add("string_too_short" if kind == "str" else "too_short")
        if "max_length" in schema:
            found.add("string_too_long" if kind == "str" else "too_long")
        if "pattern" in schema:
            found.add("string_pattern_mismatch")
        if schema.get("allow_inf_nan") is False:
            found.add("finite_number")
        if (schema.get("config") or {}).get("extra_fields_behavior") == "forbid":
            found.add("extra_forbidden")
    for name, value in schema.items():
        if name not in {"serialization", "metadata"}:  # output and annotations, not validation
            _reachable_errors(value, found)


@pytest.mark.unit
class TestValidationWording:
    def test_models_raise_invalid_not_value_error(self) -> None:
        bare = []
        for path in sorted((_SRC / "models").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id in {"ValueError", "AssertionError", "TypeError"}
                ):
                    bare.append(f"{path.relative_to(_SRC)}:{node.lineno}")
        assert not bare, "raise Invalid(rule, key, ...) instead:\n" + "\n".join(bare)

    def test_pydantic_messages_match_the_reachable_errors(self) -> None:
        found: set[str] = set()
        for model in _FILE_MODELS:
            _reachable_errors(model.__pydantic_core_schema__, found)
        assert "missing" in found  # guards against the walk matching nothing
        worded = set(_CATALOG[PYDANTIC].messages)
        assert not found - worded, f"add {PYDANTIC} messages for: {sorted(found - worded)}"
        assert not worded - found, f"no file model can raise: {sorted(worded - found)}"

    def test_pydantic_messages_use_only_the_error_context(self) -> None:
        contexts = {e["type"]: set(e["example_context"] or {}) for e in list_all_errors()}
        for key, message in _CATALOG[PYDANTIC].messages.items():
            assert key in contexts, f"{key} is not a Pydantic error type"
            extra = _needs(message) - contexts[key] - {"input"}
            assert not extra, f"{PYDANTIC}.{key} uses {sorted(extra)}"


_BAD_FILES = [
    pytest.param(
        load_system_description,
        "system_desc.json",
        {
            "division": "open",
            "publication_status": "available",
            "system_availability_status": "preview",
        },
        "system_desc.json: the availability fields disagree"
        " (publication_status='available', system_availability_status='preview')",
        id="model-level-invalid",
    ),
    pytest.param(
        load_system_power,
        "system_power.json",
        {
            "node_sets": [
                {"published_power": {"value_w": 1, "source_type": "x", "source": "y", "bogus": 1}}
            ]
        },
        "system_power.json, field `node_sets.0.published_power.bogus`: not a recognised field",
        id="extra-forbidden",
    ),
    pytest.param(
        load_point_config,
        "point.yaml",
        {"warmup": {"requests_completed": 5, "requests_issued": "two"}},
        "point.yaml, field `warmup.requests_issued`: expected a whole number, got 'two'",
        id="int-parsing",
    ),
    pytest.param(
        load_result_summary,
        "results.json",
        {"ttft": {"percentiles": {"p99": 1.0}}},
        "results.json, field `ttft.percentiles`: 'p99' is not a percentile between 0 and 100",
        id="field-level-invalid",
    ),
    pytest.param(
        load_accuracy_result,
        "accuracy.json",
        {"accuracy_scores": [{"dataset_name": "a", "score": 1}, {"dataset_name": "a", "score": 1}]},
        "accuracy.json: dataset 'a' appears more than once",
        id="before-validator-invalid",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("loader", "name", "content", "expected"), _BAD_FILES)
def test_validation_errors_are_worded_by_the_catalog(
    tmp_path: Path, loader, name: str, content: dict, expected: str
) -> None:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content) if name.endswith(".yaml") else json.dumps(content))
    model, results = loader(path)
    assert model is None
    assert expected in [r.message for r in results]
    for result in results:  # strict mode has rendered every one from the catalog
        assert "Value error" not in result.message
        assert "Input should" not in result.message
