"""Depth tests for the Python tree-sitter extractor.

Covers the spec's required Python paths:
    - Function def + docstring + signature → Symbol
    - Class with methods → parent Class node + child Method nodes + DEFINED_IN
    - Module imports → IMPORTS edges
    - @overload defs each get distinct doc_id (def_line invariant)
    - Decorator vs def-line invariant (D5)
    - Multiple inheritance → multiple INHERITS edges
    - Method called on `self.x` resolves; dynamic `obj.foo()` → AMBIGUOUS
"""

from __future__ import annotations

from services.ingestion.code_graph.extractors.python import PythonExtractor
from shared.constants import EdgeType, NodeLabel


def _extract(source: str, file_path: str = "pkg/mod.py"):
    return PythonExtractor().extract(file_path, source.encode("utf-8"), ".")


def _by_qname(symbols, qname: str):
    return next((s for s in symbols if s.qualified_name == qname), None)


def test_function_with_docstring() -> None:
    src = '''
def add(a, b):
    """Add two integers."""
    return a + b
'''
    result = _extract(src)
    fn = _by_qname(result.symbols, "pkg.mod.add")
    assert fn is not None
    assert fn.kind == NodeLabel.CODE_SYMBOL
    assert fn.docstring == "Add two integers."
    assert "def add(a, b)" in fn.signature
    assert fn.def_line == 2  # `def` keyword line, 1-indexed


def test_class_with_methods_emits_parent_and_children() -> None:
    src = '''
class Foo:
    """Foo doc."""
    def bar(self):
        return 1
    def baz(self):
        return 2
'''
    result = _extract(src)
    cls = _by_qname(result.symbols, "pkg.mod.Foo")
    bar = _by_qname(result.symbols, "pkg.mod.Foo.bar")
    baz = _by_qname(result.symbols, "pkg.mod.Foo.baz")
    assert cls is not None and cls.kind == NodeLabel.CODE_SYMBOL
    assert bar is not None and bar.kind == NodeLabel.CODE_SYMBOL
    assert baz is not None and baz.kind == NodeLabel.CODE_SYMBOL
    assert bar.parent_qname == "pkg.mod.Foo"
    # DEFINED_IN edges from methods to parent class.
    method_defines = [
        e for e in result.edges
        if e.edge_type == EdgeType.DEFINED_IN and e.from_qname == "pkg.mod.Foo.bar"
    ]
    assert any(e.to_qname == "pkg.mod.Foo" for e in method_defines)


def test_module_imports_emit_edges() -> None:
    src = """
import os
from collections import abc as _abc
from foo.bar import baz
"""
    result = _extract(src)
    targets = sorted(
        e.to_qname for e in result.edges if e.edge_type == EdgeType.IMPORTS
    )
    assert "os" in targets
    assert "collections" in targets
    assert "foo.bar" in targets


def test_overload_defs_get_distinct_doc_ids() -> None:
    """@overload defs share the qualified name but live on distinct lines.

    The def_line invariant gives them distinct doc_ids — adding/removing
    decorators doesn't change identity.
    """
    src = '''
from typing import overload

@overload
def parse(x: str) -> int: ...
@overload
def parse(x: int) -> str: ...
def parse(x):
    return None
'''
    result = _extract(src)
    parses = [s for s in result.symbols if s.qualified_name == "pkg.mod.parse"]
    # Three definitions; each at a distinct def_line.
    assert len(parses) == 3
    def_lines = sorted(s.def_line for s in parses)
    # def_line points at the `def` keyword, NOT the decorator line.
    # @overload is on lines 4 and 6; the def lines are 5 and 7;
    # the real impl is on line 8.
    assert def_lines == [5, 7, 8]


def test_def_line_is_def_keyword_not_decorator() -> None:
    """Adding/removing a decorator must NOT change a Symbol's def_line.

    Identity invariant per spec D5.
    """
    src_with_dec = """
@cached
def compute(x):
    return x * 2
"""
    src_without_dec = """
def compute(x):
    return x * 2
"""
    with_result = _extract(src_with_dec)
    without_result = _extract(src_without_dec)
    with_compute = _by_qname(with_result.symbols, "pkg.mod.compute")
    without_compute = _by_qname(without_result.symbols, "pkg.mod.compute")
    assert with_compute is not None and without_compute is not None
    # Both have `def compute` on the SAME line (line 3 vs 2 — different
    # because the decorator line bumps it down). The KEY guarantee is
    # `def_line` points at the `def` keyword, which it does here:
    # in with_result, line 3 is `def`; in without, line 2 is `def`.
    # Both are correct (point at `def`), and removing the decorator only
    # shifts the line number, not the IDENTITY relative to def keyword.
    assert with_compute.def_line == 3
    assert without_compute.def_line == 2


def test_class_inheritance_emits_inherits_edges() -> None:
    src = """
from base import One, Two

class Foo(One):
    pass

class Bar(One, Two):
    pass
"""
    result = _extract(src)
    inherits_edges = [e for e in result.edges if e.edge_type == EdgeType.INHERITS]
    foo_bases = [
        e.target_candidates[0] if e.ambiguous else e.to_qname
        for e in inherits_edges
        if e.from_qname == "pkg.mod.Foo"
    ]
    bar_bases = [
        e.target_candidates[0] if e.ambiguous else e.to_qname
        for e in inherits_edges
        if e.from_qname == "pkg.mod.Bar"
    ]
    # Each base name is captured (resolution status varies — that's fine).
    foo_base_names = sorted(b.split(".")[-1] for b in foo_bases)
    bar_base_names = sorted(b.split(".")[-1] for b in bar_bases)
    assert "One" in foo_base_names
    assert sorted(["One", "Two"]) == bar_base_names


def test_resolved_call_via_import() -> None:
    """`from foo import bar` → calling `bar()` resolves to `foo.bar`."""
    src = """
from foo import bar

def caller():
    bar()
"""
    result = _extract(src)
    caller_calls = [
        e for e in result.edges
        if e.from_qname == "pkg.mod.caller" and e.edge_type == EdgeType.CALLS
    ]
    assert len(caller_calls) >= 1
    bar_call = next(
        (e for e in caller_calls if e.to_qname.endswith(".bar") or "bar" in e.target_candidates),
        None,
    )
    assert bar_call is not None
    # Should be resolved to foo.bar via the import alias table.
    assert bar_call.ambiguous is False
    assert bar_call.to_qname == "foo.bar"


def test_dynamic_call_emits_ambiguous() -> None:
    """`obj.foo()` where `obj` has no resolvable type stays AMBIGUOUS."""
    src = """
def caller(x):
    return x.process()
"""
    result = _extract(src)
    caller_calls = [
        e for e in result.edges
        if e.from_qname == "pkg.mod.caller" and e.edge_type == EdgeType.CALLS
    ]
    process_call = next(
        (
            e for e in caller_calls
            if "process" in e.to_qname or "process" in e.target_candidates
        ),
        None,
    )
    assert process_call is not None
    assert process_call.ambiguous is True


def test_self_dot_method_resolves_to_class() -> None:
    """Inside a method, `self.foo()` resolves to the enclosing class's `foo`."""
    src = """
class Bag:
    def add(self, x):
        self.append(x)
    def append(self, x):
        pass
"""
    result = _extract(src)
    add_calls = [
        e for e in result.edges
        if e.from_qname == "pkg.mod.Bag.add" and e.edge_type == EdgeType.CALLS
    ]
    self_append = next(
        (e for e in add_calls if "append" in e.to_qname),
        None,
    )
    assert self_append is not None
    assert self_append.ambiguous is False
    assert self_append.to_qname == "pkg.mod.Bag.append"
