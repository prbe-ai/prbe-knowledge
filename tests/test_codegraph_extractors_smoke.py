"""Smoke tests for the four non-Python extractors.

Per spec §4.5: PR-A ships TS/JS/Go/Java at smoke-level; deeper coverage
is fast-follow once a real customer repo exercises them.
"""

from __future__ import annotations

from services.ingestion.code_graph.extractors import get_extractor_for_file
from services.ingestion.code_graph.extractors.go import GoExtractor
from services.ingestion.code_graph.extractors.java import JavaExtractor
from services.ingestion.code_graph.extractors.javascript import JavaScriptExtractor
from services.ingestion.code_graph.extractors.typescript import TypeScriptExtractor
from shared.constants import EdgeType, NodeLabel


def test_typescript_extracts_function_class_and_imports() -> None:
    src = b'''
import { foo } from "./foo";

export function add(a: number, b: number): number {
    return a + b;
}

export class Calculator {
    add(a: number, b: number): number { return a + b; }
}
'''
    result = TypeScriptExtractor().extract("src/calc.ts", src, ".")
    qnames = {s.qualified_name for s in result.symbols}
    assert "src.calc" in qnames
    assert "src.calc.add" in qnames
    assert "src.calc.Calculator" in qnames
    assert "src.calc.Calculator.add" in qnames
    # IMPORTS edge present.
    assert any(e.edge_type == EdgeType.IMPORTS for e in result.edges)


def test_typescript_handles_tsx() -> None:
    src = b'''
import * as React from "react";

export function Hello({ name }: { name: string }) {
    return <div>Hello {name}</div>;
}
'''
    result = TypeScriptExtractor().extract("src/Hello.tsx", src, ".")
    assert any(s.qualified_name == "src.Hello.Hello" for s in result.symbols)


def test_typescript_extends_emits_inherits() -> None:
    src = b'''
export class Animal {}
export class Dog extends Animal { bark() {} }
'''
    result = TypeScriptExtractor().extract("src/zoo.ts", src, ".")
    inherits = [e for e in result.edges if e.edge_type == EdgeType.INHERITS]
    assert any("Animal" in (e.to_qname,) or "Animal" in e.target_candidates for e in inherits)


def test_javascript_extracts_function_and_class() -> None:
    src = b'''
import { foo } from "./foo";

export function add(a, b) { return a + b; }

export class Box {
    constructor() {}
    pack() {}
}
'''
    result = JavaScriptExtractor().extract("src/util.js", src, ".")
    qnames = {s.qualified_name for s in result.symbols}
    assert "src.util.add" in qnames
    assert "src.util.Box" in qnames
    assert "src.util.Box.pack" in qnames
    assert any(e.edge_type == EdgeType.IMPORTS for e in result.edges)


def test_go_extracts_func_method_struct_imports() -> None:
    src = b'''
package main

import (
    "fmt"
    "net/http"
)

type Service struct {
    Name string
}

func (s *Service) Greet() string {
    return fmt.Sprintf("hello %s", s.Name)
}

func main() {
    http.ListenAndServe(":8080", nil)
}
'''
    result = GoExtractor().extract("cmd/server/main.go", src, ".")
    qnames = {s.qualified_name for s in result.symbols}
    # Module qname.
    assert any("cmd.server.main" in q for q in qnames)
    # Type + method.
    assert any(q.endswith(".Service") for q in qnames)
    assert any(q.endswith(".Service.Greet") for q in qnames)
    # Top-level func.
    assert any(q.endswith(".main") for q in qnames)
    # Imports.
    targets = [e.to_qname for e in result.edges if e.edge_type == EdgeType.IMPORTS]
    assert "fmt" in targets
    assert "net/http" in targets


def test_java_extracts_class_methods_imports() -> None:
    src = b"""
package com.example;

import java.util.List;
import java.util.ArrayList;

public class Greeter implements Runnable {
    public void run() {}
    public String greet(String name) { return "hi " + name; }
}
"""
    result = JavaExtractor().extract("src/main/java/com/example/Greeter.java", src, ".")
    qnames = {s.qualified_name for s in result.symbols}
    assert any(q.endswith(".Greeter") for q in qnames)
    assert any(q.endswith(".Greeter.run") for q in qnames)
    assert any(q.endswith(".Greeter.greet") for q in qnames)
    targets = [e.to_qname for e in result.edges if e.edge_type == EdgeType.IMPORTS]
    assert any("List" in t for t in targets)
    # IMPLEMENTS edge to Runnable.
    impls = [e for e in result.edges if e.edge_type == EdgeType.IMPLEMENTS]
    assert any("Runnable" in (e.to_qname,) or "Runnable" in e.target_candidates for e in impls)


def test_extractor_dispatch_picks_longest_extension_first() -> None:
    """`.tsx` matches the TS extractor (not the JS extractor's `.cjs`/`.jsx`)."""
    extractor = get_extractor_for_file("src/component.tsx")
    assert extractor is not None
    assert extractor.language == "typescript"


def test_unsupported_extension_returns_none() -> None:
    assert get_extractor_for_file("README.md") is None
    assert get_extractor_for_file("Cargo.toml") is None


def test_module_node_emitted_for_each_file() -> None:
    """Every extractor produces exactly one Module symbol per file."""
    from services.ingestion.code_graph.extractors.python import PythonExtractor

    fixtures = [
        ("src/x.py", PythonExtractor()),
        ("src/x.ts", TypeScriptExtractor()),
        ("src/x.js", JavaScriptExtractor()),
        ("cmd/x.go", GoExtractor()),
        ("src/X.java", JavaExtractor()),
    ]
    for path, ext in fixtures:
        # A file with at least one named decl, so the module qname surfaces.
        if path.endswith(".py"):
            content = b"def f(): pass\n"
        elif path.endswith((".ts", ".js")):
            content = b"export function f() {}\n"
        elif path.endswith(".go"):
            content = b"package x\nfunc f() {}\n"
        else:
            content = b"public class X {}\n"
        result = ext.extract(path, content, ".")
        modules = [s for s in result.symbols if s.kind == NodeLabel.CODE_SYMBOL]
        assert len(modules) == 1, f"{path}: expected 1 Module symbol, got {len(modules)}"
