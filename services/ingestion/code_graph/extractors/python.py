"""Python tree-sitter extractor — deep coverage.

Emits symbols (Module/Class/Function/Method) and edges (DEFINED_IN, IMPORTS,
CALLS, INHERITS). LOW-ambition resolver per spec §4.5: resolves imports and
locals; everything else is AMBIGUOUS with candidate list.

The `def_line` invariant (spec §4.3 / D5): for any Function/Class/Method,
`def_line` is ALWAYS the line of the `def`/`class` keyword, never a
decorator line. This file's `_def_line()` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser

from services.ingestion.code_graph.extractors._common import module_qname_from_path
from services.ingestion.code_graph.extractors.registry import register
from services.ingestion.code_graph.types import (
    CodeEdge,
    Extractor,
    ExtractResult,
    Symbol,
)
from shared.constants import EdgeType, NodeLabel

_PY_LANGUAGE = Language(tsp.language())


@dataclass(slots=True)
class _ScopeFrame:
    """Tracks the current lexical scope as we walk a Python AST.

    `qname` is the dotted path from the module root.  `kind` tells call
    resolution whether the enclosing scope is a class (so unqualified
    `self.foo` references can resolve to a sibling method) or a function
    (so locals shadow imports).
    """

    qname: str
    kind: str   # 'module' | 'class' | 'function'
    locals: dict[str, str] = field(default_factory=dict)
    # parameters ⊂ locals; tracked separately so resolver can prefer
    # parameter targets over generic locals if we ever care.
    parameters: dict[str, str] = field(default_factory=dict)


class PythonExtractor:
    """Extractor implementation for Python source files.

    `extractor_version` is bumped per the contract in spec §4.6 whenever
    output changes incompatibly. Bumping invalidates the language's
    `code_repo_state` cache entries on next push.
    """

    language: ClassVar[str] = "python"
    extractor_version: ClassVar[str] = "py-v1"
    file_extensions: ClassVar[tuple[str, ...]] = (".py",)

    def __init__(self) -> None:
        self._parser = Parser(_PY_LANGUAGE)

    def extract(
        self,
        file_path: str,
        content: bytes,
        repo_root: str,  # currently unused; available for future cross-file resolution
    ) -> ExtractResult:
        del repo_root  # reserved
        result = ExtractResult()
        try:
            tree = self._parser.parse(content)
        except Exception as exc:
            result.errors.append(f"parse failed: {exc}")
            return result

        module_qname = module_qname_from_path(file_path, self.language)
        if module_qname:
            module_symbol = Symbol(
                qualified_name=module_qname,
                kind=NodeLabel.MODULE,
                file_path=file_path,
                def_line=1,
                end_line=max(1, content.count(b"\n") + 1),
                source_snippet=_module_summary(content),
                signature=f"module {module_qname}",
            )
            result.symbols.append(module_symbol)

        imports = _collect_imports(tree.root_node, content)
        for target_module in imports:
            result.edges.append(
                CodeEdge(
                    edge_type=EdgeType.IMPORTS,
                    from_qname=module_qname,
                    to_qname=target_module,
                )
            )

        # The walker carries the import table so call-resolution can map
        # `bar()` → `foo.bar` when we have `from foo import bar`.
        walker = _Walker(
            file_path=file_path,
            content=content,
            module_qname=module_qname,
            imports=imports,
            result=result,
        )
        walker.walk(tree.root_node, _ScopeFrame(qname=module_qname, kind="module"))
        return result


# ---- module-level helpers --------------------------------------------------


def _module_summary(content: bytes) -> str:
    """Top-level summary of a module: docstring + imports list.

    Used as `Document.metadata['body']` for the Module Document so the
    chunker has something compact to embed without duplicating every
    function's body (those have their own Documents).
    """
    text = content.decode("utf-8", errors="replace")
    head = "\n".join(text.splitlines()[:80])  # first 80 lines is plenty for a summary
    return head


def _collect_imports(root: Node, content: bytes) -> list[str]:
    """Pull module-level import targets from the AST.

    Returns the import-target module path for each `import x` /
    `from y import z` statement at the module level. Submodule paths are
    flattened: `from foo.bar import baz` → `foo.bar`.
    """
    imports: list[str] = []
    for child in root.children:
        if child.type == "import_statement":
            # `import foo` or `import foo.bar as baz`
            for sub in child.children:
                if sub.type == "dotted_name":
                    imports.append(_node_text(sub, content))
                elif sub.type == "aliased_import":
                    inner = sub.child_by_field_name("name")
                    if inner is not None:
                        imports.append(_node_text(inner, content))
        elif child.type == "import_from_statement":
            module_node = child.child_by_field_name("module_name")
            if module_node is not None:
                imports.append(_node_text(module_node, content))
    return imports


def _node_text(node: Node, content: bytes) -> str:
    return content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _def_line(node: Node) -> int:
    """Return 1-indexed line of the `def`/`class` keyword for a definition node.

    Tree-sitter's `function_definition` / `class_definition` nodes start at
    the `def`/`class` keyword (decorators are siblings via `decorated_definition`),
    so `start_point.row` already points to the right line. We only need to
    +1 for 1-indexed output.

    If we ever encounter a `decorated_definition`, we recurse into its
    inner `function_definition` / `class_definition` child — the inner
    node is what we want.
    """
    while node.type == "decorated_definition":
        # The actual def/class is a child; pick it.
        inner = node.child_by_field_name("definition")
        if inner is None:
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    inner = child
                    break
        if inner is None:
            break
        node = inner
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    return node.end_point[0] + 1


def _docstring(body_node: Node, content: bytes) -> str | None:
    """Extract the first-statement string literal as a docstring, if present."""
    if body_node is None or not body_node.children:
        return None
    first = body_node.children[0]
    if first.type != "expression_statement" or not first.children:
        return None
    expr = first.children[0]
    if expr.type != "string":
        return None
    raw = _node_text(expr, content)
    return raw.strip().strip("'\"")


def _signature_text(name: str, params_node: Node | None, content: bytes) -> str:
    if params_node is None:
        return f"def {name}()"
    return f"def {name}{_node_text(params_node, content)}"


# ---- AST walker ------------------------------------------------------------


class _Walker:
    """Recursive walker that emits Symbols and CodeEdges as it descends."""

    def __init__(
        self,
        file_path: str,
        content: bytes,
        module_qname: str,
        imports: list[str],
        result: ExtractResult,
    ) -> None:
        self.file_path = file_path
        self.content = content
        self.module_qname = module_qname
        # `_imports_by_alias`: locally-bound name → resolved module path.
        # Built from import statements so call-resolution can promote
        # `bar()` to `foo.bar` when we have `from foo import bar`.
        self._imports_by_alias = _alias_table(content)
        self.imports = imports
        self.result = result

    def walk(self, node: Node, frame: _ScopeFrame) -> None:
        for child in node.children:
            t = child.type
            if t == "decorated_definition":
                # Decorators don't change identity; descend to the inner def/class.
                inner_def = child.child_by_field_name("definition")
                if inner_def is None:
                    for c in child.children:
                        if c.type in ("function_definition", "class_definition"):
                            inner_def = c
                            break
                if inner_def is not None:
                    self._handle_definition(inner_def, frame)
            elif t in ("function_definition", "class_definition"):
                self._handle_definition(child, frame)
            elif t == "call":
                self._handle_call(child, frame)
            elif t == "expression_statement":
                # Recurse into expressions to catch nested calls.
                self.walk(child, frame)
            elif t == "block":
                self.walk(child, frame)
            elif t == "assignment":
                self._handle_assignment(child, frame)
            else:
                # Default: descend so we don't miss nested calls deep in
                # control-flow nodes (if/while/try). We don't open new
                # frames for these — they're not lexical scopes in Python.
                if child.children:
                    self.walk(child, frame)

    # ---- definitions ------------------------------------------------------

    def _handle_definition(self, node: Node, parent_frame: _ScopeFrame) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, self.content)
        body_node = node.child_by_field_name("body")
        params_node = node.child_by_field_name("parameters")

        own_qname = (
            f"{parent_frame.qname}.{name}" if parent_frame.qname else name
        )

        if node.type == "class_definition":
            kind = NodeLabel.CLASS
            signature = f"class {name}"
            new_frame = _ScopeFrame(qname=own_qname, kind="class")
        else:
            # function_definition — Method if parent is class, else Function.
            kind = NodeLabel.METHOD if parent_frame.kind == "class" else NodeLabel.FUNCTION
            signature = _signature_text(name, params_node, self.content)
            new_frame = _ScopeFrame(qname=own_qname, kind="function")
            # Track parameters as locals so call-resolution can fall back.
            if params_node is not None:
                for param_name in _parameter_names(params_node, self.content):
                    new_frame.parameters[param_name] = "<param>"
                    new_frame.locals[param_name] = "<param>"

        symbol = Symbol(
            qualified_name=own_qname,
            kind=kind,
            file_path=self.file_path,
            def_line=_def_line(node),
            end_line=_end_line(node),
            source_snippet=_node_text(node, self.content),
            signature=signature,
            docstring=_docstring(body_node, self.content) if body_node is not None else None,
            parent_qname=parent_frame.qname if parent_frame.kind == "class" else None,
        )
        self.result.symbols.append(symbol)

        # DEFINED_IN edge: symbol → parent scope.
        if parent_frame.kind == "module":
            self.result.edges.append(
                CodeEdge(
                    edge_type=EdgeType.DEFINED_IN,
                    from_qname=own_qname,
                    to_qname=self.module_qname,
                )
            )
        else:
            self.result.edges.append(
                CodeEdge(
                    edge_type=EdgeType.DEFINED_IN,
                    from_qname=own_qname,
                    to_qname=parent_frame.qname,
                )
            )

        # INHERITS edges for class definitions.
        if node.type == "class_definition":
            superclasses = node.child_by_field_name("superclasses")
            if superclasses is not None:
                for base in _argument_targets(superclasses, self.content):
                    resolved = self._resolve_name(base, parent_frame)
                    self.result.edges.append(
                        CodeEdge(
                            edge_type=EdgeType.INHERITS,
                            from_qname=own_qname,
                            to_qname=resolved or base,
                            ambiguous=resolved is None,
                            target_candidates=[base] if resolved is None else [],
                        )
                    )

        if body_node is not None:
            self.walk(body_node, new_frame)

    # ---- calls ------------------------------------------------------------

    def _handle_call(self, node: Node, frame: _ScopeFrame) -> None:
        # Only emit CALLS edges when the enclosing scope is a function/method.
        # Module-level top-of-file calls (e.g., `app = FastAPI()`) become
        # an edge from the module symbol, but we treat those as REFERENCES
        # to keep CALLS scoped to function bodies.
        function_node = node.child_by_field_name("function")
        if function_node is None:
            return
        target_text = _node_text(function_node, self.content)

        edge_type = EdgeType.CALLS if frame.kind == "function" else EdgeType.REFERENCES

        resolved = self._resolve_name(target_text, frame)
        candidates: list[str] = []
        ambiguous = resolved is None
        if ambiguous:
            candidates = [target_text]

        self.result.edges.append(
            CodeEdge(
                edge_type=edge_type,
                from_qname=frame.qname,
                to_qname=resolved or target_text,
                ambiguous=ambiguous,
                target_candidates=candidates,
            )
        )

        # Recurse into argument list — calls can nest.
        args = node.child_by_field_name("arguments")
        if args is not None:
            self.walk(args, frame)

    # ---- assignments ------------------------------------------------------

    def _handle_assignment(self, node: Node, frame: _ScopeFrame) -> None:
        # Track LHS names so call-resolution knows about locals. We don't
        # track RHS types — that's beyond LOW-ambition resolution.
        left = node.child_by_field_name("left")
        if left is None:
            return
        if left.type == "identifier":
            name = _node_text(left, self.content)
            frame.locals[name] = "<local>"
        # Ignore tuple-targets / pattern-targets for now; they bias toward
        # AMBIGUOUS resolution which is the safe failure mode.
        # Recurse RHS in case there are nested calls.
        right = node.child_by_field_name("right")
        if right is not None:
            self.walk(right, frame)

    # ---- resolver ---------------------------------------------------------

    def _resolve_name(self, raw_target: str, frame: _ScopeFrame) -> str | None:
        """LOW-ambition resolver. Returns a resolved qualified name or None.

        Order of precedence:
            1. If the target is a bare name and matches a frame local → unresolved
               (we know it's a parameter or local but don't know its type).
            2. If the target is a bare name and matches an import alias →
               resolve to `<imported_module>.<name>`.
            3. If the target is `self.foo` and we're inside a class scope →
               resolve to the class's `foo` attribute (unverified — could
               be inherited).
            4. Otherwise → None (caller emits AMBIGUOUS).
        """
        head = raw_target.split(".", 1)[0]
        if head in frame.locals:
            return None
        if head in self._imports_by_alias:
            module_path = self._imports_by_alias[head]
            tail = raw_target[len(head) :]
            return f"{module_path}{tail}" if tail else module_path
        if raw_target.startswith("self."):
            # Walk up to find the enclosing class — `_ScopeFrame` doesn't
            # currently track parent frames, so we approximate from the
            # function's `parent_qname` baked into its qname.
            #   frame.qname = <module>.<Class>.<method>
            # → the class qname is everything except the last segment.
            parts = frame.qname.rsplit(".", 1)
            if len(parts) == 2:
                class_qname = parts[0]
                attr = raw_target.split(".", 1)[1]
                return f"{class_qname}.{attr}"
        return None


def _alias_table(content: bytes) -> dict[str, str]:
    """Build a quick alias → module-path table from import statements.

    Re-parses the same content cheaply to keep the walker's API clean.
    Robust against malformed imports — non-import constructs are skipped.
    """
    table: dict[str, str] = {}
    parser = Parser(_PY_LANGUAGE)
    tree = parser.parse(content)
    for child in tree.root_node.children:
        if child.type == "import_statement":
            # `import foo` or `import foo.bar as baz`
            for sub in child.children:
                if sub.type == "dotted_name":
                    name = _node_text(sub, content)
                    alias = name.split(".")[0]
                    table[alias] = name
                elif sub.type == "aliased_import":
                    inner = sub.child_by_field_name("name")
                    alias_node = sub.child_by_field_name("alias")
                    if inner is not None and alias_node is not None:
                        table[_node_text(alias_node, content)] = _node_text(
                            inner, content
                        )
        elif child.type == "import_from_statement":
            module_node = child.child_by_field_name("module_name")
            if module_node is None:
                continue
            module_path = _node_text(module_node, content)
            # Each `name` field on the from-import is a brought-in symbol.
            for sub in child.children:
                if sub.type == "dotted_name" and sub != module_node:
                    sym = _node_text(sub, content)
                    table[sym] = f"{module_path}.{sym}"
                elif sub.type == "aliased_import":
                    inner = sub.child_by_field_name("name")
                    alias = sub.child_by_field_name("alias")
                    if inner is not None and alias is not None:
                        sym = _node_text(inner, content)
                        table[_node_text(alias, content)] = f"{module_path}.{sym}"
    return table


def _argument_targets(args_node: Node, content: bytes) -> list[str]:
    """Return the textual targets in an argument-list / superclass-list node."""
    out: list[str] = []
    for child in args_node.children:
        if child.type in ("identifier", "attribute"):
            out.append(_node_text(child, content))
    return out


def _parameter_names(params_node: Node, content: bytes) -> list[str]:
    out: list[str] = []
    for child in params_node.children:
        if child.type == "identifier":
            out.append(_node_text(child, content))
        elif child.type == "typed_parameter":
            for sub in child.children:
                if sub.type == "identifier":
                    out.append(_node_text(sub, content))
                    break
        elif child.type == "default_parameter":
            name = child.child_by_field_name("name")
            if name is not None:
                out.append(_node_text(name, content))
    return out


# Verify the Protocol contract at import time, then register.
_INSTANCE: Extractor = PythonExtractor()
register(_INSTANCE)


__all__ = ["PythonExtractor"]
