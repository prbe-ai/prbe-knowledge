"""Go tree-sitter extractor — smoke level for PR-A.

Extracts top-level function declarations, method declarations (receiver
binds them to a type), type declarations (struct/interface), and import
specs. No call resolution at this level.
"""

from __future__ import annotations

from typing import ClassVar

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from services.ingestion.code_graph.extractors._common import module_qname_from_path
from services.ingestion.code_graph.extractors.registry import register
from services.ingestion.code_graph.types import (
    CodeEdge,
    Extractor,
    ExtractResult,
    Symbol,
)
from shared.constants import CodeSymbolKind, EdgeType

_GO_LANGUAGE = Language(tsgo.language())


class GoExtractor:
    language: ClassVar[str] = "go"
    extractor_version: ClassVar[str] = "go-v1"
    file_extensions: ClassVar[tuple[str, ...]] = (".go",)

    def __init__(self) -> None:
        self._parser = Parser(_GO_LANGUAGE)

    def extract(
        self,
        file_path: str,
        content: bytes,
        repo_root: str,
    ) -> ExtractResult:
        del repo_root
        result = ExtractResult()
        try:
            tree = self._parser.parse(content)
        except Exception as exc:
            result.errors.append(f"parse failed: {exc}")
            return result

        module_qname = module_qname_from_path(file_path, self.language)
        if module_qname:
            result.symbols.append(
                Symbol(
                    qualified_name=module_qname,
                    kind=CodeSymbolKind.MODULE,
                    file_path=file_path,
                    def_line=1,
                    end_line=max(1, content.count(b"\n") + 1),
                    source_snippet=_summary(content),
                    signature=f"module {module_qname}",
                )
            )

        for child in tree.root_node.children:
            t = child.type
            if t == "function_declaration":
                _emit_function(child, content, file_path, module_qname, result)
            elif t == "method_declaration":
                _emit_method(child, content, file_path, module_qname, result)
            elif t == "type_declaration":
                _emit_types(child, content, file_path, module_qname, result)
            elif t == "import_declaration":
                _emit_imports(child, content, module_qname, result)

        return result


def _summary(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[:80])


def _node_text(node, content: bytes) -> str:
    return content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _emit_function(node, content, file_path, module_qname, result):
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node, content)
    qname = f"{module_qname}.{name}" if module_qname else name
    result.symbols.append(
        Symbol(
            qualified_name=qname,
            kind=CodeSymbolKind.FUNCTION,
            file_path=file_path,
            def_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_snippet=_node_text(node, content),
            signature=f"func {name}",
        )
    )
    result.edges.append(
        CodeEdge(
            edge_type=EdgeType.DEFINED_IN,
            from_qname=qname,
            to_qname=module_qname,
        )
    )


def _emit_method(node, content, file_path, module_qname, result):
    name_node = node.child_by_field_name("name")
    receiver_node = node.child_by_field_name("receiver")
    if name_node is None:
        return
    name = _node_text(name_node, content)

    # Receiver: `(s *Service)` -> we want "Service" as the parent type.
    receiver_type = ""
    if receiver_node is not None:
        for sub in _walk_descendants(receiver_node):
            if sub.type == "type_identifier":
                receiver_type = _node_text(sub, content)
                break

    if receiver_type and module_qname:
        parent_qname = f"{module_qname}.{receiver_type}"
        qname = f"{parent_qname}.{name}"
    elif module_qname:
        qname = f"{module_qname}.{name}"
        parent_qname = module_qname
    else:
        qname = name
        parent_qname = ""

    result.symbols.append(
        Symbol(
            qualified_name=qname,
            kind=CodeSymbolKind.METHOD,
            file_path=file_path,
            def_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_snippet=_node_text(node, content),
            signature=f"func ({receiver_type}) {name}" if receiver_type else f"func {name}",
            parent_qname=parent_qname or None,
        )
    )
    if parent_qname:
        result.edges.append(
            CodeEdge(
                edge_type=EdgeType.DEFINED_IN,
                from_qname=qname,
                to_qname=parent_qname,
            )
        )


def _emit_types(node, content, file_path, module_qname, result):
    for spec in node.children:
        if spec.type != "type_spec":
            continue
        name_node = spec.child_by_field_name("name")
        type_node = spec.child_by_field_name("type")
        if name_node is None:
            continue
        name = _node_text(name_node, content)
        qname = f"{module_qname}.{name}" if module_qname else name
        kind = CodeSymbolKind.CLASS  # struct/interface both surface as Class for retrieval
        result.symbols.append(
            Symbol(
                qualified_name=qname,
                kind=kind,
                file_path=file_path,
                def_line=spec.start_point[0] + 1,
                end_line=spec.end_point[0] + 1,
                source_snippet=_node_text(spec, content),
                signature=(
                    f"type {name} {type_node.type}" if type_node is not None else f"type {name}"
                ),
            )
        )
        result.edges.append(
            CodeEdge(
                edge_type=EdgeType.DEFINED_IN,
                from_qname=qname,
                to_qname=module_qname,
            )
        )


def _emit_imports(node, content, module_qname, result):
    for spec in _walk_descendants(node):
        if spec.type == "import_spec":
            path_node = spec.child_by_field_name("path")
            if path_node is None:
                # tree-sitter-go often returns the path as a string child
                # without a named field — fall through to generic search.
                for sub in spec.children:
                    if sub.type in ("interpreted_string_literal", "raw_string_literal"):
                        path_node = sub
                        break
            if path_node is None:
                continue
            target = _node_text(path_node, content).strip("`\"")
            result.edges.append(
                CodeEdge(
                    edge_type=EdgeType.IMPORTS,
                    from_qname=module_qname,
                    to_qname=target,
                )
            )


def _walk_descendants(node):
    """Yield every descendant of a tree-sitter node, depth-first."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


_INSTANCE: Extractor = GoExtractor()
register(_INSTANCE)


__all__ = ["GoExtractor"]
