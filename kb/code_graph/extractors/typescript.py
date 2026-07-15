"""TypeScript / TSX tree-sitter extractor — smoke level for PR-A.

Extracts top-level function / class / method declarations + module imports.
Deep call/type resolution is a fast-follow PR per spec §11.

NOTE: tree-sitter-typescript publishes TWO grammars in one package: a base
TypeScript grammar and a TSX grammar (both packaged together). We pick TSX
because it parses a strict superset of TypeScript without breaking — `.ts`
files parse fine through TSX.
"""

from __future__ import annotations

from typing import ClassVar

import tree_sitter_typescript as tsts
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

_TSX_LANGUAGE = Language(tsts.language_tsx())


class TypeScriptExtractor:
    language: ClassVar[str] = "typescript"
    extractor_version: ClassVar[str] = "ts-v1"
    file_extensions: ClassVar[tuple[str, ...]] = (".ts", ".tsx")

    def __init__(self) -> None:
        self._parser = Parser(_TSX_LANGUAGE)

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
            if t in ("function_declaration", "function_signature"):
                _emit_function(child, content, file_path, module_qname, result)
            elif t in ("class_declaration", "abstract_class_declaration"):
                _emit_class(child, content, file_path, module_qname, result)
            elif t == "import_statement":
                _emit_import(child, content, module_qname, result)
            elif t == "export_statement":
                # `export function foo() { ... }` etc. — descend into the
                # underlying decl.
                for sub in child.children:
                    if sub.type in ("function_declaration", "function_signature"):
                        _emit_function(sub, content, file_path, module_qname, result)
                    elif sub.type in ("class_declaration", "abstract_class_declaration"):
                        _emit_class(sub, content, file_path, module_qname, result)

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
            signature=f"function {name}",
        )
    )
    result.edges.append(
        CodeEdge(
            edge_type=EdgeType.DEFINED_IN,
            from_qname=qname,
            to_qname=module_qname,
        )
    )


def _emit_class(node, content, file_path, module_qname, result):
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node, content)
    qname = f"{module_qname}.{name}" if module_qname else name
    result.symbols.append(
        Symbol(
            qualified_name=qname,
            kind=CodeSymbolKind.CLASS,
            file_path=file_path,
            def_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_snippet=_node_text(node, content),
            signature=f"class {name}",
        )
    )
    result.edges.append(
        CodeEdge(
            edge_type=EdgeType.DEFINED_IN,
            from_qname=qname,
            to_qname=module_qname,
        )
    )

    # Methods inside the class body.
    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.children:
            if member.type in ("method_definition", "method_signature"):
                method_name_node = member.child_by_field_name("name")
                if method_name_node is None:
                    continue
                method_name = _node_text(method_name_node, content)
                method_qname = f"{qname}.{method_name}"
                result.symbols.append(
                    Symbol(
                        qualified_name=method_qname,
                        kind=CodeSymbolKind.METHOD,
                        file_path=file_path,
                        def_line=member.start_point[0] + 1,
                        end_line=member.end_point[0] + 1,
                        source_snippet=_node_text(member, content),
                        signature=f"{name}.{method_name}",
                        parent_qname=qname,
                    )
                )
                result.edges.append(
                    CodeEdge(
                        edge_type=EdgeType.DEFINED_IN,
                        from_qname=method_qname,
                        to_qname=qname,
                    )
                )

    # Heritage clauses — `extends X`, `implements Y, Z`.
    for child in node.children:
        if child.type == "class_heritage":
            for clause in child.children:
                if clause.type == "extends_clause":
                    for sub in clause.children:
                        if sub.type in ("identifier", "type_identifier"):
                            base = _node_text(sub, content)
                            result.edges.append(
                                CodeEdge(
                                    edge_type=EdgeType.INHERITS,
                                    from_qname=qname,
                                    to_qname=base,
                                    ambiguous=True,
                                    target_candidates=[base],
                                )
                            )
                elif clause.type == "implements_clause":
                    for sub in clause.children:
                        if sub.type in ("identifier", "type_identifier"):
                            base = _node_text(sub, content)
                            result.edges.append(
                                CodeEdge(
                                    edge_type=EdgeType.IMPLEMENTS,
                                    from_qname=qname,
                                    to_qname=base,
                                    ambiguous=True,
                                    target_candidates=[base],
                                )
                            )


def _emit_import(node, content, module_qname, result):
    for child in node.children:
        if child.type == "string":
            target = _node_text(child, content).strip("'\"")
            result.edges.append(
                CodeEdge(
                    edge_type=EdgeType.IMPORTS,
                    from_qname=module_qname,
                    to_qname=target,
                )
            )
            break


_INSTANCE: Extractor = TypeScriptExtractor()
register(_INSTANCE)


__all__ = ["TypeScriptExtractor"]
