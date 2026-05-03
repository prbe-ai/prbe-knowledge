"""Java tree-sitter extractor — smoke level for PR-A.

Extracts class/interface declarations + their method members + import
declarations. No deep type resolution; everything past the file level is
AMBIGUOUS or unresolved.
"""

from __future__ import annotations

from typing import ClassVar

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from services.ingestion.code_graph.extractors._common import module_qname_from_path
from services.ingestion.code_graph.extractors.registry import register
from services.ingestion.code_graph.types import (
    CodeEdge,
    Extractor,
    ExtractResult,
    Symbol,
)
from shared.constants import EdgeType, NodeLabel

_JAVA_LANGUAGE = Language(tsjava.language())


class JavaExtractor:
    language: ClassVar[str] = "java"
    extractor_version: ClassVar[str] = "java-v1"
    file_extensions: ClassVar[tuple[str, ...]] = (".java",)

    def __init__(self) -> None:
        self._parser = Parser(_JAVA_LANGUAGE)

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
                    kind=NodeLabel.MODULE,
                    file_path=file_path,
                    def_line=1,
                    end_line=max(1, content.count(b"\n") + 1),
                    source_snippet=_summary(content),
                    signature=f"module {module_qname}",
                )
            )

        for child in tree.root_node.children:
            t = child.type
            if t in ("class_declaration", "interface_declaration", "enum_declaration"):
                _emit_class_like(child, content, file_path, module_qname, result)
            elif t == "import_declaration":
                _emit_import(child, content, module_qname, result)

        return result


def _summary(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[:80])


def _node_text(node, content: bytes) -> str:
    return content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _emit_class_like(node, content, file_path, module_qname, result):
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node, content)
    qname = f"{module_qname}.{name}" if module_qname else name

    keyword = "class"
    if node.type == "interface_declaration":
        keyword = "interface"
    elif node.type == "enum_declaration":
        keyword = "enum"

    result.symbols.append(
        Symbol(
            qualified_name=qname,
            kind=NodeLabel.CLASS,
            file_path=file_path,
            def_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_snippet=_node_text(node, content),
            signature=f"{keyword} {name}",
        )
    )
    result.edges.append(
        CodeEdge(
            edge_type=EdgeType.DEFINED_IN,
            from_qname=qname,
            to_qname=module_qname,
        )
    )

    # Heritage: superclass + interfaces.
    superclass = node.child_by_field_name("superclass")
    if superclass is not None:
        for sub in superclass.children:
            if sub.type == "type_identifier":
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
    interfaces = node.child_by_field_name("interfaces")
    if interfaces is not None:
        for sub in _walk_descendants(interfaces):
            if sub.type == "type_identifier":
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

    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.children:
            if member.type in ("method_declaration", "constructor_declaration"):
                method_name_node = member.child_by_field_name("name")
                if method_name_node is None:
                    continue
                method_name = _node_text(method_name_node, content)
                method_qname = f"{qname}.{method_name}"
                result.symbols.append(
                    Symbol(
                        qualified_name=method_qname,
                        kind=NodeLabel.METHOD,
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


def _emit_import(node, content, module_qname, result):
    # `import foo.bar.Baz;` — the path is a `scoped_identifier` child.
    for child in node.children:
        if child.type in ("scoped_identifier", "identifier"):
            target = _node_text(child, content)
            result.edges.append(
                CodeEdge(
                    edge_type=EdgeType.IMPORTS,
                    from_qname=module_qname,
                    to_qname=target,
                )
            )
            return


def _walk_descendants(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


_INSTANCE: Extractor = JavaExtractor()
register(_INSTANCE)


__all__ = ["JavaExtractor"]
