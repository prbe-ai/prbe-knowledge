"""Manual document upload parsing helpers.

The upload endpoint accepts any file, but this module only admits files where
we can extract useful text without OCR/multimedia/PDF parsing. Original bytes
are staged in R2 first, then deleted after the queued payload normalizes.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException

from shared.constants import DocType

MAX_MANUAL_UPLOAD_FILES = 10
MAX_MANUAL_UPLOAD_BYTES = 10 * 1024 * 1024
# MAX_MANUAL_UPLOAD_BYTES bounds only the COMPRESSED upload; DEFLATE ratios
# above 1000x turn a 10 MB .docx into multi-GB XML (decompression bomb). This
# caps the total DECOMPRESSED bytes read across all XML parts of one archive.
MAX_DOCX_DECOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_EXTRACTED_CHARS = 2_000_000

_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".sql",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".scss",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".zsh",
    ".fish",
}
_UNSUPPORTED_BINARY_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".heic",
    ".mp3",
    ".mp4",
    ".mov",
    ".wav",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
}
_UNSUPPORTED_BINARY_MIME_PREFIXES = ("image/", "audio/", "video/")


class ManualUploadParseError(ValueError):
    """Raised when an uploaded file has no extractable text."""


@dataclass(frozen=True, slots=True)
class ParsedManualUpload:
    filename: str
    text: str
    parse_engine: str
    doc_type: DocType


def safe_filename(filename: str | None) -> str:
    name = PurePath(filename or "upload").name.strip()
    if not name or name in {".", ".."}:
        return "upload"
    return re.sub(r"[\x00-\x1f\x7f]", "", name)[:240] or "upload"


def parse_manual_upload(
    filename: str | None,
    content_type: str | None,
    body: bytes,
) -> ParsedManualUpload:
    name = safe_filename(filename)
    lower_name = name.lower()
    content_type = (content_type or "application/octet-stream").lower()

    if not body:
        raise ManualUploadParseError("file is empty")

    if _is_docx(lower_name, content_type):
        text = _extract_docx(body)
        return ParsedManualUpload(
            filename=name,
            text=_clean_extracted_text(text),
            parse_engine="docx-xml",
            doc_type=DocType.MANUAL_UPLOAD_DOCX,
        )

    if _is_known_unsupported_binary(lower_name, content_type):
        raise ManualUploadParseError("no extractable text found")

    text = _decode_probable_text(body)
    if not text:
        raise ManualUploadParseError("no extractable text found")

    return ParsedManualUpload(
        filename=name,
        text=_clean_extracted_text(text),
        parse_engine="plain-text",
        doc_type=_text_doc_type(lower_name, content_type),
    )


def _is_docx(filename: str, content_type: str) -> bool:
    return filename.endswith(".docx") or content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def _is_known_unsupported_binary(filename: str, content_type: str) -> bool:
    ext = PurePath(filename).suffix.lower()
    return (
        ext in _UNSUPPORTED_BINARY_EXTENSIONS
        or content_type == "application/pdf"
        or content_type.startswith(_UNSUPPORTED_BINARY_MIME_PREFIXES)
    )


def _read_bounded_entry(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: int
) -> bytes:
    if info.file_size > budget:
        raise ManualUploadParseError(
            f"docx decompresses beyond {MAX_DOCX_DECOMPRESSED_BYTES} byte limit"
        )
    # ZipInfo.file_size comes from the attacker-controlled header, so bound the
    # actual decompressed read too instead of trusting the declared size.
    with archive.open(info) as entry:
        data = entry.read(budget + 1)
    if len(data) > budget:
        raise ManualUploadParseError(
            f"docx decompresses beyond {MAX_DOCX_DECOMPRESSED_BYTES} byte limit"
        )
    return data


def _extract_docx(body: bytes) -> str:
    budget = MAX_DOCX_DECOMPRESSED_BYTES
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            xml_names = [
                "word/document.xml",
                *sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("word/header")
                    or name.startswith("word/footer")
                    or name.startswith("word/footnotes")
                    or name.startswith("word/endnotes")
                ),
            ]
            paragraphs: list[str] = []
            for xml_name in xml_names:
                try:
                    info = archive.getinfo(xml_name)
                except KeyError:
                    continue
                raw_xml = _read_bounded_entry(archive, info, budget)
                budget -= len(raw_xml)
                paragraphs.extend(_paragraphs_from_word_xml(raw_xml))
    except (
        zipfile.BadZipFile,
        SafeElementTree.ParseError,
        DefusedXmlException,
        KeyError,
    ) as exc:
        raise ManualUploadParseError("docx text extraction failed") from exc

    text = "\n\n".join(p for p in paragraphs if p.strip())
    if not text.strip():
        raise ManualUploadParseError("docx contains no extractable text")
    return text


def _paragraphs_from_word_xml(raw_xml: bytes) -> list[str]:
    root = SafeElementTree.fromstring(raw_xml, forbid_dtd=True)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_DOCX_NS}p"):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_DOCX_NS}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{_DOCX_NS}tab":
                pieces.append("\t")
            elif node.tag == f"{_DOCX_NS}br":
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _decode_probable_text(body: bytes) -> str | None:
    encodings: list[str] = []
    if body.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.append("utf-16")
    encodings.extend(["utf-8-sig", "utf-8", "cp1252"])

    seen: set[str] = set()
    for encoding in encodings:
        if encoding in seen:
            continue
        seen.add(encoding)
        try:
            text = body.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(text):
            return text
    return None


def _looks_like_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.count("\x00") / max(len(stripped), 1) > 0.01:
        return False
    printable = 0
    checked = 0
    for ch in stripped[:10000]:
        if ch in "\n\r\t" or ch.isprintable():
            printable += 1
        checked += 1
    return checked > 0 and printable / checked >= 0.85


def _clean_extracted_text(text: str) -> str:
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()
    if not cleaned:
        raise ManualUploadParseError("no extractable text found")
    return cleaned[:MAX_EXTRACTED_CHARS]


def _text_doc_type(filename: str, content_type: str) -> DocType:
    if filename.endswith((".md", ".markdown")) or content_type in {
        "text/markdown",
        "text/x-markdown",
    }:
        return DocType.MANUAL_UPLOAD_MARKDOWN
    if filename.endswith(".txt") or content_type.startswith("text/plain"):
        return DocType.MANUAL_UPLOAD_TEXT
    ext = PurePath(filename).suffix.lower()
    if ext in _TEXT_EXTENSIONS or content_type.startswith("text/"):
        return DocType.MANUAL_UPLOAD_FILE
    return DocType.MANUAL_UPLOAD_FILE
