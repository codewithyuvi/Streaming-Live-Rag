"""
retrieval/parsers.py — Universal Document Text Extractor for Streaming Live RAG.

Supports:
  - PDF (.pdf) via pypdf
  - Word (.docx) via python-docx (legacy .doc not supported)
  - PowerPoint (.pptx) via python-pptx (legacy .ppt not supported)
  - Excel (.xlsx) via openpyxl (legacy .xls not supported)
  - CSV / TSV (.csv, .tsv) via standard csv
  - JSON / JSONL (.json, .jsonl) via standard json
  - YAML (.yaml, .yml) via pyyaml
  - HTML (.html, .htm) via BeautifulSoup4
  - Rich Text (.rtf) via striprtf
  - Plain Text & Markdown (.txt, .md)

Each document is parsed into discrete, citeable sections with canonical tags:
[Doc_XX §1], [Doc_XX §2], etc.
"""

import os
import io
import re
import csv
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Normalizes whitespace and removes null bytes."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(data: bytes) -> List[str]:
    """Extracts text page by page from PDF bytes."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for idx, page in enumerate(reader.pages, 1):
        txt = _clean_text(page.extract_text() or "")
        if txt:
            pages.append(f"Page {idx}:\n{txt}")
    return pages


def extract_docx(data: bytes) -> List[str]:
    """Extracts text from Word DOCX paragraphs and tables."""
    import docx
    doc = docx.Document(io.BytesIO(data))
    sections = []
    current_block = []

    for p in doc.paragraphs:
        txt = _clean_text(p.text)
        if not txt:
            continue
        # If heading style or large block, push current block
        if p.style and "heading" in p.style.name.lower() and current_block:
            sections.append("\n".join(current_block))
            current_block = [txt]
        else:
            current_block.append(txt)
            if sum(len(x) for x in current_block) > 600:
                sections.append("\n".join(current_block))
                current_block = []

    if current_block:
        sections.append("\n".join(current_block))

    # Also extract tables
    for table in doc.tables:
        rows_txt = []
        for row in table.rows:
            row_vals = [_clean_text(cell.text) for cell in row.cells if _clean_text(cell.text)]
            if row_vals:
                rows_txt.append(" | ".join(row_vals))
        if rows_txt:
            sections.append("Table Data:\n" + "\n".join(rows_txt))

    return sections


def extract_pptx(data: bytes) -> List[str]:
    """Extracts text from PowerPoint slides."""
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    slides = []
    for idx, slide in enumerate(prs.slides, 1):
        shape_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = _clean_text(shape.text_frame.text)
                if txt:
                    shape_texts.append(txt)
            elif shape.has_table:
                for row in shape.table.rows:
                    row_vals = [_clean_text(cell.text) for cell in row.cells if _clean_text(cell.text)]
                    if row_vals:
                        shape_texts.append(" | ".join(row_vals))
        if shape_texts:
            slides.append(f"Slide {idx}:\n" + "\n".join(shape_texts))
    return slides


def extract_xlsx(data: bytes) -> List[str]:
    """Extracts text from Excel spreadsheets per worksheet."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    sheets = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_txt = []
        for row in ws.iter_rows(values_only=True):
            non_empty = [str(v).strip() for v in row if v is not None and str(v).strip()]
            if non_empty:
                rows_txt.append(" | ".join(non_empty))
        if rows_txt:
            sheets.append(f"Sheet: {sheet_name}\n" + "\n".join(rows_txt[:100]))  # cap at 100 rows per sheet
    return sheets


def extract_csv(data: bytes) -> List[str]:
    """Extracts text from CSV/TSV bytes."""
    text = data.decode("utf-8", errors="replace")
    dialect = csv.Sniffer().sniff(text[:2048]) if len(text) > 10 else "excel"
    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows = []
    for row in reader:
        non_empty = [c.strip() for c in row if c.strip()]
        if non_empty:
            rows.append(" | ".join(non_empty))
    # Group rows into manageable chunks
    chunks = []
    batch_size = 20
    for i in range(0, len(rows), batch_size):
        chunk_rows = rows[i : i + batch_size]
        chunks.append(f"Rows {i+1}-{i+len(chunk_rows)}:\n" + "\n".join(chunk_rows))
    return chunks


def extract_json(data: bytes) -> List[str]:
    """Extracts text from JSON / JSONL."""
    raw = data.decode("utf-8", errors="replace").strip()
    sections = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                sections.append(f"Item {i}:\n{json.dumps(item, indent=2)}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                sections.append(f"Section {k}:\n{json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)}")
        else:
            sections.append(str(obj))
    except json.JSONDecodeError:
        # Try JSONL
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for i, line in enumerate(lines, 1):
            try:
                sections.append(f"Record {i}:\n{json.dumps(json.loads(line), indent=2)}")
            except Exception:
                sections.append(f"Record {i}:\n{line}")
    return sections


def extract_yaml(data: bytes) -> List[str]:
    """Extracts text from YAML."""
    import yaml
    raw = data.decode("utf-8", errors="replace")
    docs = list(yaml.safe_load_all(raw))
    sections = []
    for idx, doc in enumerate(docs, 1):
        if doc is not None:
            sections.append(f"Document {idx}:\n" + yaml.dump(doc, default_flow_style=False))
    return sections


def extract_html(data: bytes) -> List[str]:
    """Extracts readable text from HTML files with boilerplate removal."""
    from bs4 import BeautifulSoup
    raw = data.decode("utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for el in soup(["script", "style", "meta", "noscript", "nav", "footer", "header"]):
        el.extract()
    text = soup.get_text(separator="\n")
    return extract_txt_md(text.encode("utf-8"))


def extract_rtf(data: bytes) -> List[str]:
    """Extracts text from RTF. Fails closed with clean exception rather than embedding raw RTF markup."""
    try:
        from striprtf.striprtf import rtf_to_text
        raw = data.decode("utf-8", errors="replace")
        plain = rtf_to_text(raw).strip()
        if not plain:
            raise ValueError("RTF contains no extractable text.")
        return extract_txt_md(plain.encode("utf-8"))
    except Exception as e:
        logger.warning(f"striprtf failed: {e}")
        raise ValueError(f"Failed to parse RTF document: {e}") from e


def extract_txt_md(data: bytes) -> List[Any]:
    """Extracts paragraphs from plain text or markdown."""
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Check for pre-tagged format: Doc_XX §Y
    if re.search(r"Doc_\d+\s*§\d+", text):
        blocks = [b.strip() for b in re.split(r"(?=Doc_\d+\s*§\d+)", text) if b.strip()]
        out = []
        for b in blocks:
            lines = b.split("\n", 1)
            if len(lines) >= 2 and (m := re.search(r"§(\d+)", lines[0])):
                sec_num = m.group(1)
                out.append((sec_num, lines[1].strip()))
            else:
                out.append(b.strip())
        return out

    # Standard split by double newline or headers
    blocks = [b.strip() for b in re.split(r"\n\s*\n|(?=^#{1,3}\s)", text, flags=re.MULTILINE) if b.strip()]
    # Combine very short blocks (< 100 chars)
    combined = []
    buf = []
    for b in blocks:
        buf.append(b)
        if sum(len(x) for x in buf) >= 200:
            combined.append("\n\n".join(buf))
            buf = []
    if buf:
        combined.append("\n\n".join(buf))
    return combined if combined else [text]


def _slice_long_unit(unit: str, max_chars: int, overlap: int) -> List[str]:
    """Slices a single unit that exceeds max_chars using newline, word, or character boundaries."""
    if len(unit) <= max_chars:
        return [unit]

    # Try splitting by newline
    if "\n" in unit:
        lines = [line.strip() for line in unit.split("\n") if line.strip()]
        if len(lines) > 1:
            res = []
            for line in lines:
                res.extend(_slice_long_unit(line, max_chars, overlap))
            return res

    # Try splitting by whitespace (words)
    words = unit.split()
    if len(words) > 1:
        res = []
        cur = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + (1 if cur else 0) > max_chars and cur:
                res.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += len(w) + (1 if len(cur) > 1 else 0)
        if cur:
            res.append(" ".join(cur))
        final_res = []
        for c in res:
            if len(c) > max_chars:
                final_res.extend(_slice_long_unit(c, max_chars, overlap))
            else:
                final_res.append(c)
        return final_res

    # Hard boundary-less fallback (single token or unsegmented string > max_chars): character slicing
    step = max(1, max_chars - overlap)
    return [unit[i : i + max_chars] for i in range(0, len(unit), step)]


def _subdivide_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    """
    Subdivides long text blocks on sentence boundaries to enforce a chunk-size ceiling.
    Falls back to newline, word, or character slicing when sentence boundaries are absent.
    Prevents silent truncation by 512-token models (e.g. bge-small-en-v1.5).
    """
    if len(text) <= max_chars:
        return [text]

    raw_sentences = re.split(r"(?<=[.?!])\s+", text)
    units = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if len(s_clean) > max_chars:
            units.extend(_slice_long_unit(s_clean, max_chars, overlap))
        else:
            units.append(s_clean)

    chunks = []
    current_chunk = []
    current_len = 0

    for u in units:
        if current_len + len(u) + (1 if current_chunk else 0) > max_chars and current_chunk:
            chunk_str = " ".join(current_chunk)
            chunks.append(chunk_str)
            # Retain tail for overlap
            overlap_buf = []
            overlap_len = 0
            for prev_u in reversed(current_chunk):
                if overlap_len + len(prev_u) <= overlap:
                    overlap_buf.insert(0, prev_u)
                    overlap_len += len(prev_u)
                else:
                    break
            current_chunk = overlap_buf + [u]
            current_len = sum(len(x) for x in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(u)
            current_len += len(u) + (1 if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def extract_sections(filename: str, content: bytes, doc_id: str) -> List[Dict[str, Any]]:
    """
    Universal dispatcher that converts any file's raw bytes into canonical
    chunk dictionaries ready for vector embedding and retrieval:
    [
        {
            "doc_id": "Doc_03",
            "section": "1.1",
            "tag": "Doc_03 §1.1",
            "text": "..."
        },
        ...
    ]
    """
    ext = os.path.splitext(filename.lower())[1]

    # Explicitly reject legacy binary formats with clear guidance
    if ext in (".doc", ".ppt", ".xls"):
        raise ValueError(
            f"Legacy binary Office format '{ext}' is not supported. Please export/convert to modern format (.docx, .pptx, or .xlsx)."
        )

    if ext == ".pdf":
        raw_sections = extract_pdf(content)
    elif ext == ".docx":
        raw_sections = extract_docx(content)
    elif ext == ".pptx":
        raw_sections = extract_pptx(content)
    elif ext == ".xlsx":
        raw_sections = extract_xlsx(content)
    elif ext in (".csv", ".tsv"):
        raw_sections = extract_csv(content)
    elif ext in (".json", ".jsonl"):
        raw_sections = extract_json(content)
    elif ext in (".yaml", ".yml"):
        raw_sections = extract_yaml(content)
    elif ext in (".html", ".htm"):
        raw_sections = extract_html(content)
    elif ext == ".rtf":
        raw_sections = extract_rtf(content)
    else:  # .txt, .md, and fallbacks
        raw_sections = extract_txt_md(content)

    sections = []
    for idx, item in enumerate(raw_sections, 1):
        if isinstance(item, tuple):
            explicit_sec_num, raw_txt = item
        else:
            explicit_sec_num, raw_txt = str(idx), item

        cleaned = _clean_text(raw_txt)
        if not cleaned:
            continue
        sub_chunks = _subdivide_text(cleaned, max_chars=1200, overlap=100)
        if len(sub_chunks) == 1:
            sec_num = explicit_sec_num
            tag = f"{doc_id} §{sec_num}"
            sections.append({
                "doc_id": doc_id,
                "section": sec_num,
                "tag": tag,
                "text": sub_chunks[0],
                "filename": filename,
            })
        else:
            for sub_idx, sub_txt in enumerate(sub_chunks, 1):
                sec_num = f"{explicit_sec_num}.{sub_idx}"
                tag = f"{doc_id} §{sec_num}"
                sections.append({
                    "doc_id": doc_id,
                    "section": sec_num,
                    "tag": tag,
                    "text": sub_txt,
                    "filename": filename,
                })

    return sections
