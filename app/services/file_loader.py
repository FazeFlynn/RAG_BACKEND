"""
File loader service - handles PDF, TXT, CSV, XLS/XLSX, DOCX, Markdown, HTML.
Returns list of dicts with 'content' and 'metadata'.
"""

import os
from pathlib import Path
from typing import Any
from loguru import logger
import pandas as pd


def load_file(file_path: str) -> list[dict[str, Any]]:
    """Load a file and return a list of document dicts."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    loaders = {
        ".pdf": _load_pdf,
        ".txt": _load_text,
        ".md": _load_text,
        ".csv": _load_csv,
        ".xls": _load_excel,
        ".xlsx": _load_excel,
        ".docx": _load_docx,
        ".html": _load_html,
        ".htm": _load_html,
        ".json": _load_text,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    logger.info(f"Loading file: {path.name} (type: {suffix})")
    documents = loader(file_path)
    logger.info(f"Loaded {len(documents)} document(s) from {path.name}")
    return documents


def _load_pdf(file_path: str) -> list[dict[str, Any]]:
    import fitz  # PyMuPDF

    documents = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": page_num + 1,
                        "type": "pdf",
                    },
                })
    return documents


def _load_text(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()

    if not content:
        return []

    return [{
        "content": content,
        "metadata": {
            "source": os.path.basename(file_path),
            "type": "text",
        },
    }]


def _load_csv(file_path: str) -> list[dict[str, Any]]:
    df = pd.read_csv(file_path)
    return _dataframe_to_documents(df, file_path, "csv")


def _load_excel(file_path: str) -> list[dict[str, Any]]:
    documents = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        docs = _dataframe_to_documents(df, file_path, "excel", sheet_name)
        documents.extend(docs)
    return documents


def _dataframe_to_documents(
    df: pd.DataFrame,
    file_path: str,
    file_type: str,
    sheet_name: str | None = None,
) -> list[dict[str, Any]]:
    documents = []
    # Convert dataframe to readable text in row groups
    columns = df.columns.tolist()
    header = " | ".join(str(c) for c in columns)

    # Group rows into chunks of 20 for manageable documents
    group_size = 20
    for start in range(0, len(df), group_size):
        end = min(start + group_size, len(df))
        rows_text = []
        for _, row in df.iloc[start:end].iterrows():
            row_str = " | ".join(str(v) for v in row.values)
            rows_text.append(row_str)

        content = f"Columns: {header}\n\n" + "\n".join(rows_text)
        metadata = {
            "source": os.path.basename(file_path),
            "type": file_type,
            "rows": f"{start + 1}-{end}",
        }
        if sheet_name:
            metadata["sheet"] = sheet_name

        documents.append({"content": content, "metadata": metadata})

    return documents


def _load_docx(file_path: str) -> list[dict[str, Any]]:
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        return []

    content = "\n\n".join(paragraphs)
    return [{
        "content": content,
        "metadata": {
            "source": os.path.basename(file_path),
            "type": "docx",
        },
    }]


def _load_html(file_path: str) -> list[dict[str, Any]]:
    from bs4 import BeautifulSoup

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        return []

    return [{
        "content": text,
        "metadata": {
            "source": os.path.basename(file_path),
            "type": "html",
        },
    }]


def get_supported_extensions() -> list[str]:
    return [".pdf", ".txt", ".md", ".csv", ".xls", ".xlsx", ".docx", ".html", ".htm", ".json"]
