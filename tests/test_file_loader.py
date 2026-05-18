"""Tests for the file loader service."""

import os
import tempfile
import pytest
import pandas as pd

from app.services.file_loader import load_file, get_supported_extensions


def test_supported_extensions():
    exts = get_supported_extensions()
    assert ".pdf" in exts
    assert ".csv" in exts
    assert ".txt" in exts
    assert ".xlsx" in exts


def test_load_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello, this is a test document.\nIt has multiple lines.\nThird line here.")
        f.flush()
        path = f.name

    try:
        docs = load_file(path)
        assert len(docs) == 1
        assert "Hello" in docs[0]["content"]
        assert docs[0]["metadata"]["type"] == "text"
    finally:
        os.unlink(path)


def test_load_csv_file():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    df.to_csv(path, index=False)

    try:
        docs = load_file(path)
        assert len(docs) >= 1
        assert "Alice" in docs[0]["content"]
        assert docs[0]["metadata"]["type"] == "csv"
    finally:
        os.unlink(path)


def test_load_unsupported_file():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(path)
    finally:
        os.unlink(path)


def test_load_empty_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("")
        path = f.name

    try:
        docs = load_file(path)
        assert len(docs) == 0
    finally:
        os.unlink(path)
