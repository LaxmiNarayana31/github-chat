"""Pytest root configuration providing offline mock for tiktoken and Windows FAISS DLL stability."""

import sys
from unittest.mock import MagicMock

import tiktoken

# Ensure FAISS C++ DLL failures on Windows do not crash pytest during test collection
try:
    import faiss
except Exception:
    mock_faiss = MagicMock()
    sys.modules["faiss"] = mock_faiss
    sys.modules["faiss.swigfaiss"] = mock_faiss
    sys.modules["faiss.loader"] = mock_faiss


class _MockEncoding:
    def __init__(self, name: str = "cl100k_base"):
        self.name = name

    def encode(self, text: str, *args, **kwargs):
        return [ord(c) for c in text[:1000]]

    def decode(self, tokens, *args, **kwargs):
        try:
            return "".join(chr(t) for t in tokens if 0 <= t <= 1114111)
        except Exception:
            return ""


_orig_get_encoding = tiktoken.get_encoding


def _safe_get_encoding(name: str):
    try:
        return _orig_get_encoding(name)
    except Exception:
        return _MockEncoding(name)


tiktoken.get_encoding = _safe_get_encoding
