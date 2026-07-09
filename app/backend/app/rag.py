from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    text: str
    score: float


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end]
        if end < len(text):
            candidates = [piece.rfind(mark) for mark in ["。", "！", "？", "；", "\n", "."]]
            cut = max(candidates)
            if cut > chunk_size * 0.5:
                end = start + cut + 1
                piece = text[start:end]
        chunks.append(normalize_text(piece))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [x for x in chunks if x]


class HybridRAGEngine:
    """
    从用户原 enhanced_rag.py 的设计迁移而来：
    word TF-IDF + char TF-IDF + 轻量去重。
    与 Streamlit 解耦，作为 FastAPI 后端长驻服务使用。
    """

    def __init__(self, knowledge_dir: Path, index_path: Path, chunk_size: int = 500, overlap: int = 90):
        self.knowledge_dir = knowledge_dir
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lock = RLock()
        self.chunks: list[dict] = []
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.word_matrix = None
        self.char_matrix = None
        self.load_or_rebuild()

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            raw = path.read_bytes()
            for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
                try:
                    return raw.decode(encoding)
                except UnicodeDecodeError:
                    pass
            return raw.decode("utf-8", errors="ignore")

        if suffix == ".csv":
            rows = []
            raw = path.read_bytes()
            decoded = None
            for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
                try:
                    decoded = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    pass
            if decoded is None:
                decoded = raw.decode("utf-8", errors="ignore")
            for row in csv.reader(decoded.splitlines()):
                text = "；".join(f"字段{i+1}：{cell.strip()}" for i, cell in enumerate(row) if cell.strip())
                if text:
                    rows.append(text)
            return "\n".join(rows)

        return ""

    def rebuild(self) -> None:
        with self.lock:
            rows = []
            for path in sorted(self.knowledge_dir.glob("*")):
                if path.suffix.lower() not in {".md", ".txt", ".csv"}:
                    continue
                raw = self._read_file(path)
                for idx, piece in enumerate(chunk_text(raw, self.chunk_size, self.overlap)):
                    rows.append({
                        "chunk_id": f"{path.stem}_{idx:04d}",
                        "doc_id": path.stem,
                        "source_file": path.name,
                        "text": piece,
                    })

            self.chunks = rows
            if not rows:
                self.word_vectorizer = self.char_vectorizer = None
                self.word_matrix = self.char_matrix = None
                self.save()
                return

            corpus = [row["text"] for row in rows]
            self.word_vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                min_df=1,
                max_features=40000,
            )
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=1,
                max_features=60000,
            )
            self.word_matrix = self.word_vectorizer.fit_transform(corpus)
            self.char_matrix = self.char_vectorizer.fit_transform(corpus)
            self.save()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "chunks": self.chunks,
            "word_vectorizer": self.word_vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "word_matrix": self.word_matrix,
            "char_matrix": self.char_matrix,
        }, self.index_path)

    def load_or_rebuild(self) -> None:
        with self.lock:
            if self.index_path.exists():
                try:
                    obj = joblib.load(self.index_path)
                    self.chunks = obj.get("chunks", [])
                    self.word_vectorizer = obj.get("word_vectorizer")
                    self.char_vectorizer = obj.get("char_vectorizer")
                    self.word_matrix = obj.get("word_matrix")
                    self.char_matrix = obj.get("char_matrix")
                    return
                except Exception:
                    pass
            self.rebuild()

    def status(self) -> tuple[int, int, list[str]]:
        sources = sorted({item["source_file"] for item in self.chunks})
        return len(sources), len(self.chunks), sources

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query = normalize_text(query)
        if not query or not self.chunks or self.word_vectorizer is None:
            return []

        with self.lock:
            wq = self.word_vectorizer.transform([query])
            cq = self.char_vectorizer.transform([query])
            w_score = cosine_similarity(wq, self.word_matrix).ravel()
            c_score = cosine_similarity(cq, self.char_matrix).ravel()
            scores = 0.45 * w_score + 0.55 * c_score
            order = np.argsort(-scores)

            selected, selected_tokens = [], []
            for index in order:
                if scores[index] <= 0:
                    continue
                item = self.chunks[int(index)]
                tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", item["text"].lower()))
                if any((len(tokens & old) / max(1, len(tokens | old))) > 0.72 for old in selected_tokens):
                    continue
                selected.append(RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    source_file=item["source_file"],
                    text=item["text"],
                    score=round(float(scores[index]), 4),
                ))
                selected_tokens.append(tokens)
                if len(selected) >= top_k:
                    break
            return selected

    @staticmethod
    def format_evidence(items: list[RetrievedChunk]) -> str:
        if not items:
            return "无可用的本地知识库证据。"
        return "\n\n".join(
            f"【证据{i}｜来源：{item.source_file}｜相关度：{item.score:.3f}】\n{item.text}"
            for i, item in enumerate(items, 1)
        )
