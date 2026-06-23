from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import Doc, Hit


IndexKind = Literal["embedding", "tfidf"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class TfidfIndex:
    kind: IndexKind = "tfidf"

    def __init__(self, *, vectorizer: TfidfVectorizer, matrix, docs: list[Doc]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.docs = docs

    @classmethod
    def build(cls, docs: list[Doc]) -> "TfidfIndex":
        texts = [d.text for d in docs]
        v = TfidfVectorizer(
            max_features=300_000,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        )
        m = v.fit_transform(texts)
        return cls(vectorizer=v, matrix=m, docs=docs)

    def search(self, query: str, *, k: int = 10) -> list[Hit]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        if k <= 0:
            return []
        idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        hits: list[Hit] = []
        for i in idx:
            d = self.docs[int(i)]
            hits.append(
                Hit(
                    doc_id=d.doc_id,
                    score=float(sims[int(i)]),
                    source=d.source,
                    title=d.title,
                    uri=d.uri,
                    text=d.text,
                    meta=d.meta,
                )
            )
        return hits

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        _ensure_dir(out)
        (out / "kind.txt").write_text("tfidf", encoding="utf-8")
        # vectorizer (pickle via joblib to avoid non-JSON params)
        import joblib

        joblib.dump(self.vectorizer, out / "vectorizer.joblib")
        # docs + sparse matrix
        with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        from scipy import sparse  # scikit-learn dependency

        sparse.save_npz(out / "matrix.npz", self.matrix)

    @classmethod
    def load(cls, in_dir: str) -> "TfidfIndex":
        p = Path(in_dir)
        from scipy import sparse

        kind = (p / "kind.txt").read_text(encoding="utf-8").strip()
        if kind != "tfidf":
            raise ValueError(f"Not a tfidf index: {kind}")
        import joblib

        v = joblib.load(p / "vectorizer.joblib")
        m = sparse.load_npz(p / "matrix.npz")
        docs: list[Doc] = []
        with (p / "docs.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                docs.append(Doc(**d))
        return cls(vectorizer=v, matrix=m, docs=docs)


class EmbeddingIndex:
    kind: IndexKind = "embedding"

    def __init__(self, *, vectors: np.ndarray, docs: list[Doc]):
        # vectors normalized (L2)
        self.vectors = vectors.astype(np.float32)
        self.docs = docs

    @classmethod
    def build(cls, docs: list[Doc], *, embeddings: list[list[float]]) -> "EmbeddingIndex":
        vec = np.asarray(embeddings, dtype=np.float32)
        if vec.ndim != 2 or vec.shape[0] != len(docs):
            raise ValueError("Embedding shape mismatch with docs")
        # L2 normalize for cosine via dot
        denom = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        vec = vec / denom
        return cls(vectors=vec, docs=docs)

    def search(self, query_embedding: list[float], *, k: int = 10) -> list[Hit]:
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (self.vectors @ q).ravel()
        if k <= 0:
            return []
        idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        hits: list[Hit] = []
        for i in idx:
            d = self.docs[int(i)]
            hits.append(
                Hit(
                    doc_id=d.doc_id,
                    score=float(sims[int(i)]),
                    source=d.source,
                    title=d.title,
                    uri=d.uri,
                    text=d.text,
                    meta=d.meta,
                )
            )
        return hits

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        _ensure_dir(out)
        (out / "kind.txt").write_text("embedding", encoding="utf-8")
        np.save(out / "vectors.npy", self.vectors)
        with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, in_dir: str) -> "EmbeddingIndex":
        p = Path(in_dir)
        kind = (p / "kind.txt").read_text(encoding="utf-8").strip()
        if kind != "embedding":
            raise ValueError(f"Not an embedding index: {kind}")
        vectors = np.load(p / "vectors.npy")
        docs: list[Doc] = []
        with (p / "docs.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                docs.append(Doc(**d))
        return cls(vectors=vectors, docs=docs)

