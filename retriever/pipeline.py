from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from .build_corpus import doc_from_beazley_desc_row, doc_from_limc_row
from .caption import structured_caption_beazley_desc, structured_caption_limc
from .index import EmbeddingIndex, TfidfIndex
from .io_csv import iter_csv_rows
from .local_llm import embed_texts, nl_caption
from .types import Doc, Hit


CaptionMode = Literal["structured", "nl"]
IndexMode = Literal["auto", "embedding", "tfidf"]


def build_docs(
    *,
    limc_csv_path: str,
    beazley_desc_csv_path: str,
    caption_mode: CaptionMode,
    max_docs: int | None = None,
    nl_model_max_docs: int | None = None,
    nl_extra_instructions: str = "",
) -> list[Doc]:
    docs: list[Doc] = []

    if caption_mode == "structured":
        cap = max_docs if (max_docs is not None and max_docs > 0) else None
        seen = 0
        for row in iter_csv_rows(limc_csv_path):
            if cap is not None and seen >= cap:
                return docs
            docs.append(doc_from_limc_row(row, caption_mode="structured"))
            seen += 1
        for row in iter_csv_rows(beazley_desc_csv_path):
            if cap is not None and seen >= cap:
                return docs
            docs.append(doc_from_beazley_desc_row(row, caption_mode="structured"))
            seen += 1
        return docs

    # nl caption: call local model; optionally cap doc count for speed.
    max_docs = nl_model_max_docs if (nl_model_max_docs is not None and nl_model_max_docs > 0) else None

    def _iter_all_rows():
        for row in iter_csv_rows(limc_csv_path):
            yield "LIMC", row
        for row in iter_csv_rows(beazley_desc_csv_path):
            yield "BEAZLEY_DESC", row

    seen = 0
    it = _iter_all_rows()
    for source, row in tqdm(it, desc="nl-caption"):
        if max_docs is not None and seen >= max_docs:
            break
        if source == "LIMC":
            structured = structured_caption_limc(row)
            cap = nl_caption(structured_caption=structured, extra_instructions=nl_extra_instructions)
            docs.append(doc_from_limc_row(row, caption_mode="nl", nl_caption_text=cap))
        else:
            structured = structured_caption_beazley_desc(row)
            cap = nl_caption(structured_caption=structured, extra_instructions=nl_extra_instructions)
            docs.append(doc_from_beazley_desc_row(row, caption_mode="nl", nl_caption_text=cap))
        seen += 1

    return docs


def build_index(
    *,
    docs: list[Doc],
    out_dir: str,
    index_mode: IndexMode = "auto",
    embedding_model: str = "text-embedding-3-large",
    embedding_batch_size: int = 128,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save raw docs for traceability.
    with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    chosen = index_mode
    if chosen == "auto":
        chosen = "embedding"

    if chosen == "embedding":
        try:
            embs: list[list[float]] = []
            texts = [d.text for d in docs]
            for i in tqdm(range(0, len(texts), embedding_batch_size), desc="embed"):
                batch = texts[i : i + embedding_batch_size]
                embs.extend(embed_texts(texts=batch, model=embedding_model))
            idx = EmbeddingIndex.build(docs, embeddings=embs)
            idx.save(str(out / "index"))
            return {"ok": True, "index_kind": "embedding", "out_dir": str(out)}
        except Exception as e:
            if index_mode == "embedding":
                raise
            # fall back
            chosen = "tfidf"

    idx2 = TfidfIndex.build(docs)
    idx2.save(str(out / "index"))
    return {"ok": True, "index_kind": "tfidf", "out_dir": str(out)}


def load_index(index_dir: str):
    # accept either the base build dir or the inner index dir
    p0 = Path(index_dir)
    p = p0 if (p0 / "kind.txt").exists() else (p0 / "index")
    kind = (p / "kind.txt").read_text(encoding="utf-8").strip()
    if kind == "embedding":
        return EmbeddingIndex.load(str(p))
    if kind == "tfidf":
        return TfidfIndex.load(str(p))
    raise ValueError(f"Unknown index kind: {kind}")

