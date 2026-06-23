from __future__ import annotations

import argparse
import json
from pathlib import Path

from .local_llm import embed_texts
from .pipeline import build_docs, build_index, load_index


def cmd_build(args: argparse.Namespace) -> int:
    docs = build_docs(
        limc_csv_path=args.limc_csv,
        beazley_desc_csv_path=args.descriptions_csv,
        caption_mode=args.caption_mode,
        max_docs=args.max_docs if args.max_docs > 0 else None,
        nl_model_max_docs=args.nl_max_docs,
        nl_extra_instructions=args.nl_extra_instructions or "",
    )
    res = build_index(
        docs=docs,
        out_dir=args.out,
        index_mode=args.index_mode,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    idx = load_index(args.index_dir)
    k = args.k
    if getattr(idx, "kind", "") == "embedding":
        q_emb = embed_texts(texts=[args.query], model=args.embedding_model)[0]
        hits = idx.search(q_emb, k=k)
    else:
        hits = idx.search(args.query, k=k)

    out = []
    for h in hits:
        out.append(
            {
                "doc_id": h.doc_id,
                "score": h.score,
                "source": h.source,
                "title": h.title,
                "uri": h.uri,
                "meta": h.meta,
                "text": h.text if args.include_text else "",
            }
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retriever")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build corpus + index from CSVs")
    b.add_argument("--limc-csv", required=True)
    b.add_argument("--descriptions-csv", required=True)
    b.add_argument("--caption-mode", choices=["structured", "nl"], default="structured")
    b.add_argument("--max-docs", type=int, default=0, help="Cap total docs (0=all)")
    b.add_argument("--nl-max-docs", type=int, default=0, help="Cap docs for nl-caption (0=all)")
    b.add_argument("--nl-extra-instructions", default="")
    b.add_argument("--out", required=True, help="Output directory for built index")
    b.add_argument("--index-mode", choices=["auto", "embedding", "tfidf"], default="auto")
    b.add_argument("--embedding-model", default="text-embedding-3-large")
    b.add_argument("--embedding-batch-size", type=int, default=128)
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="Query an index")
    q.add_argument("--index-dir", required=True, help="Either build dir or build_dir/index")
    q.add_argument("--query", required=True)
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--embedding-model", default="text-embedding-3-large")
    q.add_argument("--include-text", action="store_true")
    q.set_defaults(func=cmd_query)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

