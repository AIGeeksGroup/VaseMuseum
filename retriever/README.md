## Museum retriever (LIMC + Beazley descriptions)

This folder builds a local retriever over:

- `dataset/kb/LIMC.csv`
- `dataset/kb/descriptions.csv`

It supports two caption modes:

- **structured**: deterministic template caption from CSV fields
- **nl**: call your local OpenAI-compatible LLM on port **8001** to rewrite the structured caption into a 1–3 sentence natural-language caption

### Build (structured caption)

```bash
python -m retriever.cli build \
  --limc-csv "dataset/kb/LIMC.csv" \
  --descriptions-csv "dataset/kb/descriptions.csv" \
  --caption-mode structured \
  --max-docs 5000 \
  --out "retriever_runs/structured_full" \
  --index-mode auto
```

### Build (nl caption; you can cap docs while iterating)

```bash
python -m retriever.cli build \
  --limc-csv "dataset/kb/LIMC.csv" \
  --descriptions-csv "dataset/kb/descriptions.csv" \
  --caption-mode nl \
  --nl-max-docs 500 \
  --out "retriever_runs/nl_500" \
  --index-mode tfidf
```

### Query

```bash
python -m retriever.cli query \
  --index-dir "retriever_runs/structured_full" \
  --query "Triptolemos on a winged chariot between Demeter and Persephone" \
  -k 5 \
  --include-text
```

### Local model config (port 8001)

Defaults match `vase-agent/llm_env.py`:

- `LLM_HOST` (default `127.0.0.1`)
- `LLM_PORT` (default `8001`)
- `LLM_BASE_URL` (optional; should end with `/v1`)
- `LLM_MODEL` (chat model name/id)
- `LLM_API_KEY` (default `EMPTY`)

For embeddings, set:

- `EMBEDDING_MODEL` (default in CLI is `text-embedding-3-large`)
