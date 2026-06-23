python scripts/run_paper_experiments.py \
  --eval-jsonl ../dataset/data/v_only_100_flat.jsonl \
  --out-dir runs/exp_v_only \
  --methods direct,neither_control,source_control_only,response_control_only,vase_full \
  --max-samples 100 \
  --resume \
  --judge \
  --judge-workers 4 \
  --log-jsonl \
  # --judge-now \