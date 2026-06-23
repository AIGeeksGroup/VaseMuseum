python scripts/run_tf_grpo_accumulate.py \
    --practice-jsonl ../dataset/data/grpo_tf_600_flat.jsonl \
    --out-dir runs/tf_grpo_neither_control_1 \
    --control neither_control \
    --grpo-n 4 \
    --experience-batch-size 8 \
    --max-rows 24 \
    --epochs 5 \
    # --no-partial-only