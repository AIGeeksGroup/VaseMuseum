#!/usr/bin/env python3
"""
CLI: training-free–style experience accumulation on a practice JSONL.

From ``vase-agent/``::

  python scripts/run_tf_grpo_accumulate.py \\
    --practice-jsonl ../dataset/data/grpo_tf_600_flat.jsonl \\
    --out-dir runs/tf_grpo_vase_full \\
    --control vase_full \\
    --grpo-n 4 \\
    --experience-batch-size 8 \\
    --max-rows 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_VASE_AGENT = _SCRIPTS_DIR.parent
if str(_VASE_AGENT) not in sys.path:
    sys.path.insert(0, str(_VASE_AGENT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def main() -> None:
    p = argparse.ArgumentParser(description="Training-free GRPO–style experience accumulation for VaseAgent")
    p.add_argument(
        "--practice-jsonl",
        type=Path,
        required=True,
        help="Practice JSONL (question + image + answer reference), e.g. dataset/data/grpo_tf_600_flat.jsonl",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for experiences + logs")
    p.add_argument(
        "--control",
        type=str,
        default="vase_full",
        choices=sorted(
            [
                "neither_control",
                "source_control_only",
                "response_control_only",
                "vase_full",
            ]
        ),
        help="Control ablation applied during rollouts (matches paper experiment presets)",
    )
    p.add_argument("--grpo-n", type=int, default=4, help="Rollouts per practice row (pass@K group size)")
    p.add_argument(
        "--experience-batch-size",
        type=int,
        default=8,
        help="Number of dataset rows per experience-update step (each row contributes grpo-n rollouts)",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows after --start")
    p.add_argument("--max-tool-rounds", type=int, default=10)
    p.add_argument("--rollout-temperature", type=float, default=0.7)
    p.add_argument("--rollout-concurrency", type=int, default=2)
    p.add_argument("--experience-concurrency", type=int, default=8)
    p.add_argument("--num-experiences-per-query", type=int, default=2)
    p.add_argument("--no-given-ground-truth", action="store_true", help="Hide refs from advantage prompts")
    p.add_argument("--tool-mode", choices=["openai", "xml", ""], default="", help="Override VASE_TOOL_MODE for agent")
    p.add_argument("--no-rollout-log", action="store_true", help="Skip practice_rollouts.jsonl")
    p.add_argument(
        "--no-partial-only",
        action="store_true",
        help="Distill experiences from every question group (not only mean reward in (0,1)); yields non-empty pools more often.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Discard checkpoint files in --out-dir (rollout_records.jsonl, steps_complete.json, run_state.json, experience_llm_traces.jsonl, experiences_epoch*.json) and run from scratch.",
    )
    p.add_argument(
        "--no-experience-llm-trace",
        action="store_true",
        help="Do not append experience-distillation meta-LLM chats to experience_llm_traces.jsonl.",
    )
    args = p.parse_args()

    from training_free_grpo.pipeline import AccumulationConfig, run_accumulation

    cfg = AccumulationConfig(
        practice_jsonl=args.practice_jsonl,
        out_dir=args.out_dir,
        control_key=args.control,
        grpo_n=max(1, args.grpo_n),
        experience_batch_size=max(1, args.experience_batch_size),
        epochs=max(1, args.epochs),
        shuffle=bool(args.shuffle),
        max_rows=max(0, args.max_rows),
        start=max(0, args.start),
        max_tool_rounds=max(1, args.max_tool_rounds),
        rollout_temperature=float(args.rollout_temperature),
        rollout_concurrency=max(1, args.rollout_concurrency),
        experience_concurrency=max(1, args.experience_concurrency),
        num_experiences_per_query=max(1, args.num_experiences_per_query),
        given_ground_truth=not args.no_given_ground_truth,
        tool_mode=args.tool_mode if args.tool_mode else None,
        log_rollouts_jsonl=not args.no_rollout_log,
        seed=args.seed,
        partial_groups_only=not args.no_partial_only,
        fresh=bool(args.fresh),
        resume=not bool(args.fresh),
        log_experience_llm_traces=not args.no_experience_llm_trace,
    )

    def _prog(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    exps = run_accumulation(cfg, progress=_prog)
    print(f"Final experiences count: {len(exps)} → {cfg.out_dir / 'experiences_final.json'}")


if __name__ == "__main__":
    main()
