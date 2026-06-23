"""
Training-free GRPO–style experience accumulation for VaseAgent.

Roll out multiple samples per problem on a practice JSONL, compute rewards with
ground-truth references, then distill comparative insights into a persistent
experience pool (same high-level pipeline as youtu-agent ``TrainingFreeGRPO``).

Import from ``training_free_grpo`` with ``scripts/`` on ``PYTHONPATH``, or run
``python scripts/run_tf_grpo_accumulate.py`` from ``vase-agent/``.
"""

from .control_profiles import CONTROL_PROFILES, ControlProfile, apply_control_profile_to_env
from .pipeline import AccumulationConfig, format_prompt_block, run_accumulation

__all__ = [
    "CONTROL_PROFILES",
    "ControlProfile",
    "apply_control_profile_to_env",
    "AccumulationConfig",
    "format_prompt_block",
    "run_accumulation",
]
