"""
Control presets aligned with ``scripts/run_paper_experiments.py`` agent methods.

During **practice / accumulation**, these flags shape rollouts; eval later can be
plain inference with experiences injected into the prompt only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlProfile:
    """Inference-time control switches for one ablation."""

    key: str
    label: str
    response_control_gate: bool
    apply_uncertain_preamble: bool
    disable_source_control: bool


CONTROL_PROFILES: dict[str, ControlProfile] = {
    "neither_control": ControlProfile(
        key="neither_control",
        label="+Tools — neither source nor response control",
        response_control_gate=False,
        apply_uncertain_preamble=False,
        disable_source_control=True,
    ),
    "source_control_only": ControlProfile(
        key="source_control_only",
        label="Source control only — evidence pool; no response gate",
        response_control_gate=False,
        apply_uncertain_preamble=False,
        disable_source_control=False,
    ),
    "response_control_only": ControlProfile(
        key="response_control_only",
        label="Response control only — gate + preamble; source pool off",
        response_control_gate=True,
        apply_uncertain_preamble=True,
        disable_source_control=True,
    ),
    "vase_full": ControlProfile(
        key="vase_full",
        label="Full — source + response control",
        response_control_gate=True,
        apply_uncertain_preamble=True,
        disable_source_control=False,
    ),
}


def apply_control_profile_to_env(profile: ControlProfile) -> None:
    """Mirror ``run_paper_experiments._set_source_control_env``."""
    if profile.disable_source_control:
        os.environ["VASE_DISABLE_SOURCE_CONTROL"] = "1"
    else:
        os.environ.pop("VASE_DISABLE_SOURCE_CONTROL", None)
