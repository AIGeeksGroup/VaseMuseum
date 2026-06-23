"""
Embedded practice prompts (adapted from youtu-agent ``practice/experience.yaml``).

Uses plain ``str.format`` placeholders: ``agent_objective``, ``learning_objective``,
``question``, ``answer``, ``trajectory``, ``critique``, ``trajectories``, ``num_experiences``,
``existing_experiences``, ``new_experiences``, ``experiences_and_operations``.
"""

from __future__ import annotations

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP = """Your goal is to extract useful supervision signals from ONE agent trajectory for \
the learning objective.

<Working Agent Objective>
{agent_objective}
</Working Agent Objective>
<Learning Objective>
{learning_objective}
</Learning Objective>

Analyze tool usage, reasoning, and outcomes; relate everything to the learning objective (not necessarily identical to the agent objective).

Output a structured summary with:
- Step-by-step actions (tools, params, key tool outputs)
- Missed opportunities vs ground truth
- Final answer recap
Keep language consistent with the trajectory."""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP = """<Working Agent Input>
{question}
</Working Agent Input>

<Ground Truth>
{answer}
</Ground Truth>

<Critique>
{critique}
</Critique>

<Trajectory>
{trajectory}
</Trajectory>

Follow the system instructions."""

SINGLE_QUERY_GROUP_ADVANTAGE_SP = """Your goal is to distill transferable experiences from MULTIPLE attempts at the SAME task.

<Working Agent Objective>
{agent_objective}
</Working Agent Objective>
<Learning Objective>
{learning_objective}
</Learning Objective>

Compare attempts (rewards given). Extract principles that improve future behavior with respect to the learning objective.

You MUST include a block:
<Experiences>
1. ...
2. ...
</Experiences>

Include AT MOST {num_experiences} numbered items inside <Experiences>."""

SINGLE_QUERY_GROUP_ADVANTAGE_UP = """<Agent Input>
{question}
</Agent Input>

<Ground Truth>
{answer}
</Ground Truth>

<Trajectories>
{trajectories}
</Trajectories>

Follow the system instructions."""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP = """You maintain a compact experience pool for the learning objective.

<Working Agent Objective>
{agent_objective}
</Working Agent Objective>
<Learning Objective>
{learning_objective}
</Learning Objective>

Given existing experiences and NEW experiences text for ONE query group, propose JSON operations.

Return ONLY a JSON array; each element MUST look like:
{{"operation": "ADD" | "UPDATE" | "DELETE" | "NONE", "id": "<existing id or null>", "content": "Short name: one sentence."}}

Rules:
- ADD when novel and useful
- UPDATE when refining an existing ID
- DELETE when harmful/outdated (reference id)
- NONE when redundant"""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP = """<Existing Experiences>
{existing_experiences}
</Existing Experiences>

<New Experiences>
{new_experiences}
</New Experiences>

Return ONLY the JSON array."""

BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP = """Merge a BATCH of proposed operations into a net revision plan for the pool.

<Working Agent Objective>
{agent_objective}
</Working Agent Objective>
<Learning Objective>
{learning_objective}
</Learning Objective>

Return ONLY a JSON array of operations with fields operation/id/content as in group update.
Prefer merging duplicates; DELETE overrides UPDATE for the same id."""

BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP = """<Experiences and Proposed Operations>
{experiences_and_operations}
</Experiences and Proposed Operations>

Return ONLY the JSON array."""
