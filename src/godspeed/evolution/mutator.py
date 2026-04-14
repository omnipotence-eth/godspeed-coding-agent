"""Evolution engine — GEPA-style LLM-guided mutations.

Generates improved candidates for tool descriptions, system prompt sections,
and compaction prompts using failure analysis from the trace analyzer.
"""

from __future__ import annotations

import dataclasses
import logging

from godspeed.evolution.trace_analyzer import (
    EvolutionReport,
    ToolCall,
    ToolFailurePattern,
    ToolSequence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class MutationCandidate:
    """A proposed improvement to an agent artifact."""

    artifact_type: str  # "tool_description", "prompt_section", "compaction_prompt", "skill"
    artifact_id: str  # tool name, section name, etc.
    original: str
    mutated: str
    mutation_rationale: str
    model_used: str


@dataclasses.dataclass(frozen=True, slots=True)
class SkillCandidate:
    """A proposed auto-generated skill from repeated tool patterns."""

    name: str
    description: str
    trigger: str
    content: str
    source_sequence: ToolSequence


# ---------------------------------------------------------------------------
# Mutation prompts
# ---------------------------------------------------------------------------

TOOL_DESCRIPTION_MUTATION_PROMPT = """\
You are improving a coding agent's tool description. The description tells the LLM \
when and how to use this tool.

Tool name: {tool_name}
Current description:
{current_description}

Problems identified from usage data:
{failure_summary}

Generate an improved description that:
1. Addresses the specific failures listed above
2. Maintains the same core functionality
3. Is clear and unambiguous for an LLM to follow
4. Includes a brief example if the current version caused confusion

Return ONLY the improved description text. No explanation, no preamble."""

PROMPT_SECTION_MUTATION_PROMPT = """\
You are improving a section of a coding agent's system prompt.

Section name: {section_name}
Current text:
{current_text}

Usage analysis summary:
- Sessions analyzed: {sessions_analyzed}
- Overall error rate: {error_rate:.1%}
- Top failing tools: {top_failures}

Generate an improved version that reduces errors and improves clarity.
Return ONLY the improved text. No explanation, no preamble."""

COMPACTION_MUTATION_PROMPT = """\
You are improving a coding agent's conversation compaction prompt. This prompt \
is used to summarize long conversations so they fit in the context window.

Current compaction prompt:
{current_prompt}

Quality feedback (0-1 scores from recent compactions):
{quality_scores}

Generate an improved compaction prompt that produces higher-quality summaries.
Return ONLY the improved prompt text. No explanation, no preamble."""

TOOL_EXAMPLES_PROMPT = """\
Based on these successful tool calls, generate 2-3 concise usage examples \
for the tool '{tool_name}'.

Successful calls:
{examples}

Format each example as:
- Example: <one-line description>
  Arguments: <JSON arguments>

Return ONLY the examples. No explanation."""

SKILL_GENERATION_PROMPT = """\
The following tool sequence is used frequently ({frequency} times across sessions):
Tools: {tool_chain}

Example successful executions:
{example_traces}

Generate a reusable skill definition in this format:
---
name: <kebab-case-name>
description: <one line>
trigger: <trigger word>
---

<Step-by-step instructions using the tools above>

Return ONLY the skill definition. No explanation."""


# ---------------------------------------------------------------------------
# Evolution Engine
# ---------------------------------------------------------------------------


class EvolutionEngine:
    """Generate improved candidates using LLM-guided mutations.

    Uses any LiteLLM-compatible model. Default: auto-detected based on
    available VRAM (from RTX 5070 Ti 16GB down to Jetson Orin Nano 8GB shared).
    """

    def __init__(self, model: str = "") -> None:
        from godspeed.evolution.hardware import select_evolution_model

        self._model = select_evolution_model(model)

    @property
    def model(self) -> str:
        return self._model

    async def mutate_tool_description(
        self,
        tool_name: str,
        current_desc: str,
        failure_patterns: list[ToolFailurePattern],
        num_candidates: int = 3,
    ) -> list[MutationCandidate]:
        """Generate improved tool descriptions based on failure analysis.

        Args:
            tool_name: Name of the tool to improve.
            current_desc: Current tool description.
            failure_patterns: Failures specific to this tool.
            num_candidates: Number of candidates to generate.

        Returns:
            List of MutationCandidate objects.
        """
        if not failure_patterns:
            logger.debug("No failures for tool=%s — skipping mutation", tool_name)
            return []

        failure_summary = self._format_failures(failure_patterns)
        prompt = TOOL_DESCRIPTION_MUTATION_PROMPT.format(
            tool_name=tool_name,
            current_description=current_desc,
            failure_summary=failure_summary,
        )

        candidates: list[MutationCandidate] = []
        for i in range(num_candidates):
            try:
                mutated = await self._call_llm(prompt)
                if mutated and mutated.strip() != current_desc.strip():
                    candidates.append(
                        MutationCandidate(
                            artifact_type="tool_description",
                            artifact_id=tool_name,
                            original=current_desc,
                            mutated=mutated.strip(),
                            mutation_rationale=(
                                f"Address {len(failure_patterns)} failure"
                                f" pattern(s): {failure_summary[:200]}"
                            ),
                            model_used=self._model,
                        )
                    )
            except Exception:
                logger.warning(
                    "Mutation generation failed tool=%s attempt=%d",
                    tool_name,
                    i,
                    exc_info=True,
                )

        return candidates

    async def mutate_prompt_section(
        self,
        section_name: str,
        current_text: str,
        report: EvolutionReport,
        num_candidates: int = 3,
    ) -> list[MutationCandidate]:
        """Generate improved system prompt sections based on usage report."""
        top_failures = ", ".join(
            f"{f.tool_name} ({f.error_category}: {f.frequency}x)" for f in report.tool_failures[:5]
        )

        prompt = PROMPT_SECTION_MUTATION_PROMPT.format(
            section_name=section_name,
            current_text=current_text,
            sessions_analyzed=report.sessions_analyzed,
            error_rate=report.error_rate,
            top_failures=top_failures or "none",
        )

        candidates: list[MutationCandidate] = []
        for i in range(num_candidates):
            try:
                mutated = await self._call_llm(prompt)
                if mutated and mutated.strip() != current_text.strip():
                    candidates.append(
                        MutationCandidate(
                            artifact_type="prompt_section",
                            artifact_id=section_name,
                            original=current_text,
                            mutated=mutated.strip(),
                            mutation_rationale=(
                                f"Reduce error rate from {report.error_rate:.1%}"
                                f" across {report.sessions_analyzed} sessions"
                            ),
                            model_used=self._model,
                        )
                    )
            except Exception:
                logger.warning(
                    "Prompt mutation failed section=%s attempt=%d",
                    section_name,
                    i,
                    exc_info=True,
                )

        return candidates

    async def mutate_compaction_prompt(
        self,
        current_prompt: str,
        quality_scores: list[float],
        num_candidates: int = 3,
    ) -> list[MutationCandidate]:
        """Generate improved compaction prompts based on quality feedback."""
        scores_str = ", ".join(f"{s:.2f}" for s in quality_scores[-10:])
        avg = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        prompt = COMPACTION_MUTATION_PROMPT.format(
            current_prompt=current_prompt,
            quality_scores=f"Recent scores: [{scores_str}], average: {avg:.2f}",
        )

        candidates: list[MutationCandidate] = []
        for i in range(num_candidates):
            try:
                mutated = await self._call_llm(prompt)
                if mutated and mutated.strip() != current_prompt.strip():
                    candidates.append(
                        MutationCandidate(
                            artifact_type="compaction_prompt",
                            artifact_id="compaction",
                            original=current_prompt,
                            mutated=mutated.strip(),
                            mutation_rationale=f"Improve compaction quality from avg {avg:.2f}",
                            model_used=self._model,
                        )
                    )
            except Exception:
                logger.warning("Compaction mutation failed attempt=%d", i, exc_info=True)

        return candidates

    async def generate_tool_examples(
        self,
        tool_name: str,
        successful_traces: list[ToolCall],
        max_examples: int = 5,
    ) -> list[str]:
        """Create usage examples from successful tool call traces."""
        if not successful_traces:
            return []

        examples_text = "\n".join(
            f"- {tc.tool_name}({tc.arguments}) → {tc.output_length} chars, {tc.latency_ms:.0f}ms"
            for tc in successful_traces[:max_examples]
        )

        prompt = TOOL_EXAMPLES_PROMPT.format(
            tool_name=tool_name,
            examples=examples_text,
        )

        try:
            result = await self._call_llm(prompt)
            if result:
                return [line.strip() for line in result.strip().split("\n") if line.strip()]
        except Exception:
            logger.warning("Example generation failed tool=%s", tool_name, exc_info=True)

        return []

    async def suggest_new_skill(
        self,
        sequence: ToolSequence,
        example_traces: list[list[ToolCall]],
    ) -> SkillCandidate | None:
        """Auto-generate a skill from a repeated tool sequence."""
        traces_text = ""
        for i, trace in enumerate(example_traces[:3]):
            traces_text += f"\nExecution {i + 1}:\n"
            for tc in trace:
                status = "error" if tc.is_error else "ok"
                traces_text += f"  {tc.tool_name}({tc.arguments}) → {status}\n"

        prompt = SKILL_GENERATION_PROMPT.format(
            frequency=sequence.frequency,
            tool_chain=" → ".join(sequence.tools),
            example_traces=traces_text,
        )

        try:
            result = await self._call_llm(prompt)
            if result:
                return self._parse_skill_candidate(result, sequence)
        except Exception:
            logger.warning("Skill generation failed sequence=%s", sequence.tools, exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM with a simple prompt. Returns response text."""
        from godspeed.llm.client import LLMClient

        client = LLMClient(model=self._model)
        messages = [{"role": "user", "content": prompt}]
        response = await client.chat(messages)
        return response.content

    @staticmethod
    def _format_failures(patterns: list[ToolFailurePattern]) -> str:
        """Format failure patterns into a readable summary."""
        lines: list[str] = []
        for p in patterns:
            lines.append(f"- {p.error_category} ({p.frequency}x): {p.suggested_fix}")
            if p.example_args:
                lines.append(f"  Example args: {p.example_args[0]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_skill_candidate(raw_text: str, sequence: ToolSequence) -> SkillCandidate | None:
        """Parse LLM output into a SkillCandidate."""
        import yaml

        # Find YAML frontmatter
        parts = raw_text.split("---")
        if len(parts) < 3:
            logger.debug("Skill output missing frontmatter delimiters")
            return None

        try:
            metadata = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            logger.debug("Invalid YAML in skill frontmatter")
            return None

        if not isinstance(metadata, dict):
            return None

        name = metadata.get("name", "")
        description = metadata.get("description", "")
        trigger = metadata.get("trigger", "")
        content = "---".join(parts[2:]).strip()

        if not all([name, description, trigger, content]):
            return None

        return SkillCandidate(
            name=name,
            description=description,
            trigger=trigger,
            content=content,
            source_sequence=sequence,
        )
