"""Fitness evaluator — A/B testing with LLM-as-judge scoring.

Scores mutation candidates on correctness, procedure following, and conciseness
using a configurable LLM judge (default: Ollama for $0 cost).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import TYPE_CHECKING

from godspeed.evolution.mutator import MutationCandidate

if TYPE_CHECKING:
    from godspeed.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Raw verdict from the LLM judge comparing two outputs."""

    a_correctness: float
    a_procedure: float
    a_conciseness: float
    b_correctness: float
    b_procedure: float
    b_conciseness: float
    winner: str  # "a" | "b" | "tie"


@dataclasses.dataclass(frozen=True, slots=True)
class FitnessScore:
    """Multi-dimensional fitness score for a mutation candidate."""

    correctness: float  # 0-1
    procedure_following: float  # 0-1
    conciseness: float  # 0-1
    overall: float  # weighted composite
    length_penalty: float  # 0-1 penalty for size growth
    confidence: float  # 0-1 based on number of test cases evaluated


# Weights for the composite score
CORRECTNESS_WEIGHT = 0.5
PROCEDURE_WEIGHT = 0.3
CONCISENESS_WEIGHT = 0.2

# Maximum allowed growth ratio before length penalty kicks in
MAX_LENGTH_RATIO = 2.0

JUDGE_PROMPT = """\
Compare two versions of a coding agent artifact for the same purpose.

Purpose: {purpose}

Version A (original):
{original}

Version B (mutated):
{mutated}

Score each version on these criteria (0-10 scale):
1. Correctness: Is it factually correct, complete, and free of errors?
2. Procedure: Does it guide the agent through the right steps in the right order?
3. Conciseness: Is it appropriately sized — not too verbose, not missing info?

Return ONLY a JSON object with these exact keys:
{{"a_correctness": N, "a_procedure": N, "a_conciseness": N, \
"b_correctness": N, "b_procedure": N, "b_conciseness": N, \
"winner": "a"|"b"|"tie"}}"""


# ---------------------------------------------------------------------------
# Fitness Evaluator
# ---------------------------------------------------------------------------


class FitnessEvaluator:
    """Score mutation candidates using LLM-as-judge."""

    def __init__(self, judge_model: str = "", llm_client: LLMClient | None = None) -> None:
        from godspeed.evolution.hardware import select_evolution_model

        self._judge_model = select_evolution_model(judge_model)
        self._llm_client = llm_client

    @property
    def judge_model(self) -> str:
        return self._judge_model

    async def evaluate(
        self,
        candidate: MutationCandidate,
        test_cases: list[str] | None = None,
    ) -> FitnessScore:
        """Evaluate a mutation candidate's fitness.

        Args:
            candidate: The mutation to evaluate.
            test_cases: Optional list of task descriptions to test against.
                If None, uses the candidate's own context for evaluation.

        Returns:
            FitnessScore with multi-dimensional scores.
        """
        cases = test_cases or [f"Evaluate {candidate.artifact_type} for {candidate.artifact_id}"]

        # Run judge calls concurrently for all test cases
        async def judge_case(case: str) -> JudgeVerdict | None:
            try:
                return await self._judge(
                    purpose=case,
                    original=candidate.original,
                    mutated=candidate.mutated,
                )
            except Exception:
                logger.warning(
                    "Judge call failed case=%s artifact=%s",
                    case[:50],
                    candidate.artifact_id,
                    exc_info=True,
                )
                return None

        verdicts_raw = await asyncio.gather(*(judge_case(c) for c in cases))
        verdicts = [v for v in verdicts_raw if v is not None]

        if not verdicts:
            return FitnessScore(
                correctness=0.0,
                procedure_following=0.0,
                conciseness=0.0,
                overall=0.0,
                length_penalty=0.0,
                confidence=0.0,
            )

        return self._aggregate_verdicts(verdicts, candidate, len(cases))

    async def _judge(
        self,
        purpose: str,
        original: str,
        mutated: str,
    ) -> JudgeVerdict | None:
        """Call the LLM judge to compare original vs mutated."""
        prompt = JUDGE_PROMPT.format(
            purpose=purpose,
            original=original,
            mutated=mutated,
        )

        try:
            response_text = await self._call_llm(prompt)
            return self._parse_verdict(response_text)
        except Exception:
            logger.debug("Failed to parse judge response", exc_info=True)
            return None

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured judge LLM. Reuses client when provided."""
        if self._llm_client is not None:
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm_client.chat(messages)
            return response.content

        from godspeed.llm.client import LLMClient

        client = LLMClient(model=self._judge_model)
        messages = [{"role": "user", "content": prompt}]
        response = await client.chat(messages)
        return response.content

    @staticmethod
    def _parse_verdict(text: str) -> JudgeVerdict | None:
        """Parse LLM response into a JudgeVerdict."""
        # Try to find JSON in the response
        text = text.strip()

        # Handle responses wrapped in markdown code blocks
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # Find the JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

        required_keys = {
            "a_correctness",
            "a_procedure",
            "a_conciseness",
            "b_correctness",
            "b_procedure",
            "b_conciseness",
            "winner",
        }
        if not required_keys.issubset(data.keys()):
            return None

        def _clamp(val: float) -> float:
            try:
                return max(0.0, min(10.0, float(val)))
            except (ValueError, TypeError):
                logger.debug("Non-numeric judge score value: %s", val)
                return 0.0

        winner = str(data["winner"]).lower()
        if winner not in ("a", "b", "tie"):
            winner = "tie"

        return JudgeVerdict(
            a_correctness=_clamp(data["a_correctness"]),
            a_procedure=_clamp(data["a_procedure"]),
            a_conciseness=_clamp(data["a_conciseness"]),
            b_correctness=_clamp(data["b_correctness"]),
            b_procedure=_clamp(data["b_procedure"]),
            b_conciseness=_clamp(data["b_conciseness"]),
            winner=winner,
        )

    @staticmethod
    def _aggregate_verdicts(
        verdicts: list[JudgeVerdict],
        candidate: MutationCandidate,
        total_cases: int,
    ) -> FitnessScore:
        """Aggregate multiple verdicts into a single FitnessScore."""
        n = len(verdicts)

        # Average the B scores (mutated version) normalized to 0-1
        avg_correctness = sum(v.b_correctness for v in verdicts) / (n * 10)
        avg_procedure = sum(v.b_procedure for v in verdicts) / (n * 10)
        avg_conciseness = sum(v.b_conciseness for v in verdicts) / (n * 10)

        # Length penalty
        orig_len = len(candidate.original)
        mut_len = len(candidate.mutated)
        length_ratio = mut_len / orig_len if orig_len > 0 else 1.0
        length_penalty = max(0.0, (length_ratio - MAX_LENGTH_RATIO) / MAX_LENGTH_RATIO)
        length_penalty = min(1.0, length_penalty)

        # Weighted composite
        overall = (
            CORRECTNESS_WEIGHT * avg_correctness
            + PROCEDURE_WEIGHT * avg_procedure
            + CONCISENESS_WEIGHT * avg_conciseness
            - length_penalty * 0.2  # Penalty reduces overall by up to 0.2
        )
        overall = max(0.0, min(1.0, overall))

        # Confidence based on how many test cases were successfully evaluated
        confidence = n / total_cases if total_cases > 0 else 0.0

        return FitnessScore(
            correctness=round(avg_correctness, 4),
            procedure_following=round(avg_procedure, 4),
            conciseness=round(avg_conciseness, 4),
            overall=round(overall, 4),
            length_penalty=round(length_penalty, 4),
            confidence=round(confidence, 4),
        )
