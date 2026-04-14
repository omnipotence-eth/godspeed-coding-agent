"""Safety gate — prevent regressions from evolved artifacts.

All mutations must pass safety checks before being applied:
- Test suite still passes (100%)
- Size limits respected (no >2x growth)
- Semantic drift within bounds
- Fitness above threshold
- High-impact changes flagged for human review
"""

from __future__ import annotations

import dataclasses
import logging
import re

from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """Result of running all safety checks on a mutation candidate."""

    passed: bool
    checks: tuple[tuple[str, bool, str], ...]  # (check_name, passed, reason)
    requires_human_review: bool


# Artifacts that always need human review
HIGH_IMPACT_ARTIFACT_TYPES = frozenset({"prompt_section"})
HIGH_IMPACT_ARTIFACT_IDS = frozenset({"core", "security", "permissions"})


# ---------------------------------------------------------------------------
# Safety Gate
# ---------------------------------------------------------------------------


class SafetyGate:
    """Run safety checks on mutation candidates before applying them."""

    def __init__(
        self,
        max_growth: float = 2.0,
        min_similarity: float = 0.3,
        min_fitness: float = 0.6,
    ) -> None:
        self._max_growth = max_growth
        self._min_similarity = min_similarity
        self._min_fitness = min_fitness

    def gate(
        self,
        candidate: MutationCandidate,
        score: FitnessScore,
    ) -> SafetyVerdict:
        """Run all safety checks and return a verdict.

        Does NOT run the test suite (that's an async operation handled
        separately). This gate covers the fast, synchronous checks.
        """
        checks: list[tuple[str, bool, str]] = []

        # Check 1: Size limit
        size_ok, size_msg = self.check_size_limit(candidate)
        checks.append(("size_limit", size_ok, size_msg))

        # Check 2: Semantic drift
        drift_ok, drift_msg = self.check_semantic_drift(candidate)
        checks.append(("semantic_drift", drift_ok, drift_msg))

        # Check 3: Fitness threshold
        fitness_ok, fitness_msg = self.check_fitness_threshold(score)
        checks.append(("fitness_threshold", fitness_ok, fitness_msg))

        # Check 4: Confidence threshold
        conf_ok = score.confidence >= 0.5
        conf_msg = f"confidence={score.confidence:.2f} (min=0.50)"
        checks.append(("confidence", conf_ok, conf_msg))

        all_passed = all(ok for _, ok, _ in checks)
        needs_review = self.requires_human_review(candidate)

        return SafetyVerdict(
            passed=all_passed,
            checks=tuple(checks),
            requires_human_review=needs_review,
        )

    def check_size_limit(self, candidate: MutationCandidate) -> tuple[bool, str]:
        """Check that mutated text is not excessively larger than original."""
        orig_len = len(candidate.original)
        mut_len = len(candidate.mutated)

        if orig_len == 0:
            return True, "original is empty — no size check"

        ratio = mut_len / orig_len
        passed = ratio <= self._max_growth
        msg = f"size ratio={ratio:.2f} (max={self._max_growth:.1f})"
        return passed, msg

    def check_semantic_drift(self, candidate: MutationCandidate) -> tuple[bool, str]:
        """Check that mutated text hasn't drifted too far from original.

        Uses word-overlap Jaccard similarity (no embeddings needed).
        """
        orig_words = self._tokenize(candidate.original)
        mut_words = self._tokenize(candidate.mutated)

        if not orig_words and not mut_words:
            return True, "both empty"
        if not orig_words or not mut_words:
            return False, "one side is empty"

        intersection = orig_words & mut_words
        union = orig_words | mut_words
        similarity = len(intersection) / len(union) if union else 0.0

        passed = similarity >= self._min_similarity
        msg = f"similarity={similarity:.3f} (min={self._min_similarity:.2f})"
        return passed, msg

    def check_fitness_threshold(self, score: FitnessScore) -> tuple[bool, str]:
        """Check that fitness score meets minimum threshold."""
        passed = score.overall >= self._min_fitness
        msg = f"overall={score.overall:.3f} (min={self._min_fitness:.2f})"
        return passed, msg

    def requires_human_review(self, candidate: MutationCandidate) -> bool:
        """Determine if this mutation needs human approval."""
        if candidate.artifact_type in HIGH_IMPACT_ARTIFACT_TYPES:
            return True
        return candidate.artifact_id in HIGH_IMPACT_ARTIFACT_IDS

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple word tokenization for similarity comparison."""
        return set(re.findall(r"\w+", text.lower()))
