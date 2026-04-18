"""Stage I — assemble the final Phase A1 training corpus.

Concatenates the four source streams into a single shuffled, deduplicated,
validated JSONL. Run this ONCE after ``make a1-run`` completes.

Sources (all expected under ``experiments/phase_a1/data/``):

  - ``phase_a1_full.jsonl``         synthetic (judge-gated, ~4-5k kept)
  - ``anchor_opus_50.jsonl``        50 Opus-hand-authored gold
  - ``phase_a1_swesmith_distilled.jsonl``   1500 re-inferred swesmith
  - ``phase_a1_augmented.jsonl``    200 param-shuffle rare-tool boost

What it does:

  1. Loads every source file, skipping any that don't exist.
  2. Validates each record via ``validate.validate_record`` — drops
     malformed records and records the drop count per source.
  3. Deduplicates by a normalized hash of the user prompt — keeps the
     first-seen copy so anchor and augment (earlier in merge order) win
     over synthetic/distill for near-duplicates.
  4. Shuffles with a deterministic seed.
  5. Writes a single ``phase_a1_final.jsonl`` plus a ``phase_a1_final.stats.json``
     summary (per-source counts, per-tool coverage, category mix).

The output of this step is the canonical file the ml-lab training reader
consumes. Seed is pinned so the file is bit-identical across re-runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.phase_a1.validate import validate_record

logger = logging.getLogger(__name__)


DEFAULT_SEED: int = 42


# Order matters: earlier sources win dedup ties because anchor + augment are
# the highest-quality streams and should be preferred over lossy distill /
# noisier synthetic.
DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    ("anchor", "anchor_opus_50.jsonl"),
    ("augment", "phase_a1_augmented.jsonl"),
    ("synthetic", "phase_a1_full.jsonl"),
    ("distill", "phase_a1_swesmith_distilled.jsonl"),
)


_WS_RE = re.compile(r"\s+")


@dataclass
class SourceStats:
    name: str
    file: Path
    loaded: int = 0
    invalid: int = 0
    duplicates: int = 0
    kept: int = 0


def _extract_user_prompt(record: dict[str, Any]) -> str:
    """First user message content, or empty string."""
    for msg in record.get("messages") or []:
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _prompt_hash(prompt: str) -> str:
    """Normalized SHA-1 of the user prompt for dedup.

    Normalization strips whitespace and lowercases so trivial formatting
    differences don't hide duplicates. The hash is deterministic across
    processes.
    """
    normalized = _WS_RE.sub(" ", prompt.strip().lower())
    # sha1 is fine here — non-cryptographic dedup key, collisions are only
    # a correctness issue at astronomical scale.
    return hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()


def _load_source(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("skipping invalid JSON in %s: %s", path, e)
    return records


def _infer_category(record: dict[str, Any]) -> str:
    n_calls = 0
    for msg in record.get("messages") or []:
        n_calls += len(msg.get("tool_calls") or [])
    if n_calls == 0:
        return "no_tool"
    if n_calls == 1:
        return "single_tool"
    return "multi_turn"


def assemble(
    data_dir: Path,
    output_path: Path,
    *,
    sources: tuple[tuple[str, str], ...] = DEFAULT_SOURCES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Merge, validate, dedup, shuffle, write."""
    seen_hashes: set[str] = set()
    kept: list[tuple[str, dict[str, Any]]] = []  # (source_name, record)
    source_stats: list[SourceStats] = []

    for name, filename in sources:
        path = data_dir / filename
        stats = SourceStats(name=name, file=path)
        records = _load_source(path)
        stats.loaded = len(records)

        for rec in records:
            errs, _, _ = validate_record(rec)
            if errs:
                stats.invalid += 1
                continue
            h = _prompt_hash(_extract_user_prompt(rec))
            if h in seen_hashes:
                stats.duplicates += 1
                continue
            seen_hashes.add(h)
            kept.append((name, rec))
            stats.kept += 1

        source_stats.append(stats)
        logger.info(
            "source=%s file=%s  loaded=%d invalid=%d dup=%d kept=%d",
            stats.name,
            stats.file.name,
            stats.loaded,
            stats.invalid,
            stats.duplicates,
            stats.kept,
        )

    rng = random.Random(seed)
    rng.shuffle(kept)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tool_usage: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as fp:
        for source_name, rec in kept:
            source_counts[source_name] += 1
            category_counts[_infer_category(rec)] += 1
            for msg in rec.get("messages") or []:
                for tc in msg.get("tool_calls") or []:
                    name = (tc.get("function") or {}).get("name")
                    if name:
                        tool_usage[name] += 1
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output_path),
        "total_kept": len(kept),
        "seed": seed,
        "per_source": {
            s.name: {
                "loaded": s.loaded,
                "invalid": s.invalid,
                "duplicates": s.duplicates,
                "kept": s.kept,
            }
            for s in source_stats
        },
        "final_source_counts": dict(source_counts),
        "final_category_mix": dict(category_counts),
        "final_tool_usage": dict(sorted(tool_usage.items(), key=lambda kv: -kv[1])),
    }

    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d records to %s", len(kept), output_path)
    logger.info("wrote stats to %s", stats_path)
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the final Phase A1 training JSONL.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments/phase_a1/data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase_a1/data/phase_a1_final.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    summary = assemble(args.data_dir, args.output, seed=args.seed)
    logger.info(
        "assemble complete  total_kept=%d  categories=%s  top_tools=%s",
        summary["total_kept"],
        summary["final_category_mix"],
        dict(list(summary["final_tool_usage"].items())[:5]),
    )
    return 0 if summary["total_kept"] > 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
