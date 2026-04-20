"""Top-level pipeline runner (minimal).

Composes Stages A through D for a configurable number of samples:

    specs.build_specs(total, floor_per_tool)
    -> for each spec:
         blueprints.generate_blueprint(spec) -> Blueprint
         executor.execute_blueprint(blueprint)  -> SessionArtifact
         narrator.narrate_session(blueprint, session_path)
         emit.emit_and_append(session_path, output_jsonl)

Judge + dedup + validate are NOT wired yet; those come after judge.py and
validate.py are implemented. This version is sufficient for end-to-end smoke.

Output:
    experiments/phase_a1/data/phase_a1_smoke.jsonl (or --output override)
    experiments/phase_a1/data/run_metrics.jsonl (one line per sample: cost,
        tokens, provider, latency, validation errors, status)

Resumable via ``--resume``: existing output-jsonl lines are counted and specs
from that index onward are processed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.phase_a1.blueprints import generate_blueprint
from experiments.phase_a1.emit import append_jsonl, emit_session
from experiments.phase_a1.executor import execute_blueprint
from experiments.phase_a1.judge import judge_sample, load_few_shots
from experiments.phase_a1.narrator import narrate_session
from experiments.phase_a1.providers import (
    AllProvidersExhausted,
    ProviderRouter,
    default_router,
)
from experiments.phase_a1.registry_builder import build_registry
from experiments.phase_a1.specs import build_specs
from experiments.phase_a1.validate import validate_record

logger = logging.getLogger(__name__)


@dataclass
class SampleMetrics:
    spec_index: int
    primary_tool: str
    category: str
    # status: "ok" | "blueprint_error" | "execute_error" | "narrate_error"
    #         | "emit_error" | "validate_dropped" | "judge_dropped" | "judge_error"
    status: str
    provider_blueprint: str = ""
    provider_narrator: str = ""
    provider_judge: str = ""
    blueprint_in_tokens: int = 0
    blueprint_out_tokens: int = 0
    narrator_in_tokens: int = 0
    narrator_out_tokens: int = 0
    judge_in_tokens: int = 0
    judge_out_tokens: int = 0
    judge_scores: dict = field(default_factory=dict)
    judge_min_score: int = 0
    validate_errors: list = field(default_factory=list)
    total_latency_s: float = 0.0
    error: str = ""


def _infer_category_for_anchor(anchor: dict) -> str:
    """Infer ``single_tool|multi_turn|no_tool`` from an anchor's tool-call count.

    Anchors aren't tagged with a category; we recover it from the messages
    so blueprint few-shots can be matched to the spec being generated.
    """
    n_calls = 0
    for msg in anchor.get("messages") or []:
        n_calls += len(msg.get("tool_calls") or [])
    if n_calls == 0:
        return "no_tool"
    if n_calls == 1:
        return "single_tool"
    return "multi_turn"


def _pick_blueprint_few_shots(
    anchors_by_category: dict[str, list[dict]],
    spec_category: str,
    spec_seed: int,
    *,
    n: int = 1,
) -> list[dict]:
    """Pick ``n`` anchors matching ``spec_category``, deterministic on seed.

    Falls back to any-category if the bucket is empty. Returns [] if no
    anchors are loaded at all.
    """
    bucket = anchors_by_category.get(spec_category) or []
    if not bucket:
        # Fall back to a flat pool of all categories.
        bucket = [a for lst in anchors_by_category.values() for a in lst]
    if not bucket:
        return []
    rng = random.Random(spec_seed)
    return rng.sample(bucket, k=min(n, len(bucket)))


async def _run_one(
    spec,
    registry,
    router: ProviderRouter,
    *,
    sessions_dir: Path,
    fixtures_dir: Path,
    output_jsonl: Path,
    judge_enabled: bool = False,
    few_shots: list[dict] | None = None,
    blueprint_anchors_by_category: dict[str, list[dict]] | None = None,
    blueprint_few_shot_n: int = 0,
) -> SampleMetrics:
    """Generate one sample with inline validate (always) and judge (optional).

    Quality gates, in order:
      1. Schema validate — drops samples with malformed shape, missing
         tool_call_ids, invalid args. Always on; cheap.
      2. Judge (optional) — drops samples scoring <4 on any of the 4 rubric
         dims. Adds one judge LLM call per sample (~1s, free-tier).

    Dropped samples never reach ``output_jsonl`` but are recorded in
    ``run_metrics.jsonl`` with their drop reason for post-run analysis.
    """
    started = time.perf_counter()
    m = SampleMetrics(
        spec_index=spec.index,
        primary_tool=spec.primary_tool,
        category=spec.category,
        status="ok",
    )

    bp_few_shots: list[dict] | None = None
    if blueprint_anchors_by_category and blueprint_few_shot_n > 0:
        bp_few_shots = _pick_blueprint_few_shots(
            blueprint_anchors_by_category,
            spec.category,
            spec.seed,
            n=blueprint_few_shot_n,
        )

    try:
        blueprint, bp_resp = await generate_blueprint(spec, router, few_shots=bp_few_shots)
        m.provider_blueprint = f"{bp_resp.provider}/{bp_resp.model}"
        m.blueprint_in_tokens = bp_resp.input_tokens
        m.blueprint_out_tokens = bp_resp.output_tokens
    except (ValueError, AllProvidersExhausted) as e:
        logger.warning("blueprint failed spec#%d: %s", spec.index, e, exc_info=True)
        m.status = "blueprint_error"
        m.error = str(e)[:300]
        m.total_latency_s = time.perf_counter() - started
        return m

    try:
        artifact = await execute_blueprint(
            blueprint,
            registry,
            output_dir=sessions_dir,
            fixtures_dir=fixtures_dir,
        )
    except Exception as e:
        logger.warning("execute failed spec#%d: %s", spec.index, e, exc_info=True)
        m.status = "execute_error"
        m.error = str(e)[:300]
        m.total_latency_s = time.perf_counter() - started
        return m

    try:
        narr_resp = await narrate_session(blueprint, artifact.session_path, router)
        m.provider_narrator = f"{narr_resp.provider}/{narr_resp.model}"
        m.narrator_in_tokens = narr_resp.input_tokens
        m.narrator_out_tokens = narr_resp.output_tokens
    except (ValueError, AllProvidersExhausted) as e:
        logger.warning("narrate failed spec#%d: %s", spec.index, e, exc_info=True)
        m.status = "narrate_error"
        m.error = str(e)[:300]
        m.total_latency_s = time.perf_counter() - started
        return m

    # Build the training record without writing yet.
    try:
        record = emit_session(artifact.session_path, registry)
    except RuntimeError as e:
        logger.warning("emit failed spec#%d: %s", spec.index, e, exc_info=True)
        m.status = "emit_error"
        m.error = str(e)[:300]
        m.total_latency_s = time.perf_counter() - started
        return m

    if record is None:
        m.status = "emit_error"
        m.error = "exporter returned None"
        m.total_latency_s = time.perf_counter() - started
        return m

    # Gate 1 — schema validation. Always on.
    errs, _, _ = validate_record(record)
    if errs:
        m.status = "validate_dropped"
        m.validate_errors = [str(e)[:200] for e in errs[:5]]
        m.total_latency_s = time.perf_counter() - started
        return m

    # Gate 2 — judge (optional, --judge flag).
    if judge_enabled:
        verdict = await judge_sample(record, router, few_shots=few_shots)
        if verdict.error:
            m.status = "judge_error"
            m.error = verdict.error[:300]
            m.total_latency_s = time.perf_counter() - started
            return m
        m.judge_scores = dict(verdict.scores)
        m.judge_min_score = verdict.min_score()
        # provider/tokens are not exposed by judge_sample; capture from raw_text.
        if verdict.raw_text:
            # Approximate: char count / 4 as token proxy when SDK doesn't report.
            m.judge_in_tokens = 0
            m.judge_out_tokens = len(verdict.raw_text) // 4
        if not verdict.passed:
            m.status = "judge_dropped"
            m.error = (verdict.reason or "below threshold")[:200]
            m.total_latency_s = time.perf_counter() - started
            return m

    append_jsonl(record, output_jsonl)
    m.total_latency_s = time.perf_counter() - started
    return m


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


async def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    sessions_dir = data_dir / "sessions"
    fixtures_dir = Path(args.fixtures_dir)
    output_jsonl = Path(args.output)
    metrics_jsonl = Path(args.metrics)

    data_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    if args.reset and not args.resume:
        output_jsonl.unlink(missing_ok=True)
        metrics_jsonl.unlink(missing_ok=True)
        if sessions_dir.exists():
            shutil.rmtree(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)

    router = default_router()
    registry = build_registry()
    logger.info(
        "registered %d tools; tools_schema_size=%d",
        len(registry.list_tools()),
        len(registry.get_schemas()),
    )

    specs = build_specs(
        total=args.total,
        floor_per_tool=max(1, args.total // len(registry.list_tools())),
        seed=args.seed,
        stream="synthetic",
    )

    start_index = _count_existing(output_jsonl) if args.resume else 0
    remaining = specs[start_index : start_index + args.limit]
    logger.info(
        "specs total=%d start=%d limit=%d → processing=%d",
        len(specs),
        start_index,
        args.limit,
        len(remaining),
    )

    sem = asyncio.Semaphore(args.concurrency)

    few_shots: list[dict] | None = None
    if args.judge:
        few_shots = load_few_shots(Path(args.anchor), limit=args.judge_few_shots)
        logger.info(
            "judge gating ENABLED - using %d anchor few-shots from %s",
            len(few_shots) if few_shots else 0,
            args.anchor,
        )

    blueprint_anchors_by_category: dict[str, list[dict]] = {}
    if args.blueprint_few_shots > 0:
        all_anchors = load_few_shots(Path(args.anchor), limit=10_000)
        for a in all_anchors:
            cat = _infer_category_for_anchor(a)
            blueprint_anchors_by_category.setdefault(cat, []).append(a)
        total_loaded = sum(len(v) for v in blueprint_anchors_by_category.values())
        logger.info(
            "blueprint few-shots ENABLED - %d anchors loaded across %d categories",
            total_loaded,
            len(blueprint_anchors_by_category),
        )

    async def task(spec):
        async with sem:
            return await _run_one(
                spec,
                registry,
                router,
                sessions_dir=sessions_dir,
                fixtures_dir=fixtures_dir,
                output_jsonl=output_jsonl,
                judge_enabled=args.judge,
                few_shots=few_shots,
                blueprint_anchors_by_category=blueprint_anchors_by_category,
                blueprint_few_shot_n=args.blueprint_few_shots,
            )

    ok = 0
    failed = 0
    t0 = time.perf_counter()
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_jsonl, "a", encoding="utf-8") as mf:
        coros = [task(s) for s in remaining]
        for fut in asyncio.as_completed(coros):
            m = await fut
            mf.write(json.dumps(asdict(m)) + "\n")
            mf.flush()
            if m.status == "ok":
                ok += 1
                judge_tag = f" judge={m.judge_min_score}/5" if m.judge_min_score else ""
                logger.info(
                    "ok  #%04d tool=%-18s cat=%-14s  bp=%s narr=%s%s  %.2fs",
                    m.spec_index,
                    m.primary_tool,
                    m.category,
                    m.provider_blueprint,
                    m.provider_narrator,
                    judge_tag,
                    m.total_latency_s,
                )
            else:
                failed += 1
                logger.warning(
                    "FAIL #%04d tool=%-18s cat=%-14s  status=%s  %s",
                    m.spec_index,
                    m.primary_tool,
                    m.category,
                    m.status,
                    m.error[:160],
                )

    wall = time.perf_counter() - t0
    logger.info(
        "run complete  ok=%d failed=%d  wall=%.1fs  usage=%s",
        ok,
        failed,
        wall,
        json.dumps(router.usage_snapshot()),
    )
    logger.info("output: %s (size=%d lines)", output_jsonl, _count_existing(output_jsonl))
    return 0 if ok > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=6200, help="Spec universe size")
    parser.add_argument("--limit", type=int, default=3, help="How many specs to process this run")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="experiments/phase_a1/data")
    parser.add_argument("--fixtures-dir", type=str, default="experiments/phase_a1/fixtures")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/phase_a1/data/phase_a1_smoke.jsonl",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="experiments/phase_a1/data/run_metrics.jsonl",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete output + metrics + sessions before start (ignored with --resume)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable inline judge gating: drops samples scoring <4 on any rubric dim. "
        "Adds one judge LLM call per sample (~1s, free-tier).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="experiments/phase_a1/data/anchor_opus_50.jsonl",
        help="Path to the Opus-anchor JSONL used for judge few-shots.",
    )
    parser.add_argument(
        "--judge-few-shots",
        type=int,
        default=3,
        help="How many anchor samples to feed the judge as calibration examples.",
    )
    parser.add_argument(
        "--blueprint-few-shots",
        type=int,
        default=0,
        help=(
            "How many anchor examples to inject into each blueprint prompt as "
            "quality calibration. 0 disables; 1 is recommended for prod (small "
            "token cost, meaningful quality lift)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Quiet the very-chatty http logs from provider SDKs
    for name in ("httpx", "httpcore", "openai._base_client"):
        with contextlib.suppress(Exception):
            logging.getLogger(name).setLevel(logging.WARNING)

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
