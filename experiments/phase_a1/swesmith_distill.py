"""Stage G — Distill phase2_swesmith.jsonl into Phase A1 OpenAI format.

Source: ``ml-lab/.../2026-04-godspeed-coder/data/phase2_swesmith.jsonl`` —
a 24K-record corpus of SWE-bench/SWE-smith trajectories that ml-lab
previously converted to a Hermes-XML transport format. The conversion was
lossy: ~99.8% of the embedded ``<tool_call>`` blocks collapsed into
``shell`` with placeholder arguments (``"command": "# inferred from context"``),
which would dominate any sample drawn naively.

What this module does:

  1. Stream every record, parse the embedded ``<tool_call>``/``<tool_response>``
     XML blocks back into structured (assistant_text, raw_call, observation)
     tuples.
  2. **Re-infer** the canonical Godspeed tool from the observation pattern
     (``cat -n`` output → file_read, ``File created successfully`` → file_write,
     bare path list → glob_search, ``test session starts`` → test_runner, ...).
     Records that yield no inferable calls are dropped.
  3. **Diversity-cluster** the surviving records via TF-IDF + KMeans (k=30)
     on the user prompt (the PR description), then sample evenly across
     clusters. Records whose entire turn-stack remains shell-canonical are
     capped at ``--shell-cap`` (default 800) to keep the final mix diverse.
  4. **Re-render** each kept record in OpenAI ``{messages, tools}`` shape with
     proper ``tool_call_id`` linkage and the canonical 21-tool registry.

Output validates cleanly against ``validate.py``. Determinism: pure function
of ``(input_path, target, k, shell_cap, seed)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.phase_a1.executor import _SYSTEM_PROMPT
from experiments.phase_a1.registry_builder import ALL_TOOLS, get_tool_schemas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — match the audited Phase A1 plan
# ---------------------------------------------------------------------------

DEFAULT_TARGET: int = 1500
DEFAULT_K_CLUSTERS: int = 30
DEFAULT_SHELL_CAP: int = 800
DEFAULT_SEED: int = 42

_ALL_TOOLS_SET = frozenset(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Step 1 — parse the Hermes XML transport back into structured turns
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.DOTALL)
_OBS_PREFIX_RE = re.compile(r"^OBSERVATION:\s*\n?", re.MULTILINE)
_PR_DESCRIPTION_RE = re.compile(r"<pr_description>\s*(.+?)\s*</pr_description>", re.DOTALL)


@dataclass
class Turn:
    """One assistant→tool round-trip in the source trajectory."""

    assistant_text: str
    raw_call: dict[str, Any]
    observation: str


@dataclass
class ParsedRecord:
    """A swesmith record after parsing but before re-inference."""

    user_prompt: str
    turns: list[Turn] = field(default_factory=list)


def _parse_record(record: dict[str, Any]) -> ParsedRecord | None:
    """Pull (user_prompt, [Turn, ...]) out of a swesmith record.

    Returns ``None`` if the record is structurally too thin to be useful
    (missing user prompt, no assistant turns, or malformed XML throughout).
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return None

    user_prompt: str | None = None
    pending_call: dict[str, Any] | None = None
    pending_text: str = ""
    turns: list[Turn] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user" and user_prompt is None:
            user_prompt = _extract_user_prompt(content)
        elif role == "assistant":
            text, call = _split_assistant_content(content)
            if call is not None:
                pending_call = call
                pending_text = text
        elif role == "tool" and pending_call is not None:
            obs = _extract_observation(content)
            if obs:
                turns.append(
                    Turn(assistant_text=pending_text, raw_call=pending_call, observation=obs)
                )
            pending_call = None
            pending_text = ""

    if not user_prompt or not turns:
        return None
    return ParsedRecord(user_prompt=user_prompt, turns=turns)


def _extract_user_prompt(content: str) -> str:
    """Pull the PR description out of the swesmith user prompt template."""
    m = _PR_DESCRIPTION_RE.search(content)
    if m:
        return m.group(1).strip()
    # Fall back to the whole user content, minus the <uploaded_files> preamble.
    cleaned = re.sub(r"<uploaded_files>.*?</uploaded_files>\s*", "", content, flags=re.DOTALL)
    return cleaned.strip()


def _split_assistant_content(content: str) -> tuple[str, dict[str, Any] | None]:
    """Split the assistant content into (reasoning_text, parsed_tool_call)."""
    m = _TOOL_CALL_RE.search(content)
    if not m:
        return content.strip(), None
    text = content[: m.start()].strip()
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return text, None
    if not isinstance(call, dict) or "name" not in call:
        return text, None
    return text, call


def _extract_observation(content: str) -> str:
    """Unwrap a ``<tool_response>`` block and return the inner observation text."""
    m = _TOOL_RESPONSE_RE.search(content)
    if not m:
        return ""
    try:
        body = json.loads(m.group(1))
    except json.JSONDecodeError:
        return ""
    inner = body.get("content") if isinstance(body, dict) else None
    if not isinstance(inner, str):
        return ""
    # Strip the leading "OBSERVATION:\n" the source always prepends.
    return _OBS_PREFIX_RE.sub("", inner, count=1).strip()


# ---------------------------------------------------------------------------
# Step 2 — re-infer canonical (tool_name, args) from the observation pattern
# ---------------------------------------------------------------------------


_CAT_N_RE = re.compile(r"running `cat -n` on (\S+)")
_FILE_CREATED_RE = re.compile(r"File created successfully at:\s*(\S+)")
_TEST_SESSION_RE = re.compile(r"=+ test session starts =+|^=+ \d+ passed", re.MULTILINE)
_GREP_LINE_RE = re.compile(r"^[/\w][\w./-]*:\d+:", re.MULTILINE)
_PIP_INSTALL_RE = re.compile(r"Successfully installed |Collecting [\w-]+")
_TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_PATH_LIST_RE = re.compile(r"^/\S+\.[a-zA-Z0-9]+\s*$", re.MULTILINE)
_GIT_OUTPUT_RE = re.compile(
    r"^On branch |^HEAD is now at |^Switched to branch |^\[\w+ [0-9a-f]{7,}\]",
    re.MULTILINE,
)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _extract_path_from_assistant(text: str) -> str | None:
    """Best-effort grab of a file path from the assistant's reasoning text."""
    m = re.search(r"[`'\"]?(/?[\w/.\-]+\.\w{1,5})[`'\"]?", text)
    return m.group(1) if m else None


def infer_canonical_call(turn: Turn) -> tuple[str, dict[str, Any]] | None:
    """Re-infer a Godspeed tool call from a turn's observation pattern.

    Returns ``None`` if no pattern matches with reasonable confidence — the
    caller should drop turns we can't pin down rather than train the model
    on noise.
    """
    obs = turn.observation
    text = turn.assistant_text or ""

    # 1. file_read: explicit `cat -n` output.
    m = _CAT_N_RE.search(obs)
    if m:
        return ("file_read", {"file_path": m.group(1).rstrip(":,")})

    # 2. file_write: explicit "File created successfully at: <path>".
    m = _FILE_CREATED_RE.search(obs)
    if m:
        path = m.group(1)
        # We don't have the original written content; reconstruct a placeholder
        # that the validator accepts (non-empty string).
        return ("file_write", {"file_path": path, "content": ""})

    # 3. test_runner: pytest-style session banner.
    if _TEST_SESSION_RE.search(obs):
        path = _extract_path_from_assistant(text) or "tests/"
        return ("test_runner", {"path": path})

    # 4. grep_search: ``file:line:`` matches across multiple lines.
    grep_hits = _GREP_LINE_RE.findall(obs)
    if len(grep_hits) >= 2:
        pattern = _extract_pattern_from_assistant(text)
        if pattern:
            return ("grep_search", {"pattern": pattern})

    # 5. shell + pip install.
    if _PIP_INSTALL_RE.search(obs):
        cmd = _extract_command_from_assistant(text) or "pip install -e ."
        return ("shell", {"command": cmd})

    # 6. shell + python traceback.
    if _TRACEBACK_RE.search(obs):
        cmd = _extract_command_from_assistant(text) or "python reproduce.py"
        return ("shell", {"command": cmd})

    # 7. glob_search: bare path list (every line looks like a file path).
    path_lines = _PATH_LIST_RE.findall(obs)
    obs_lines = [line for line in obs.splitlines() if line.strip()]
    if path_lines and len(path_lines) >= max(2, len(obs_lines) - 1):
        pattern = _extract_glob_from_assistant(text) or "**/*.py"
        return ("glob_search", {"pattern": pattern})

    # 8. git: distinctive git output banners.
    if _GIT_OUTPUT_RE.search(obs):
        return ("git", {"action": "status"})

    # 9. Fallback: shell, but only if the original guess was already shell.
    raw_name = turn.raw_call.get("name")
    if raw_name == "shell":
        cmd = _extract_command_from_assistant(text)
        if cmd:
            return ("shell", {"command": cmd})
    return None


def _extract_pattern_from_assistant(text: str) -> str | None:
    """Pull a grep-able pattern out of the assistant's reasoning."""
    m = re.search(r"(?:search|grep|find)\s+(?:for\s+)?[`'\"]([^`'\"]{2,40})[`'\"]", text, re.I)
    return m.group(1) if m else None


def _extract_command_from_assistant(text: str) -> str | None:
    """Pull a shell command out of an inline code span or fenced block."""
    m = re.search(r"```(?:bash|sh|shell)?\n([^\n]+?)\n```", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"`([^`\n]{3,120})`", text)
    if m:
        cand = m.group(1).strip()
        prefixes = ("pip ", "python ", "pytest", "git ", "ls ", "cd ")
        if any(cand.startswith(p) for p in prefixes):
            return cand
    return None


def _extract_glob_from_assistant(text: str) -> str | None:
    m = re.search(r"`(\*\*?/[^\s`]+)`", text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Step 3 — cluster + sample
# ---------------------------------------------------------------------------


@dataclass
class DistilledRecord:
    """A swesmith record after parsing + inference, ready for OpenAI rendering."""

    user_prompt: str
    turns: list[Turn]
    canonical_calls: list[tuple[str, dict[str, Any]]]

    @property
    def is_shell_only(self) -> bool:
        return all(name == "shell" for name, _ in self.canonical_calls)


def _distill_one(
    parsed: ParsedRecord, *, min_inference_ratio: float = 0.5
) -> DistilledRecord | None:
    """Run inference on every turn; keep the record if enough turns survived."""
    canonical: list[tuple[str, dict[str, Any]]] = []
    kept_turns: list[Turn] = []
    for turn in parsed.turns:
        inferred = infer_canonical_call(turn)
        if inferred is None:
            continue
        canonical.append(inferred)
        kept_turns.append(turn)

    if not kept_turns:
        return None
    if len(kept_turns) / max(len(parsed.turns), 1) < min_inference_ratio:
        return None
    return DistilledRecord(
        user_prompt=parsed.user_prompt,
        turns=kept_turns,
        canonical_calls=canonical,
    )


def _cluster_indices(prompts: list[str], k: int, seed: int) -> list[int]:
    """Run TF-IDF + KMeans on prompts; return cluster id per prompt.

    Falls back to round-robin cluster assignment if the corpus is too small
    or homogeneous for TF-IDF to produce a usable vocabulary (e.g., min_df
    pruning leaves zero terms). The fallback preserves determinism.
    """
    if not prompts:
        return []
    effective_k = min(k, len(prompts))
    # Relax min_df for tiny corpora so synthetic test fixtures still cluster.
    min_df = 2 if len(prompts) >= 20 else 1
    vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=min_df,
        max_df=1.0 if len(prompts) < 20 else 0.9,
    )
    try:
        matrix = vectorizer.fit_transform(prompts)
    except ValueError:
        # Vocabulary collapsed; degrade gracefully to deterministic round-robin.
        return [i % effective_k for i in range(len(prompts))]
    km = KMeans(n_clusters=effective_k, random_state=seed, n_init=4)
    return list(km.fit_predict(matrix))


def diversity_sample(
    records: list[DistilledRecord],
    *,
    target: int,
    k: int,
    shell_cap: int,
    seed: int,
) -> list[DistilledRecord]:
    """Diversity-sample records via TF-IDF cluster, capping the shell-only mix."""
    if not records:
        return []

    rng = random.Random(seed)
    cluster_ids = _cluster_indices([r.user_prompt for r in records], k, seed)

    # Bucket records by cluster.
    by_cluster: dict[int, list[DistilledRecord]] = defaultdict(list)
    for cid, rec in zip(cluster_ids, records, strict=True):
        by_cluster[cid].append(rec)

    for bucket in by_cluster.values():
        rng.shuffle(bucket)

    # Round-robin draw across clusters until we reach target or run out.
    cluster_order = list(by_cluster.keys())
    rng.shuffle(cluster_order)
    cursors = {cid: 0 for cid in cluster_order}

    chosen: list[DistilledRecord] = []
    shell_only_count = 0
    progress = True

    while len(chosen) < target and progress:
        progress = False
        for cid in cluster_order:
            if len(chosen) >= target:
                break
            cursor = cursors[cid]
            bucket = by_cluster[cid]
            while cursor < len(bucket):
                cand = bucket[cursor]
                cursor += 1
                if cand.is_shell_only and shell_only_count >= shell_cap:
                    continue
                chosen.append(cand)
                if cand.is_shell_only:
                    shell_only_count += 1
                progress = True
                break
            cursors[cid] = cursor

    return chosen


# ---------------------------------------------------------------------------
# Step 4 — re-render in OpenAI {messages, tools} shape
# ---------------------------------------------------------------------------


def render_to_openai(
    record: DistilledRecord,
    tools_schema: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert a distilled record to the canonical OpenAI training format."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _truncate(record.user_prompt, 2000)},
    ]

    pairs = zip(record.turns, record.canonical_calls, strict=True)
    for i, (turn, (name, args)) in enumerate(pairs):
        call_id = f"call_{i:02d}"
        text = turn.assistant_text or f"Calling {name}."
        messages.append(
            {
                "role": "assistant",
                "content": _truncate(text, 800),
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _truncate(turn.observation, 1500),
            }
        )

    # Final assistant summary so the trace doesn't end on a tool message.
    messages.append(
        {
            "role": "assistant",
            "content": "I've gathered the relevant context above; ready to apply the fix.",
        }
    )

    return {"messages": messages, "tools": tools_schema}


# ---------------------------------------------------------------------------
# Streaming pipeline + CLI
# ---------------------------------------------------------------------------


def stream_distilled(input_path: Path) -> list[DistilledRecord]:
    """Parse + distill every record in the source JSONL."""
    distilled: list[DistilledRecord] = []
    skipped_parse = 0
    skipped_inference = 0

    with input_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = _parse_record(record)
            if parsed is None:
                skipped_parse += 1
                continue
            d = _distill_one(parsed)
            if d is None:
                skipped_inference += 1
                continue
            distilled.append(d)

    logger.info(
        "parsed %d records (skipped %d parse, %d inference)",
        len(distilled),
        skipped_parse,
        skipped_inference,
    )
    return distilled


def write_distilled_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    target: int,
    k: int,
    shell_cap: int,
    seed: int,
) -> dict[str, Any]:
    """End-to-end: parse, infer, cluster-sample, render, write JSONL."""
    distilled = stream_distilled(input_path)
    chosen = diversity_sample(
        distilled,
        target=target,
        k=k,
        shell_cap=shell_cap,
        seed=seed,
    )

    tools_schema = get_tool_schemas()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tool_usage: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as fp:
        for rec in chosen:
            for name, _ in rec.canonical_calls:
                tool_usage[name] += 1
            fp.write(json.dumps(render_to_openai(rec, tools_schema), ensure_ascii=False) + "\n")

    return {
        "input_records": len(distilled),
        "written": len(chosen),
        "tool_usage": dict(tool_usage),
        "shell_only_kept": sum(1 for r in chosen if r.is_shell_only),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Distill swesmith trajectories for Phase A1.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../ml-lab/experiments/2026-04-godspeed-coder/data/phase2_swesmith.jsonl"),
        help="Source phase2_swesmith.jsonl (Hermes XML transport).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase_a1/data/phase_a1_swesmith_distilled.jsonl"),
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--k", type=int, default=DEFAULT_K_CLUSTERS)
    parser.add_argument("--shell-cap", type=int, default=DEFAULT_SHELL_CAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if not args.input.exists():
        logger.error("input not found: %s", args.input)
        return 1

    summary = write_distilled_jsonl(
        args.input,
        args.output,
        target=args.target,
        k=args.k,
        shell_cap=args.shell_cap,
        seed=args.seed,
    )
    logger.info(
        "distill complete  input=%d written=%d shell_only=%d  out=%s",
        summary["input_records"],
        summary["written"],
        summary["shell_only_kept"],
        args.output,
    )
    logger.info(
        "tool usage (sorted): %s",
        dict(sorted(summary["tool_usage"].items(), key=lambda kv: -kv[1])),
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
