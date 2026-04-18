"""Stage A — LLM blueprint generation.

Given a ``GenerationSpec(primary_tool, category, seed)``, ask the primary-tier
LLM to produce a structured JSON ``Blueprint``:

    {
      "user_intent": "concise realistic user request",
      "planned_calls": [
        {"tool_name": "...", "arguments": {...}},
        ...
      ],
      "expected_outcome": "what the agent should have achieved after the calls"
    }

We do NOT ask the LLM to invent tool outputs here — those come from real
sandbox execution in executor.py. The blueprint is the skeleton only.

Category rules:
  - ``single_tool``    -> exactly 1 call using ``primary_tool``
  - ``multi_turn``     -> 2-4 calls, starting with ``primary_tool``
  - ``no_tool``        -> 0 calls (realistic conceptual Q)
  - ``error_recovery`` -> 2 calls: first one predicted to fail, second is a
                         corrected retry

The LLM is forced into ``response_format={"type":"json_object"}``. We validate
the resulting structure; on invalid JSON or schema mismatch we raise.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from experiments.phase_a1.executor import Blueprint, PlannedCall
from experiments.phase_a1.providers import LLMResponse, ProviderRouter
from experiments.phase_a1.registry_builder import ALL_TOOLS
from experiments.phase_a1.specs import GenerationSpec
from experiments.phase_a1.validate import validate_tool_call_args

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 2


_CATEGORY_RULES: dict[str, str] = {
    "single_tool": (
        "Exactly ONE planned_call using the primary_tool. "
        "The user's request should naturally require that single call."
    ),
    "multi_turn": (
        "Between 2 and 4 planned_calls. The FIRST call MUST use the primary_tool. "
        "Subsequent calls may use other tools. Each call's purpose should logically "
        "depend on the previous call's output (e.g., search → read → edit)."
    ),
    "no_tool": (
        "ZERO planned_calls (empty list). user_intent must be a question or "
        "request that the assistant can and should answer directly without calling "
        "any tool (conceptual questions, clarifications, meta-questions about "
        "capabilities, greetings, explanations of a concept)."
    ),
    "error_recovery": (
        "Exactly 2 planned_calls, both using the primary_tool. The FIRST call's "
        "arguments should plausibly FAIL in the sandbox (nonexistent file path, "
        "invalid regex, wrong line numbers, etc.). The SECOND call is a corrected "
        "retry with realistic fixed arguments."
    ),
}


_SYSTEM_TEMPLATE = """You are a data-generation assistant producing JSON blueprints
for a coding-agent training corpus. Godspeed is a CLI coding agent with 21 tools.

You will receive a spec with:
  - primary_tool: the tool the blueprint should primarily exercise
  - category: one of single_tool | multi_turn | no_tool | error_recovery
  - seed: an integer for variation (use it to diversify style)

Your job: emit ONE JSON object matching this exact schema:

{
  "user_intent": "<concise realistic user request, 1-2 sentences>",
  "planned_calls": [
    {"tool_name": "<one of the 21 tool names>", "arguments": {...}},
    ...
  ],
  "expected_outcome": "<what the agent should have achieved after the calls>"
}

HARD RULES:
1. Output ONLY valid JSON, no prose around it, no markdown fences.
2. tool_name MUST be exactly one of: __TOOLS_CSV__.
3. arguments MUST be a valid object matching that tool's JSON schema.
4. planned_calls[0].tool_name MUST equal primary_tool (EXCEPT category=no_tool
   which uses an empty planned_calls list).
5. user_intent must sound like a real developer request: realistic filenames,
   realistic modules, realistic bugs. Avoid placeholder wording.
6. Vary phrasing style across samples; use the seed to diversify tone and domain.
7. For the sandbox project (seeded): src/main.py has greet(name), src/utils.py
   has add/slugify, tests/ has pytest tests, there is a README.md and
   pyproject.toml, git is initialized. If the category suggests operating on
   real files, prefer these real paths; otherwise invent plausible-looking
   paths and the sandbox will miss them (fine - that drives realism).

Tool quick-reference (names and 1-line purposes):
  file_read        - read a file's contents (path + optional offset/limit)
  file_write       - write text to a file (path + content)
  file_edit        - surgical edit: replace an old_string with new_string
  diff_apply       - apply a unified diff patch to files
  glob_search      - find files matching a glob pattern (src/**/*.py)
  grep_search      - regex search file contents, returning matches
  code_search      - semantic code search (embedding-based)
  repo_map         - summarize repo structure / key symbols
  shell            - run a shell command (NOT dangerous commands)
  test_runner      - run tests (pytest / unittest) and return results
  verify           - lint / typecheck / compile a file
  background_check - inspect or poll background processes
  git              - git operations (status/diff/log/commit/add/checkout/...)
  github           - GitHub API operations (issues, PRs, files)
  web_search       - search the web
  web_fetch        - fetch a URL and return its content
  image_read       - read an image file and describe it
  pdf_read         - extract text from a PDF
  notebook_edit    - modify a .ipynb notebook cell
  tasks            - manage internal task list (add/list/complete)
  spawn_agent      - spawn a sub-agent for a subtask

REQUIRED ARGUMENTS (arguments MUST include these keys with non-empty values):
  file_read        - file_path (str)
  file_write       - file_path (str), content (str, may be "")
  file_edit        - file_path (str), old_string (str), new_string (str)
  diff_apply       - diff (str; must contain "@@" or "---")
  glob_search      - pattern (str)
  grep_search      - pattern (str)       # NEVER empty — regex to search for
  code_search      - query (str)
  shell            - command (str; not destructive)
  git              - action (one of: status, diff, add, commit, branch, checkout,
                     log, show, restore, stash, push, pull, fetch, merge, rebase,
                     reset, tag)
  github           - action (one of: list_prs, get_pr, create_pr, list_issues,
                     get_issue, create_issue, comment_issue, comment_pr)
  tasks            - action (one of: create, update, list, complete).
                     NOT "add" — use "create". Required fields per action:
                       create   -> title (str)
                       update   -> task_id (int), status (one of: pending,
                                   in_progress, completed)
                       complete -> task_id (int)
                       list     -> no extra args
  notebook_edit    - notebook_path (str), action (one of: edit_cell, add_cell,
                     delete_cell, move_cell)
  background_check - action (one of: status, output, kill), id (str for the
                     background process)
  web_search       - query (str)
  web_fetch        - url (str; must start with http:// or https://)
  image_read       - file_path (str)
  pdf_read         - file_path (str)
  spawn_agent      - task (str; non-empty description of the subtask)
  repo_map, test_runner, verify - no required args (may include optional
                     args like a file_path; invent realistic ones)

For tools with an "action" field (git, github, tasks, notebook_edit,
background_check): NEVER leave it null or empty; pick an action from the
enumerated list above that matches the user_intent. Do not invent actions
that don't appear in the list (e.g. "add" for tasks — it's "create").

CATEGORY-SPECIFIC RULES:
__CATEGORY_RULE__
"""


_USER_TEMPLATE = """Produce ONE blueprint JSON object now.

Spec:
  primary_tool: {primary_tool}
  category:     {category}
  seed:         {seed}
{few_shot_block}
Return only the JSON object described in the system prompt."""


def _format_anchor_for_few_shot(anchor: dict[str, Any]) -> dict[str, Any] | None:
    """Project a full anchor record down to the blueprint shape.

    Anchors are full ``{messages, tools}`` records; for blueprint calibration
    we only need the user_intent + the planned tool calls. Returns None if the
    anchor doesn't have the expected shape.
    """
    messages = anchor.get("messages") or []
    user_intent = ""
    planned_calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "user" and not user_intent:
            user_intent = str(msg.get("content") or "").strip()
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}
                if fn.get("name"):
                    planned_calls.append({"tool_name": fn["name"], "arguments": args})
    if not user_intent:
        return None
    return {
        "user_intent": user_intent,
        "planned_calls": planned_calls,
        "expected_outcome": "(elided)",
    }


def _render_few_shot_block(few_shots: list[dict[str, Any]] | None) -> str:
    """Render zero or more anchor examples as a 'gold reference' block.

    Anchor calibration shifts the blueprint generator toward the publication-
    quality distribution captured by anchor_opus_50.jsonl. Cost: ~150-300
    extra prompt tokens per example.
    """
    if not few_shots:
        return ""
    header = "Reference gold blueprints (study the user_intent phrasing and tool argument quality):"
    parts = ["", header]
    for shot in few_shots:
        projected = _format_anchor_for_few_shot(shot)
        if projected is None:
            continue
        parts.append(json.dumps(projected, ensure_ascii=False, indent=2))
    parts.append("")
    return "\n".join(parts)


def _render_prompts(
    spec: GenerationSpec,
    few_shots: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system = _SYSTEM_TEMPLATE.replace("__TOOLS_CSV__", ", ".join(ALL_TOOLS)).replace(
        "__CATEGORY_RULE__", _CATEGORY_RULES[spec.category]
    )
    user = _USER_TEMPLATE.format(
        primary_tool=spec.primary_tool,
        category=spec.category,
        seed=spec.seed,
        few_shot_block=_render_few_shot_block(few_shots),
    )
    return system, user


def _validate_blueprint(d: dict[str, Any], spec: GenerationSpec) -> list[str]:
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["top-level must be a JSON object"]
    if "user_intent" not in d or not isinstance(d["user_intent"], str) or not d["user_intent"]:
        errs.append("missing non-empty user_intent")
    if "planned_calls" not in d or not isinstance(d["planned_calls"], list):
        errs.append("missing planned_calls list")
    if "expected_outcome" not in d or not isinstance(d["expected_outcome"], str):
        errs.append("missing expected_outcome")

    if errs:
        return errs

    calls = d["planned_calls"]
    if spec.category == "no_tool":
        if calls:
            errs.append(f"no_tool must have 0 calls, got {len(calls)}")
    elif spec.category == "single_tool":
        if len(calls) != 1:
            errs.append(f"single_tool must have exactly 1 call, got {len(calls)}")
    elif spec.category == "multi_turn" and not 2 <= len(calls) <= 4:
        errs.append(f"multi_turn must have 2-4 calls, got {len(calls)}")
    elif spec.category == "error_recovery" and len(calls) != 2:
        errs.append(f"error_recovery must have exactly 2 calls, got {len(calls)}")

    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            errs.append(f"planned_calls[{i}] not an object")
            continue
        tn = call.get("tool_name")
        args = call.get("arguments")
        if tn not in ALL_TOOLS:
            errs.append(f"planned_calls[{i}].tool_name '{tn}' not in registry")
            # Can't run per-tool validator if the tool name is unknown.
            continue
        if not isinstance(args, dict):
            errs.append(
                f"planned_calls[{i}].arguments must be an object, got {type(args).__name__}"
            )
        else:
            # Per-tool arg schema — same invariants validate.py enforces post-
            # execution, applied here so malformed blueprints fail fast.
            for arg_err in validate_tool_call_args(tn, args):
                errs.append(f"planned_calls[{i}]: {arg_err}")
        if i == 0 and spec.category != "no_tool" and tn != spec.primary_tool:
            errs.append(
                f"planned_calls[0].tool_name must equal primary_tool="
                f"{spec.primary_tool!r}, got {tn!r}"
            )
    return errs


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Lenient JSON parse — strip common fencing LLMs add despite instructions."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove ```lang\n ... ```
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    return json.loads(stripped)


async def generate_blueprint(
    spec: GenerationSpec,
    router: ProviderRouter,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    few_shots: list[dict[str, Any]] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[Blueprint, LLMResponse]:
    """LLM-plan a single blueprint for one spec. Raises on invalid output.

    ``few_shots`` is an optional list of anchor records ({messages, tools})
    that will be projected to the blueprint shape and embedded in the user
    prompt as quality calibration. The orchestrator filters anchors to ones
    matching the spec's category before passing them in.

    On validation failure (invalid JSON or schema mismatch) we retry up to
    ``max_retries`` times with a bumped temperature to break determinism.
    This converts the majority of one-shot LLM slips (missing action field,
    empty pattern, wrong call count) into eventual successes without
    polluting the corpus with bad samples. Only the final attempt's failure
    is surfaced to the caller.
    """
    system, user = _render_prompts(spec, few_shots=few_shots)
    last_error: str = ""
    resp: LLMResponse | None = None

    for attempt in range(max_retries + 1):
        # Bump temperature slightly on each retry to avoid repeating the same
        # mistake; capped at 1.1 so the model stays coherent.
        effective_temp = min(1.1, temperature + 0.1 * attempt)
        resp = await router.complete(
            tier="primary",
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=effective_temp,
            json_mode=True,
        )

        try:
            data = _parse_llm_json(resp.text)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON: {e}. text={resp.text[:200]!r}"
            logger.info(
                "blueprint retry spec#%d attempt %d/%d: %s",
                spec.index,
                attempt + 1,
                max_retries + 1,
                last_error,
            )
            continue

        errors = _validate_blueprint(data, spec)
        if not errors:
            blueprint = Blueprint(
                user_intent=data["user_intent"],
                planned_calls=[
                    PlannedCall(tool_name=c["tool_name"], arguments=c.get("arguments", {}))
                    for c in data["planned_calls"]
                ],
                expected_outcome=data["expected_outcome"],
                category=spec.category,
                primary_tool=spec.primary_tool,
                spec_index=spec.index,
                spec_seed=spec.seed,
            )
            return blueprint, resp

        last_error = str(errors)
        logger.info(
            "blueprint retry spec#%d attempt %d/%d: %s",
            spec.index,
            attempt + 1,
            max_retries + 1,
            last_error,
        )

    msg = (
        f"blueprint validation failed for spec#{spec.index} "
        f"after {max_retries + 1} attempts: {last_error}"
    )
    raise ValueError(msg)


if __name__ == "__main__":
    import asyncio

    from experiments.phase_a1.providers import default_router
    from experiments.phase_a1.specs import GenerationSpec

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    async def _demo() -> None:
        router = default_router()
        spec = GenerationSpec(index=0, primary_tool="grep_search", category="single_tool", seed=42)
        bp, resp = await generate_blueprint(spec, router)
        logger.info(
            "provider=%s model=%s latency=%.2fs tokens in=%d out=%d",
            resp.provider,
            resp.model,
            resp.latency_s,
            resp.input_tokens,
            resp.output_tokens,
        )
        logger.info("blueprint=%s", json.dumps(bp.to_dict(), indent=2))

    asyncio.run(_demo())
