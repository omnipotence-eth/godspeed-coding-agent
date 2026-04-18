"""Stage H — Param-shuffle + intent paraphrase augmentation for Phase A1.

Generates 200 ``{messages, tools}`` samples concentrated on the tools that
the synthetic + distill streams under-cover (web_*, image/pdf_read, tasks,
background_check, notebook_edit, spawn_agent, repo_map, diff_apply). Per
tool we keep:

  * 4-6 ``intent templates`` — natural-language user prompts with ``{slot}``
    placeholders.
  * Per-slot vocabularies — small lists of realistic file paths, URLs, search
    queries, task ids, etc.
  * 2-3 ``pre-call`` and ``post-call`` paraphrases for the assistant prose.
  * 2-3 ``output templates`` — realistic-looking tool responses, also slotted.

Sample N for each tool is the cartesian product of those pools, drawn
deterministically with a per-tool seed (so the corpus is reproducible bit
for bit and you can re-roll a single tool without disturbing the others).

Output validates cleanly against ``validate.py`` and is consumed by the
training reader the same way as the anchor + distill streams.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

from experiments.phase_a1.executor import _SYSTEM_PROMPT
from experiments.phase_a1.registry_builder import get_tool_schemas

logger = logging.getLogger(__name__)


DEFAULT_TOTAL: int = 200
DEFAULT_SEED: int = 42

# 10 tools x 20 samples = 200. Pick the tools that distill leaves cold and that
# are also rare in the synthetic stream's natural prompt distribution.
TARGET_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_fetch",
    "image_read",
    "pdf_read",
    "spawn_agent",
    "background_check",
    "notebook_edit",
    "tasks",
    "repo_map",
    "diff_apply",
)


# ---------------------------------------------------------------------------
# Template definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolTemplate:
    """All slots + prose pools needed to synthesize samples for one tool."""

    tool: str
    user_intents: tuple[str, ...]
    pre_call: tuple[str, ...]
    post_call: tuple[str, ...]
    arg_specs: tuple[dict[str, str], ...]
    output_templates: tuple[str, ...]
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _slot_keys(template: str) -> set[str]:
    """All ``{name}`` placeholders in a template string."""
    return {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}


def _render(template: str, slots: dict[str, str]) -> str:
    """Format ``template`` against ``slots``; missing keys raise eagerly."""
    return template.format(**slots)


# ---------------------------------------------------------------------------
# Tool templates — one ToolTemplate per under-represented tool
# ---------------------------------------------------------------------------


_PROJECT_PATHS: tuple[str, ...] = (
    "/home/dev/projects/api",
    "/home/dev/projects/web",
    "/srv/services/billing",
    "/opt/apps/notebooks",
    "/var/www/dashboard",
)

_PAPER_TITLES: tuple[str, ...] = (
    "FlashAttention-3",
    "Mamba SSM",
    "Direct Preference Optimization",
    "RingAttention",
    "Mixture of Depths",
)


def _t_web_search() -> ToolTemplate:
    return ToolTemplate(
        tool="web_search",
        user_intents=(
            "Find recent docs on {topic}.",
            "Search the web for {topic}.",
            "What does the official documentation say about {topic}?",
            "Look up {topic} - I need authoritative sources.",
        ),
        pre_call=(
            "Searching the web for {topic}.",
            "Querying for authoritative sources on {topic}.",
            "Pulling up docs on {topic}.",
        ),
        post_call=(
            "Top hits look authoritative - the official site covers {topic} directly. "
            "Want me to fetch any of these for the full text?",
            "Plenty of relevant material on {topic}; the first result is the official "
            "reference and answers most basic questions inline.",
            "Solid coverage on {topic} - first result is official, the others give "
            "practical examples. Pick one to fetch in full?",
        ),
        arg_specs=({"query": "{topic}"},),
        output_templates=(
            "1. {topic} - official documentation: {site}/docs/{slug}\n"
            "   Detailed reference and conceptual overview.\n"
            "2. Practical guide: {site}/guides/{slug}-tutorial\n"
            "   Hands-on walkthrough with examples.\n"
            "3. Common pitfalls: {site}/blog/{slug}-gotchas\n"
            "   Edge cases and migration tips.\n",
        ),
        slots={
            "topic": (
                "the FastAPI dependency injection system",
                "PostgreSQL row-level security",
                "Pydantic v2 validators",
                "OpenTelemetry trace propagation in Python",
                "AWS S3 multipart upload limits",
                "Stripe webhook signature verification",
                "Kubernetes liveness vs readiness probes",
                "asyncio cancellation semantics",
            ),
            "site": (
                "fastapi.tiangolo.com",
                "postgresql.org",
                "docs.pydantic.dev",
                "opentelemetry.io",
                "docs.aws.amazon.com",
                "stripe.com",
                "kubernetes.io",
                "docs.python.org",
            ),
            "slug": (
                "dependency-injection",
                "row-level-security",
                "validators",
                "trace-propagation",
                "multipart-upload",
                "webhook-signatures",
                "probes",
                "cancellation",
            ),
        },
    )


def _t_web_fetch() -> ToolTemplate:
    return ToolTemplate(
        tool="web_fetch",
        user_intents=(
            "Fetch {url} and summarize the relevant section.",
            "Pull down {url} - I want to read the actual page.",
            "Grab the contents of {url} for me.",
            "Retrieve {url} so we can quote from it.",
        ),
        pre_call=(
            "Fetching {url}.",
            "Pulling the page from {url}.",
            "Downloading {url}.",
        ),
        post_call=(
            "Got the page. The substantive content is the section on {topic} - "
            "covers definitions, parameters, and a worked example.",
            "Page loaded. The middle section answers your question on {topic} "
            "directly; the rest is preamble and changelog.",
            "Fetched cleanly. Key takeaway: {topic} is documented with both a "
            "conceptual overview and a quickstart snippet.",
        ),
        arg_specs=({"url": "{url}"},),
        output_templates=(
            "Page: {url}\n"
            "Title: {title}\n\n"
            "## {topic}\n"
            "The {topic} feature accepts the standard configuration object with "
            "two required keys (`name`, `enabled`) and an optional `tags` list. "
            "When enabled, requests are processed inline; when disabled, they "
            "are silently dropped. The behavior is deterministic across retries.\n\n"
            "## Configuration\n"
            "See the reference section for the full parameter table.\n",
        ),
        slots={
            "url": (
                "https://docs.python.org/3/library/asyncio-task.html",
                "https://www.rfc-editor.org/rfc/rfc7231",
                "https://docs.pydantic.dev/latest/concepts/validators/",
                "https://kubernetes.io/docs/concepts/workloads/pods/",
                "https://stripe.com/docs/webhooks/signatures",
                "https://opentelemetry.io/docs/concepts/signals/traces/",
            ),
            "title": (
                "Python asyncio - Tasks reference",
                "RFC 7231 - HTTP/1.1 Semantics",
                "Pydantic v2 Validators",
                "Kubernetes Pods Overview",
                "Stripe Webhook Signature Verification",
                "OpenTelemetry Traces",
            ),
            "topic": (
                "asyncio task cancellation",
                "HTTP method semantics",
                "field validators",
                "pod lifecycle",
                "signature verification",
                "trace propagation",
            ),
        },
    )


def _t_image_read() -> ToolTemplate:
    return ToolTemplate(
        tool="image_read",
        user_intents=(
            "Take a look at {path} and describe what you see.",
            "Inspect the screenshot at {path} - what is it showing?",
            "Open {path} and tell me what's wrong in the UI.",
            "Read the image at {path}.",
        ),
        pre_call=(
            "Inspecting the image at {path}.",
            "Loading {path} for visual inspection.",
            "Viewing {path}.",
        ),
        post_call=(
            "The image shows {scene}. The most actionable thing here is the "
            "{cta_label} - that's the user's intended next step.",
            "Looks like {scene}. If this is a regression report, the "
            "{cta_label} element is the relevant focus point.",
            "Captured. The visible content is {scene}; pay attention to the "
            "{cta_label} since that's where the interaction happens.",
        ),
        arg_specs=({"file_path": "{path}"},),
        output_templates=(
            "[image: {dims} PNG]\n"
            "Description (vision model output):\n"
            "{scene}. The {cta_label} is positioned in the bottom-right with the "
            "expected primary-button styling. No console errors visible. Layout "
            "appears stable across the captured viewport.\n",
        ),
        slots={
            "path": (
                "screenshots/login_v2.png",
                "screenshots/dashboard_error.png",
                "screenshots/checkout_step3.png",
                "mockups/settings_panel.png",
                "qa/regression_2026_04.png",
            ),
            "dims": ("1024x768", "1440x900", "1920x1080", "390x844"),
            "scene": (
                "a login screen with email and password fields, a primary CTA, "
                "and a small 'forgot password' link below the form",
                "a dashboard widget panel with three KPI tiles and a line chart "
                "showing daily active users over the last 30 days",
                "a checkout summary listing two line items, applied promo code, "
                "and a total of $42.18",
                "a settings sidebar with collapsible sections for Profile, "
                "Notifications, Security, and Billing",
            ),
            "cta_label": (
                "Sign in button",
                "Refresh button",
                "Pay now button",
                "Save changes button",
            ),
        },
    )


def _t_pdf_read() -> ToolTemplate:
    return ToolTemplate(
        tool="pdf_read",
        user_intents=(
            "Read {path} and tell me the main contribution.",
            "Pull the abstract from {path}.",
            "Skim {path} - what's the headline claim?",
            "Read pages 1-3 of {path} and summarize the method.",
        ),
        pre_call=(
            "Reading the PDF at {path}.",
            "Loading {path} for the first few pages.",
            "Opening {path}.",
        ),
        post_call=(
            "The headline claim is that {paper} {claim}. The method centers on "
            "{technique}, and the reported gains hold across the standard "
            "benchmarks they cite.",
            "{paper} argues {claim}. Mechanically it's about {technique}, with "
            "the speedup attributed to better memory access patterns.",
            "Core idea: {claim}. The author frames it as a small change to "
            "{technique} that compounds nicely at scale.",
        ),
        arg_specs=({"file_path": "{path}", "pages": "{pages}"},),
        output_templates=(
            "[Page 1] {paper}.\n"
            "Abstract. We present a method that {claim}. The core idea is to "
            "{technique}, which reduces compute by a meaningful constant factor "
            "without changing the model architecture or training objective. We "
            "evaluate on standard benchmarks and report consistent gains.\n\n"
            "[Page 2] Method.\n"
            "Our approach modifies the standard pipeline by {technique}. The "
            "modification is drop-in and requires no retraining of existing "
            "checkpoints.\n",
        ),
        slots={
            "path": (
                "papers/flashattn3_2024.pdf",
                "papers/mamba_2024.pdf",
                "papers/dpo_2023.pdf",
                "papers/ringattn_2024.pdf",
                "papers/mod_2024.pdf",
            ),
            "pages": ("1-2", "1-3", "1-4"),
            "paper": _PAPER_TITLES,
            "claim": (
                "halves the wall-clock time of attention without quality loss",
                "matches transformer quality with sub-quadratic compute",
                "removes the need for a separate reward model in RLHF",
                "scales context length linearly with the number of devices",
                "skips redundant compute in unimportant tokens",
            ),
            "technique": (
                "fusing the softmax with the matmul kernel",
                "replacing attention with a selective state-space scan",
                "directly optimizing a preference-derived loss",
                "sharding the KV cache and rotating it across the ring",
                "routing each token to a learned subset of layers",
            ),
        },
    )


def _t_spawn_agent() -> ToolTemplate:
    return ToolTemplate(
        tool="spawn_agent",
        user_intents=(
            "Have a {agent} subagent investigate {topic} and report back.",
            "Spawn a {agent} agent to figure out {topic}.",
            "Delegate the {topic} question to a {agent} subagent.",
            "Get a {agent} to look into {topic} - I want a recommendation.",
        ),
        pre_call=(
            "Delegating that to a {agent} subagent.",
            "Spawning a {agent} to investigate {topic}.",
            "Handing this off to a {agent} subagent.",
        ),
        post_call=(
            "Subagent recommends {recommendation}. The reasoning is "
            "well-grounded; ready to act on it when you give the word.",
            "{agent} came back with {recommendation}. Sources are credible and "
            "the recommendation aligns with how this codebase already handles "
            "similar concerns.",
            "Got it - {recommendation}. Worth running by the team if this "
            "touches anything load-bearing, otherwise we can proceed.",
        ),
        arg_specs=({"agent": "{agent}", "task": "{task}"},),
        output_templates=(
            "[{agent} subagent finished]\n"
            "Recommendation: {recommendation}.\n"
            "Reasoning:\n"
            " - The recommended option has the best fit for our existing\n"
            "   architecture and operational constraints.\n"
            " - Alternatives were considered but reject for ergonomic or\n"
            "   long-term-maintenance reasons.\n"
            "Sources: official docs, two recent blog posts, and our own\n"
            "internal precedent in adjacent modules.\n",
        ),
        slots={
            "agent": ("researcher", "code-reviewer", "explorer", "planner"),
            "topic": (
                "whether to use Pydantic Settings or environs for config",
                "the right caching layer for our read-heavy endpoints",
                "how to roll out feature flags safely",
                "the migration path from Celery to Dramatiq",
                "OpenAPI codegen options for a TypeScript client",
            ),
            "task": (
                "Compare options and recommend one based on fit, ergonomics, "
                "and maintenance cost. Cite docs.",
                "Survey the landscape, identify two viable choices, and "
                "recommend one with justification.",
                "Investigate the question end-to-end and return a single "
                "concrete recommendation with sources.",
            ),
            "recommendation": (
                "Pydantic Settings, for consistency with our existing models",
                "Redis with explicit TTLs, since we already operate it in prod",
                "Unleash, because it gives us percentage rollouts and audit logs out of the box",
                "Dramatiq, for the simpler operational model on small teams",
                "openapi-typescript, since it's stateless and has no runtime",
            ),
        },
    )


def _t_background_check() -> ToolTemplate:
    return ToolTemplate(
        tool="background_check",
        user_intents=(
            "How is task {tid} doing?",
            "Check on {tid} for me.",
            "What's the status of background task {tid}?",
            "Is {tid} done yet?",
        ),
        pre_call=(
            "Checking the status of {tid}.",
            "Polling {tid}.",
            "Looking up {tid}.",
        ),
        post_call=(
            "Still {state}. {detail} Check back in a couple minutes if you "
            "want to confirm completion.",
            "{state.capitalize}. {detail}",
            "Currently {state}. {detail} Nothing actionable yet.",
        ),
        arg_specs=({"task_id": "{tid}"},),
        output_templates=(
            "task_id: {tid}\n"
            "status: {state}\n"
            "elapsed: {elapsed}\n"
            "command: {cmd}\n"
            "exit_code: null\n",
        ),
        slots={
            "tid": (
                "build_42",
                "deploy_17",
                "ingest_88",
                "backfill_201",
                "lint_99",
                "test_run_55",
            ),
            "state": ("running", "running", "completed", "running"),
            "detail": (
                "Currently in the middle of dependency installation.",
                "Just started building Docker layers.",
                "Compiling assets right now.",
                "Streaming logs - nothing concerning so far.",
            ),
            "elapsed": ("1m 04s", "3m 22s", "5m 41s", "8m 12s"),
            "cmd": (
                "docker build -t app:dev .",
                "kubectl apply -f deploy/k8s/",
                "python scripts/ingest_warehouse.py",
                "alembic upgrade head",
                "ruff check . --fix",
                "pytest tests/integration/",
            ),
        },
    )


def _t_notebook_edit() -> ToolTemplate:
    return ToolTemplate(
        tool="notebook_edit",
        user_intents=(
            "Update cell {cell} of {nb} to {change}.",
            "In {nb}, replace cell {cell}'s contents with {change}.",
            "Modify {nb} - cell {cell} should now {change}.",
            "Edit {nb} cell {cell}: {change}.",
        ),
        pre_call=(
            "Editing cell {cell} of {nb}.",
            "Updating cell {cell} in {nb}.",
            "Replacing cell {cell} of {nb}.",
        ),
        post_call=(
            "Cell {cell} updated. Re-run from {cell} downward to make sure "
            "nothing depends on the prior output.",
            "Done - cell {cell} now reflects the change. Worth restarting the "
            "kernel if any earlier cell was caching state.",
            "Edit applied. The notebook will need a re-run from the modified "
            "cell to pick up the new value.",
        ),
        arg_specs=({"notebook_path": "{nb}", "cell_id": "{cell}", "new_source": "{source}"},),
        output_templates=("edited cell {cell} (code) in {nb}",),
        slots={
            "nb": (
                "notebooks/exploration.ipynb",
                "notebooks/training.ipynb",
                "notebooks/eval.ipynb",
                "notebooks/data_audit.ipynb",
            ),
            "cell": ("2", "3", "5", "7"),
            "change": (
                "use df.describe(include='all') instead of df.head()",
                "set learning_rate to 2e-5",
                "increase batch size to 32",
                "switch to the new tokenizer config",
            ),
            "source": (
                "df.describe(include='all')",
                "learning_rate = 2e-5",
                "batch_size = 32",
                "tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')",
            ),
        },
    )


def _t_tasks() -> ToolTemplate:
    return ToolTemplate(
        tool="tasks",
        user_intents=(
            "Add a task: {title}.",
            "Queue up a task to {title}.",
            "Track this work item - {title}.",
            "Make a note to {title} as a task.",
        ),
        pre_call=(
            "Adding it to the task list.",
            "Queueing the task.",
            "Recording the task.",
        ),
        post_call=(
            "Done - added as task #{tid} ({priority} priority). Anything else you want queued?",
            "Got it. Task #{tid} is on the list. The {priority}-priority bucket "
            "is reasonable - bump it if it becomes blocking.",
            "Added as #{tid} with {priority} priority. Want me to set a follow-up reminder?",
        ),
        arg_specs=({"action": "create", "title": "{title}", "priority": "{priority}"},),
        output_templates=(
            "added task #{tid}\n  title: {title}\n  status: pending  priority: {priority}\n",
        ),
        slots={
            "title": (
                "Migrate the legacy webhook handler from requests to httpx",
                "Add Sentry breadcrumbs to the billing service",
                "Write integration tests for the payment retry queue",
                "Document the new SSO onboarding flow",
                "Backfill the missing email_verified flag",
                "Audit and rotate the cron-triggered API tokens",
            ),
            "priority": ("normal", "high", "low", "normal"),
            "tid": ("7", "11", "13", "18", "22"),
        },
    )


def _t_repo_map() -> ToolTemplate:
    return ToolTemplate(
        tool="repo_map",
        user_intents=(
            "Give me a high-level map of {root}.",
            "Show the repo layout under {root}.",
            "Map out {root} so I can orient.",
            "I'm new to {root} - quick layout overview please.",
        ),
        pre_call=(
            "Generating a repo map for {root}.",
            "Mapping {root}.",
            "Pulling a high-level layout for {root}.",
        ),
        post_call=(
            "Standard layout - HTTP layer, business logic, async work, and "
            "tests in mirrored directories. Migrations are managed alongside "
            "the rest of the data layer.",
            "Clean separation of concerns: routes, services, workers, tests. "
            "If you're touching domain logic, the services directory is where "
            "to start.",
            "Conventional FastAPI + SQLAlchemy structure. Routes feed services, "
            "services own the data layer, workers handle async background work.",
        ),
        arg_specs=({"root": "{root}", "max_depth": 2},),
        output_templates=(
            "{root}/\n"
            "  app/\n"
            "    api/         # HTTP routers\n"
            "    core/        # settings, dependency injection\n"
            "    db/          # models + migrations\n"
            "    services/    # domain logic\n"
            "    workers/     # async tasks\n"
            "  tests/         # mirrors app/ layout\n"
            "  scripts/       # one-off ops\n"
            "  pyproject.toml | Dockerfile | docker-compose.yml\n",
        ),
        slots={"root": _PROJECT_PATHS},
    )


def _t_diff_apply() -> ToolTemplate:
    return ToolTemplate(
        tool="diff_apply",
        user_intents=(
            "Apply this patch to {path}:\n\n{diff}",
            "Land this hunk:\n\n{diff}",
            "Patch {path} with the following:\n\n{diff}",
            "Apply the upstream fix:\n\n{diff}",
        ),
        pre_call=(
            "Applying the unified diff.",
            "Landing the patch.",
            "Applying the hunk.",
        ),
        post_call=(
            "Patch applied cleanly. Worth running the test suite before "
            "committing - hunk swaps like this are easy to mis-anchor.",
            "Hunk applied. The file now reflects the upstream version of the {path} change.",
            "Diff landed. If the surrounding code was already modified locally, "
            "double-check the surrounding context still compiles.",
        ),
        arg_specs=({"diff": "{diff}"},),
        output_templates=("applied 1 hunk to {path} (1+, 1-)",),
        slots={
            "path": (
                "app/api/pagination.py",
                "app/services/billing.py",
                "app/workers/runner.py",
                "app/core/security.py",
                "app/db/models.py",
            ),
            "diff": (
                "--- a/app/api/pagination.py\n"
                "+++ b/app/api/pagination.py\n"
                "@@ -14,7 +14,7 @@\n"
                "     start = (page - 1) * per_page\n"
                "-    end = start + per_page - 1\n"
                "+    end = start + per_page\n"
                "     return start, end",
                "--- a/app/services/billing.py\n"
                "+++ b/app/services/billing.py\n"
                "@@ -42,7 +42,7 @@\n"
                "     amount = invoice.subtotal\n"
                "-    tax = amount * 0.07\n"
                "+    tax = amount * Decimal('0.0725')\n"
                "     return amount + tax",
                "--- a/app/core/security.py\n"
                "+++ b/app/core/security.py\n"
                "@@ -8,7 +8,7 @@\n"
                "     payload = jwt.decode(token, key, algorithms=['HS256'])\n"
                "-    return payload['sub']\n"
                "+    return payload.get('sub') or ''\n",
            ),
        },
    )


_TEMPLATE_BUILDERS: dict[str, Any] = {
    "web_search": _t_web_search,
    "web_fetch": _t_web_fetch,
    "image_read": _t_image_read,
    "pdf_read": _t_pdf_read,
    "spawn_agent": _t_spawn_agent,
    "background_check": _t_background_check,
    "notebook_edit": _t_notebook_edit,
    "tasks": _t_tasks,
    "repo_map": _t_repo_map,
    "diff_apply": _t_diff_apply,
}


# ---------------------------------------------------------------------------
# Sample synthesis
# ---------------------------------------------------------------------------


def _draw_slots(template: ToolTemplate, rng: random.Random) -> dict[str, str]:
    """Pick one value per slot the template advertises."""
    return {key: rng.choice(values) for key, values in template.slots.items()}


def _render_or_passthrough(text: str, slots: dict[str, str]) -> str:
    keys = _slot_keys(text)
    if not keys:
        return text
    # Allow ".capitalize" pseudo-formatting via a quick post-process.
    if any(k.endswith(".capitalize") for k in keys):
        bare = {k.replace(".capitalize", ""): v.capitalize() for k, v in slots.items()}
        return text.format(**slots, **{f"{k}.capitalize": v for k, v in bare.items()})
    return text.format(**slots)


def _build_sample(
    template: ToolTemplate,
    rng: random.Random,
    tools_schema: list[dict[str, Any]],
) -> dict[str, Any]:
    slots = _draw_slots(template, rng)
    user_text = _render_or_passthrough(rng.choice(template.user_intents), slots)
    pre_text = _render_or_passthrough(rng.choice(template.pre_call), slots)
    arg_spec = rng.choice(template.arg_specs)
    args = {
        key: _render_or_passthrough(value, slots) if isinstance(value, str) else value
        for key, value in arg_spec.items()
    }
    output_text = _render_or_passthrough(rng.choice(template.output_templates), slots)
    post_text = _render_or_passthrough(rng.choice(template.post_call), slots)

    call_id = "call_00"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {
            "role": "assistant",
            "content": pre_text,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": template.tool,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output_text},
        {"role": "assistant", "content": post_text},
    ]
    return {"messages": messages, "tools": tools_schema}


def build_augment_samples(
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
    tools: tuple[str, ...] = TARGET_TOOLS,
) -> list[dict[str, Any]]:
    """Generate ``total`` samples evenly distributed across ``tools``.

    Per-tool counts are exactly ``total // len(tools)``. Each tool draws from
    its own ``random.Random(seed + i)``, so re-rolling one tool doesn't
    perturb the others.
    """
    if total % len(tools) != 0:
        raise ValueError(f"total ({total}) must divide evenly across {len(tools)} tools")
    per_tool = total // len(tools)
    tools_schema = get_tool_schemas()

    samples: list[dict[str, Any]] = []
    for i, name in enumerate(tools):
        builder = _TEMPLATE_BUILDERS[name]
        template = builder()
        rng = random.Random(seed + i)
        for _ in range(per_tool):
            samples.append(_build_sample(template, rng, tools_schema))
    return samples


def write_augment_jsonl(
    output_path: Path,
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    samples = build_augment_samples(total=total, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for rec in samples:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tool_usage: Counter[str] = Counter()
    for rec in samples:
        for msg in rec["messages"]:
            for tc in msg.get("tool_calls") or []:
                tool_usage[tc["function"]["name"]] += 1
    return {"written": len(samples), "tool_usage": dict(tool_usage)}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase A1 augmented samples.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase_a1/data/phase_a1_augmented.jsonl"),
    )
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    summary = write_augment_jsonl(args.output, total=args.total, seed=args.seed)
    logger.info("wrote %d augmented samples to %s", summary["written"], args.output)
    logger.info(
        "tool usage (sorted): %s",
        dict(sorted(summary["tool_usage"].items(), key=lambda kv: -kv[1])),
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
