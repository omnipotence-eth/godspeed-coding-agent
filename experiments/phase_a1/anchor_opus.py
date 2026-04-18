"""Stage F — Opus-hand-authored gold anchor samples for Phase A1.

50 multi-turn ``{messages, tools}`` records covering all 21 canonical
Godspeed tools. Each record is a publication-quality reference for what a
"5/5/5/5" judge verdict looks like (tool_correctness, arg_correctness,
realism, coherence). Used as:

  * Few-shot calibration for the judge (``judge.load_few_shots``).
  * Held-out eval set for measuring generation quality drift over time.
  * Regression fixture for ``validate.py`` — every sample must validate.

Coverage target: every tool appears at least twice across the 50 records.
Category mix: ~30 single-tool, ~10 multi-turn, ~5 error-recovery, ~5
no-tool conversational.

Run:  ``python -m experiments.phase_a1.anchor_opus --output <path>``
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.phase_a1.executor import _SYSTEM_PROMPT
from experiments.phase_a1.registry_builder import ALL_TOOLS, get_tool_schemas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny builder helpers — keep the 50 sample bodies readable
# ---------------------------------------------------------------------------


def _sys() -> dict[str, Any]:
    return {"role": "system", "content": _SYSTEM_PROMPT}


def _u(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _tc(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    """Construct an OpenAI-format tool_call entry with JSON-encoded arguments."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _a(content: str, *tool_calls: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = list(tool_calls)
    return msg


def _t(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ---------------------------------------------------------------------------
# Samples 1-30: single-tool exercises (one tool call, two-turn assistant)
# ---------------------------------------------------------------------------


def s01() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Read the database config and tell me the pool size."),
        _a(
            "I'll read the database configuration to find the pool size.",
            _tc("c0", "file_read", file_path="config/database.yaml"),
        ),
        _t(
            "c0",
            "default: &default\n"
            "  adapter: postgresql\n"
            "  pool: 25\n"
            "  timeout: 5000\n"
            "  encoding: unicode\n\n"
            "production:\n"
            "  <<: *default\n"
            "  database: app_prod\n"
            "  url: <%= ENV['DATABASE_URL'] %>\n",
        ),
        _a(
            "Pool size is **25** in the default block, inherited by `production` via the YAML "
            "anchor (`<<: *default`). Override it per-environment by adding `pool:` under the "
            "`production:` key if prod needs a different value."
        ),
    ]


def s02() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Create tests/test_email_normalize.py with a test that asserts "
            "normalize_email('  Foo@BAR.com ') == 'foo@bar.com'."
        ),
        _a(
            "Creating the new test module under tests/.",
            _tc(
                "c0",
                "file_write",
                file_path="tests/test_email_normalize.py",
                content=(
                    "from app.utils.email import normalize_email\n\n\n"
                    "def test_normalize_email_strips_and_lowercases() -> None:\n"
                    '    assert normalize_email("  Foo@BAR.com ") == "foo@bar.com"\n'
                ),
            ),
        ),
        _t("c0", "wrote tests/test_email_normalize.py (5 lines, 142 bytes)"),
        _a(
            "Test file created. Run `pytest tests/test_email_normalize.py -q` to confirm it "
            "passes against the current `normalize_email` implementation."
        ),
    ]


def s03() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Bump MAX_RETRIES from 3 to 5 in app/http/client.py."),
        _a(
            "Editing the constant in place.",
            _tc(
                "c0",
                "file_edit",
                file_path="app/http/client.py",
                old_string="MAX_RETRIES = 3",
                new_string="MAX_RETRIES = 5",
            ),
        ),
        _t("c0", "edited app/http/client.py: 1 replacement"),
        _a(
            "`MAX_RETRIES` is now 5. Worth checking that the exponential-backoff total wait "
            "(typically `base * (2**retries)`) is still under whatever request-timeout SLO "
            "callers depend on — bumping retries can push the worst-case latency past it."
        ),
    ]


def s04() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Apply this upstream patch that fixes the off-by-one in pagination:\n\n"
            "--- a/app/api/pagination.py\n"
            "+++ b/app/api/pagination.py\n"
            "@@ -14,7 +14,7 @@ def page_bounds(page: int, per_page: int) -> tuple[int, int]:\n"
            "     if page < 1:\n"
            '         raise ValueError("page must be >= 1")\n'
            "     start = (page - 1) * per_page\n"
            "-    end = start + per_page - 1\n"
            "+    end = start + per_page\n"
            "     return start, end"
        ),
        _a(
            "Applying the unified diff.",
            _tc(
                "c0",
                "diff_apply",
                diff=(
                    "--- a/app/api/pagination.py\n"
                    "+++ b/app/api/pagination.py\n"
                    "@@ -14,7 +14,7 @@ def page_bounds("
                    "page: int, per_page: int) -> tuple[int, int]:\n"
                    "     if page < 1:\n"
                    '         raise ValueError("page must be >= 1")\n'
                    "     start = (page - 1) * per_page\n"
                    "-    end = start + per_page - 1\n"
                    "+    end = start + per_page\n"
                    "     return start, end\n"
                ),
            ),
        ),
        _t("c0", "applied 1 hunk to app/api/pagination.py (1+, 1-)"),
        _a(
            "Patch applied cleanly. The bound is now half-open `[start, end)`, matching "
            "Python slicing convention — make sure any caller that did `rows[start:end+1]` "
            "is updated, otherwise it will read one row past the page."
        ),
    ]


def s05() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Find every Alembic migration file in the repo."),
        _a(
            "Globbing for migration scripts under the standard Alembic layout.",
            _tc("c0", "glob_search", pattern="alembic/versions/*.py"),
        ),
        _t(
            "c0",
            "alembic/versions/0001_initial.py\n"
            "alembic/versions/0002_add_users_table.py\n"
            "alembic/versions/0003_add_orders.py\n"
            "alembic/versions/0004_add_index_orders_user.py\n"
            "alembic/versions/0005_add_status_to_orders.py\n"
            "(5 matches)",
        ),
        _a(
            "5 migrations, numbered 0001 through 0005. Latest is "
            "`0005_add_status_to_orders.py` — that's the head if you're about to add another."
        ),
    ]


def s06() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Show me every TODO comment in src/."),
        _a(
            "Searching for TODO markers across src/.",
            _tc("c0", "grep_search", pattern=r"TODO[:\s]", path="src/"),
        ),
        _t(
            "c0",
            "src/payments/refund.py:42:    # TODO: idempotency key here once Stripe supports v2\n"
            "src/auth/session.py:87:    # TODO: rotate the signing secret on every deploy\n"
            "src/billing/invoice.py:159:    # TODO: handle partial credits (see RFC #214)\n"
            "src/cli/seed.py:23:    # TODO: parametrize the seed dataset path\n"
            "(4 matches)",
        ),
        _a(
            "4 TODOs in `src/`: refund idempotency, session secret rotation, partial credits, "
            "and the seeding script's hardcoded path. The Stripe and RFC #214 ones look "
            "blocked on external work; the other two are local cleanup."
        ),
    ]


def s07() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Where in this codebase do we handle JWT refresh?"),
        _a(
            "Doing a semantic search for the refresh-token logic.",
            _tc("c0", "code_search", query="JWT refresh token rotation handler"),
        ),
        _t(
            "c0",
            "## src/auth/jwt.py (lines 88-126, score=0.83)\n"
            "def refresh_access_token(refresh: str) -> TokenPair:\n"
            "    payload = decode_refresh(refresh)\n"
            "    if _is_revoked(payload['jti']):\n"
            "        raise InvalidRefresh('revoked')\n"
            "    new_access = mint_access(payload['sub'])\n"
            "    new_refresh = rotate_refresh(payload)\n"
            "    return TokenPair(new_access, new_refresh)\n\n"
            "## src/api/auth_routes.py (lines 41-58, score=0.71)\n"
            '@router.post("/auth/refresh")\n'
            "async def refresh(req: RefreshIn) -> TokenPair:\n"
            "    return refresh_access_token(req.refresh_token)\n",
        ),
        _a(
            "Two relevant locations:\n"
            "- `src/auth/jwt.py:88` — `refresh_access_token()` does the actual decode, "
            "revocation check, and rotation.\n"
            "- `src/api/auth_routes.py:41` — the `POST /auth/refresh` HTTP entry point that "
            "calls into it."
        ),
    ]


def s08() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Give me a quick high-level map of the repo."),
        _a(
            "Generating a high-level repo map.",
            _tc("c0", "repo_map", max_depth=2),
        ),
        _t(
            "c0",
            "app/\n"
            "  api/         # FastAPI routers (auth, billing, users)\n"
            "  core/        # settings, logging, dependency injection\n"
            "  db/          # SQLAlchemy models + Alembic migrations\n"
            "  services/    # domain logic (PaymentService, EmailService, ...)\n"
            "  workers/     # Celery tasks\n"
            "tests/         # pytest suite, mirrors app/ layout\n"
            "scripts/       # one-off ops scripts\n"
            "alembic/       # migration env\n"
            "pyproject.toml | uv.lock | Dockerfile | docker-compose.yml\n",
        ),
        _a(
            "Standard FastAPI + SQLAlchemy layout: HTTP layer in `app/api/`, business logic in "
            "`app/services/`, async work in `app/workers/`, tests mirror the app tree. "
            "Migrations are Alembic-managed."
        ),
    ]


def s09() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("List installed Python packages so I can see what's pinned."),
        _a(
            "Running `pip list` in the project venv.",
            _tc("c0", "shell", command="pip list --format=columns"),
        ),
        _t(
            "c0",
            "Package          Version\n"
            "---------------- -------\n"
            "fastapi          0.115.6\n"
            "pydantic         2.9.2\n"
            "sqlalchemy       2.0.36\n"
            "alembic          1.14.0\n"
            "uvicorn          0.32.1\n"
            "httpx            0.28.1\n"
            "pytest           8.3.4\n"
            "ruff             0.8.4\n",
        ),
        _a(
            "8 packages installed; nothing exotic. Pydantic v2 and SQLAlchemy 2.x — both "
            "modern majors. If you want exact transitive pins, `pip freeze` shows the full "
            "lock; `pip list` only shows top-level."
        ),
    ]


def s10() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Run the tests in tests/services/test_payment.py."),
        _a(
            "Running pytest scoped to that file.",
            _tc("c0", "test_runner", path="tests/services/test_payment.py"),
        ),
        _t(
            "c0",
            "============================= test session starts =============================\n"
            "platform linux -- Python 3.12.7, pytest-8.3.4\n"
            "collected 14 items\n\n"
            "tests/services/test_payment.py ..............                            [100%]\n\n"
            "============================= 14 passed in 1.28s ==============================\n",
        ),
        _a("All 14 tests in `tests/services/test_payment.py` pass (1.28s, no warnings)."),
    ]


def s11() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Type-check app/services/billing.py with mypy."),
        _a(
            "Running mypy via the verify tool, scoped to that file.",
            _tc("c0", "verify", file_path="app/services/billing.py", check_type="mypy"),
        ),
        _t(
            "c0",
            'app/services/billing.py:54: error: Argument 1 to "charge" of "PaymentClient" '
            'has incompatible type "str"; expected "Decimal"  [arg-type]\n'
            "app/services/billing.py:88: error: Returning Any from function declared to "
            'return "Invoice"  [no-any-return]\n'
            "Found 2 errors in 1 file (checked 1 source file)\n",
        ),
        _a(
            "Mypy found two real bugs:\n"
            "- **L54**: passing a `str` where `PaymentClient.charge` expects `Decimal` — "
            "wrap with `Decimal(amount)` (and validate first if `amount` is user input).\n"
            "- **L88**: function annotated `-> Invoice` is returning `Any` — usually means a "
            "downstream call lost its type. Annotate the returned object or add a cast."
        ),
    ]


def s12() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("How's the long Docker build going? It's task ID build_42."),
        _a(
            "Checking the background task status.",
            _tc("c0", "background_check", task_id="build_42"),
        ),
        _t(
            "c0",
            "task_id: build_42\n"
            "status: running\n"
            "elapsed: 4m 12s\n"
            "command: docker build -t app:dev .\n"
            "last_stdout: 'Step 11/18 : RUN uv sync --frozen'\n"
            "exit_code: null\n",
        ),
        _a(
            "Still running, 4m12s in. Currently on step 11/18 (`uv sync --frozen`) — that's "
            "usually the longest step. No errors so far. Check back in a couple minutes."
        ),
    ]


def s13() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("What's the working-tree status?"),
        _a(
            "Running `git status`.",
            _tc("c0", "git", action="status"),
        ),
        _t(
            "c0",
            "On branch feat/payment-retries\n"
            "Your branch is ahead of 'origin/feat/payment-retries' by 2 commits.\n\n"
            "Changes not staged for commit:\n"
            "  modified:   app/services/payment.py\n"
            "  modified:   tests/services/test_payment.py\n\n"
            "Untracked files:\n"
            "  scripts/replay_failed_charges.py\n",
        ),
        _a(
            "On `feat/payment-retries`, 2 commits ahead of origin. Two modified files "
            "(`payment.py` + its test) and one new untracked script "
            "(`scripts/replay_failed_charges.py`). Nothing staged yet."
        ),
    ]


def s14() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("List the open PRs on this repo."),
        _a(
            "Listing open pull requests via the GitHub tool.",
            _tc("c0", "github", action="list_prs", state="open"),
        ),
        _t(
            "c0",
            "#412  feat: add payment retry queue       @tremayne   ~2h    approved\n"
            "#410  fix: race in session refresh path   @maria      ~1d    changes_requested\n"
            "#407  chore: bump pydantic to 2.10        dependabot  ~3d    needs_review\n"
            "#398  docs: README quickstart cleanup     @sam        ~6d    needs_review\n",
        ),
        _a(
            "4 open PRs:\n"
            "- **#412** (yours, approved) — ready to merge.\n"
            "- **#410** (Maria) — has change requests; she's blocked on review feedback.\n"
            "- **#407** (Dependabot) — pydantic 2.10 bump, awaiting review.\n"
            "- **#398** (Sam) — README cleanup, awaiting review."
        ),
    ]


def s15() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Find official docs for Stripe's PaymentIntent automatic_payment_methods option."),
        _a(
            "Searching the web for the official Stripe documentation.",
            _tc(
                "c0",
                "web_search",
                query="Stripe PaymentIntent automatic_payment_methods documentation",
            ),
        ),
        _t(
            "c0",
            "1. PaymentIntents API | Stripe Docs — stripe.com/docs/api/payment_intents/object\n"
            "   '...automatic_payment_methods.enabled (boolean) ... When enabled, the "
            "PaymentIntent will accept all eligible payment methods enabled in your "
            "Dashboard...'\n"
            "2. Accept a payment | Stripe Docs — stripe.com/docs/payments/accept-a-payment\n"
            "   'Use automatic_payment_methods to let Stripe choose the optimal mix of "
            "payment methods for the customer's region.'\n"
            "3. PaymentIntent | Stripe API Reference — stripe.com/docs/api/payment_intents/create\n"
            '   \'automatic_payment_methods.allow_redirects (string, default: "always") '
            "controls redirect-based methods like iDEAL or Klarna.'\n",
        ),
        _a(
            "Three good hits, all from stripe.com/docs:\n"
            "- **API object reference** — defines the field on the PaymentIntent.\n"
            "- **Accept a payment guide** — usage in context of a real checkout flow.\n"
            "- **Create endpoint reference** — covers the `allow_redirects` sub-option.\n\n"
            "Want me to fetch any of those for the full text?"
        ),
    ]


def s16() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Fetch RFC 7617 (Basic auth) and tell me what the realm parameter actually does."),
        _a(
            "Fetching the RFC text.",
            _tc("c0", "web_fetch", url="https://www.rfc-editor.org/rfc/rfc7617.txt"),
        ),
        _t(
            "c0",
            "RFC 7617 — The 'Basic' HTTP Authentication Scheme\n"
            "...\n"
            "2.  The 'Basic' Authentication Scheme\n"
            "    ... A server receiving a request for a URI within the protection space\n"
            "    can challenge the client by sending a 401 (Unauthorized) status with a\n"
            '    WWW-Authenticate header containing at least the "Basic" scheme and a\n'
            '    "realm" parameter (Section 2.2 of [RFC7235]).\n\n'
            "2.2.  The 'realm' Parameter\n"
            "    The 'realm' parameter ... is a string that the user agent SHOULD\n"
            "    display to the user so they can pick the right credentials. It also\n"
            "    defines a protection space: requests to URIs in the same realm reuse\n"
            "    the same credentials. The realm is opaque to the client; the server\n"
            "    chooses any string.\n",
        ),
        _a(
            "Per RFC 7617 §2.2, the `realm` parameter does two things:\n"
            "1. **UI hint** — the browser shows it in the credential prompt so users know "
            "which login to use.\n"
            "2. **Credential cache key** — URIs sharing the same realm reuse the same "
            "credentials; different realms force a re-prompt.\n\n"
            "It's an opaque string; the server chooses it freely."
        ),
    ]


def s17() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Look at screenshots/error_dialog.png and describe what's wrong."),
        _a(
            "Inspecting the screenshot.",
            _tc("c0", "image_read", file_path="screenshots/error_dialog.png"),
        ),
        _t(
            "c0",
            "[image: 1024x768 PNG]\n"
            "Description (vision model output):\n"
            "A web app modal dialog with a red header reading 'Error 500'. The body says "
            "'An unexpected error occurred. Please try again later.' Below it, a stack-trace "
            "preview is partially visible: 'TypeError: Cannot read properties of undefined "
            "(reading id) at UserCard.render (UserCard.tsx:42:18)'. A 'Close' button is "
            "shown bottom-right.\n",
        ),
        _a(
            "The dialog is showing a generic 500 to the user, but the visible stack trace "
            "reveals the actual cause: `TypeError` in `UserCard.tsx:42` — the component is "
            "trying to read `.id` on an undefined object. Two issues here:\n"
            "1. **Bug**: `UserCard` isn't guarding against a missing user prop.\n"
            "2. **Leak**: production should never render the raw stack trace in the dialog "
            "— wrap it in a dev-only conditional or send it to your error tracker instead."
        ),
    ]


def s18() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Read the GRPO paper at papers/shao_2024_grpo.pdf and summarize the algorithm."),
        _a(
            "Reading the PDF.",
            _tc("c0", "pdf_read", file_path="papers/shao_2024_grpo.pdf", pages="1-4"),
        ),
        _t(
            "c0",
            "[Page 1] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open\n"
            "Language Models. Shao et al., 2024.\n\n"
            "[Page 3] Group Relative Policy Optimization (GRPO).\n"
            "For each prompt q, sample G outputs {o_1,...,o_G} from the old policy "
            "pi_theta_old. Compute scalar reward r_i for each. Compute the group-relative\n"
            "advantage:  A_i = (r_i - mean(r))/std(r).  Update the policy by maximizing the\n"
            "PPO-style clipped surrogate using A_i as the per-sample advantage. No critic\n"
            "network is trained — the group itself acts as the value baseline.\n",
        ),
        _a(
            "GRPO's core idea (Shao et al., 2024, §3):\n"
            "1. Sample **G** completions per prompt from the old policy.\n"
            "2. Score each one (`r_i`) with the reward model.\n"
            "3. Z-normalize within the group: `A_i = (r_i - mean(r)) / std(r)`.\n"
            "4. Apply the standard PPO clipped surrogate using `A_i` as the advantage — "
            "no value network needed, the group mean *is* the baseline.\n\n"
            "That's why GRPO is cheaper than PPO: it skips training and storing a critic."
        ),
    ]


def s19() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "In notebooks/exploration.ipynb, replace cell 3's `df.head()` with "
            "`df.describe(include='all')`."
        ),
        _a(
            "Editing cell 3 of the notebook.",
            _tc(
                "c0",
                "notebook_edit",
                notebook_path="notebooks/exploration.ipynb",
                cell_id="3",
                new_source="df.describe(include='all')",
            ),
        ),
        _t("c0", "edited cell 3 (code) in notebooks/exploration.ipynb"),
        _a(
            "Cell 3 now calls `df.describe(include='all')`. Re-run it and the cells below "
            "if any of them depend on the old `df.head()` output."
        ),
    ]


def s20() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Add a task: write integration tests for the new payment retry queue."),
        _a(
            "Adding it to the task list.",
            _tc(
                "c0",
                "tasks",
                action="add",
                title="Write integration tests for payment retry queue",
            ),
        ),
        _t(
            "c0",
            "added task #7\n"
            "  title: Write integration tests for payment retry queue\n"
            "  status: pending\n"
            "  priority: normal\n",
        ),
        _a("Added as task #7 (pending). Anything else you want queued up?"),
    ]


def s21() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Spawn a researcher subagent to figure out which OTel exporter we should "
            "use for traces."
        ),
        _a(
            "Delegating that research to a subagent.",
            _tc(
                "c0",
                "spawn_agent",
                agent="researcher",
                task=(
                    "Compare OpenTelemetry trace exporters for a Python FastAPI service "
                    "deploying to GCP. Cover: OTLP gRPC, OTLP HTTP, and Cloud Trace direct. "
                    "Recommend one based on latency overhead, ease of setup, and vendor "
                    "lock-in. Cite docs."
                ),
            ),
        ),
        _t(
            "c0",
            "[researcher subagent finished]\n"
            "Recommendation: OTLP gRPC into the OpenTelemetry Collector running as a sidecar.\n"
            "Reasoning:\n"
            " - OTLP gRPC has the lowest per-span overhead (binary protobuf, multiplexed).\n"
            " - The collector decouples app from backend, so swapping Cloud Trace for "
            "Honeycomb/Tempo later is a config change, not a redeploy.\n"
            " - Cloud Trace direct exporter works but locks the app to GCP and skips the\n"
            "   sampling/processing the collector gives you for free.\n"
            "Sources: opentelemetry.io/docs/languages/python/exporters/, "
            "cloud.google.com/trace/docs/setup/python-ot.\n",
        ),
        _a(
            "Subagent recommends **OTLP gRPC → OpenTelemetry Collector sidecar**, then "
            "Collector → Cloud Trace. Lowest overhead, no vendor lock-in, and you keep the "
            "Collector's sampling/batching. Direct Cloud Trace exporter is simpler but ties "
            "you to GCP for tracing forever."
        ),
    ]


def s22() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Read package.json and tell me the Node engine constraint."),
        _a(
            "Reading package.json.",
            _tc("c0", "file_read", file_path="package.json"),
        ),
        _t(
            "c0",
            "{\n"
            '  "name": "frontend",\n'
            '  "version": "0.14.2",\n'
            '  "engines": { "node": ">=20.10 <23" },\n'
            '  "scripts": {\n'
            '    "dev": "vite",\n'
            '    "build": "vite build",\n'
            '    "test": "vitest run"\n'
            "  },\n"
            '  "dependencies": { "react": "^18.3.1", "react-dom": "^18.3.1" }\n'
            "}\n",
        ),
        _a(
            "Engine constraint is `node >=20.10 <23`, so anything from 20.10 up to (but "
            "not including) 23 works. Node 20 LTS or 22 are the safe choices."
        ),
    ]


def s23() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Where do we read the SENTRY_DSN env var?"),
        _a(
            "Searching for SENTRY_DSN references.",
            _tc("c0", "grep_search", pattern=r"SENTRY_DSN"),
        ),
        _t(
            "c0",
            'app/core/observability.py:18:    dsn = os.getenv("SENTRY_DSN")\n'
            "app/core/observability.py:19:    if not dsn:\n"
            "app/core/observability.py:23:    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1)\n"
            "deploy/k8s/app-deployment.yaml:48:        - name: SENTRY_DSN\n"
            ".env.example:9:SENTRY_DSN=\n"
            "(5 matches in 3 files)",
        ),
        _a(
            "`SENTRY_DSN` is read in `app/core/observability.py:18` (and only there). It's "
            "wired through `deploy/k8s/app-deployment.yaml:48` for production and stubbed "
            "in `.env.example` for local development."
        ),
    ]


def s24() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Show me the last 5 commits on this branch."),
        _a(
            "Running `git log -5 --oneline`.",
            _tc("c0", "git", action="log", args="-5 --oneline"),
        ),
        _t(
            "c0",
            "8e1c44a feat(payment): retry queue worker with exponential backoff\n"
            "5a0f2b7 test(payment): cover the retry-after-503 path\n"
            "c12b938 refactor(payment): extract RetryPolicy from PaymentService\n"
            "44d0a9e chore: bump httpx to 0.28.1\n"
            "9b21e54 fix(billing): off-by-one in invoice line totals\n",
        ),
        _a(
            "Last 5 commits — three on the new payment retry queue (worker, test, refactor "
            "extraction), an httpx bump, and a billing off-by-one fix. Clean conventional "
            "commit history."
        ),
    ]


def s25() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Pull issue #214 from GitHub — I want the body and the labels."),
        _a(
            "Fetching the issue.",
            _tc("c0", "github", action="get_issue", number=214),
        ),
        _t(
            "c0",
            "#214 — Partial-credit refunds drop the cents column\n"
            "state: open  labels: [bug, billing, P1]  author: @maria\n\n"
            "Body:\n"
            "When a customer is partially refunded (e.g. $12.34 -> $7.50), the resulting\n"
            "credit memo stores 7 instead of 7.50. Suspect we're casting Decimal -> int\n"
            "somewhere in CreditMemoService.create(). Reproduces on every partial.\n",
        ),
        _a(
            "Issue **#214** is open, labeled `bug`, `billing`, `P1`, opened by Maria.\n\n"
            "**Summary**: partial refunds drop the cents — `$7.50` becomes `7`. Likely a "
            "`Decimal -> int` cast in `CreditMemoService.create()`. Reproduces on every "
            "partial refund."
        ),
    ]


def s26() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("How much disk space is free on the build host?"),
        _a(
            "Running `df -h /`.",
            _tc("c0", "shell", command="df -h /"),
        ),
        _t(
            "c0",
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            "/dev/nvme0n1p2  468G  312G  133G  71% /\n",
        ),
        _a(
            "Root filesystem is 71% full — **133G free of 468G**. Comfortable for now, but "
            "Docker layer caches and CI artifacts can climb fast. Worth pruning if it crosses "
            "85%."
        ),
    ]


def s27() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "I keep getting `psycopg.errors.InsufficientPrivilege: permission denied for "
            "schema public` after upgrading to Postgres 15. What changed?"
        ),
        _a(
            "Searching for context on the Postgres 15 schema-public change.",
            _tc(
                "c0",
                "web_search",
                query="postgres 15 permission denied for schema public CREATE",
            ),
        ),
        _t(
            "c0",
            "1. PostgreSQL 15 release notes — postgresql.org/docs/15/release-15.html\n"
            "   'Remove PUBLIC creation permission on the public schema (Noah Misch). The "
            "change applies to new databases; existing databases are unaffected on upgrade.'\n"
            "2. dba.stackexchange.com — '... since PG15, only the database owner has CREATE "
            "on public by default. Grant explicitly: GRANT CREATE ON SCHEMA public TO myuser;'\n"
            "3. supabase.com/blog/postgres-15-public-schema — explainer with migration tips.\n",
        ),
        _a(
            "Postgres 15 removed the implicit `CREATE` grant that all roles used to have on "
            "the `public` schema (it had been a long-standing security footgun). Two fixes:\n"
            "- **Explicit grant** (preferred): "
            "`GRANT CREATE ON SCHEMA public TO your_app_user;`\n"
            "- **Or restore the old behavior**: "
            "`GRANT CREATE ON SCHEMA public TO PUBLIC;` — not recommended for prod."
        ),
    ]


def s28() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Find the definition of class RetryPolicy."),
        _a(
            "Searching for the class definition.",
            _tc("c0", "code_search", query="class RetryPolicy definition"),
        ),
        _t(
            "c0",
            "## app/services/retry.py (lines 12-44, score=0.91)\n"
            "@dataclass(frozen=True)\n"
            "class RetryPolicy:\n"
            "    max_attempts: int = 5\n"
            "    base_delay_s: float = 0.5\n"
            "    max_delay_s: float = 30.0\n"
            "    jitter: bool = True\n"
            "    retry_on: tuple[type[Exception], ...] = (TransientError,)\n\n"
            "    def delay_for(self, attempt: int) -> float:\n"
            "        d = min(self.base_delay_s * 2 ** (attempt - 1), self.max_delay_s)\n"
            "        if self.jitter:\n"
            "            d *= random.uniform(0.5, 1.5)\n"
            "        return d\n",
        ),
        _a(
            "Defined in `app/services/retry.py:12` — frozen dataclass with `max_attempts=5`, "
            "exponential backoff capped at 30s, optional jitter (±50%), and a tuple of "
            "exception types to retry on (defaults to `TransientError`). The `delay_for` "
            "method computes the per-attempt sleep."
        ),
    ]


def s29() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Run the integration tests."),
        _a(
            "Running the integration test suite under tests/integration/.",
            _tc("c0", "test_runner", path="tests/integration/", marker="integration"),
        ),
        _t(
            "c0",
            "============================= test session starts =============================\n"
            "platform linux -- Python 3.12.7, pytest-8.3.4\n"
            "collected 38 items\n\n"
            "tests/integration/test_billing_flow.py ............                      [ 31%]\n"
            "tests/integration/test_payment_flow.py ...............                   [ 71%]\n"
            "tests/integration/test_webhook_flow.py ...........                       [100%]\n\n"
            "============================= 38 passed in 14.62s =============================\n",
        ),
        _a(
            "All 38 integration tests pass in 14.6s — billing, payment, and webhook flows "
            "are green."
        ),
    ]


def s30() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Lint app/ with ruff."),
        _a(
            "Running ruff via the verify tool.",
            _tc("c0", "verify", path="app/", check_type="ruff"),
        ),
        _t(
            "c0",
            "app/api/users.py:14:1: F401 [*] `typing.Optional` imported but unused\n"
            "app/services/email.py:88:5: B007 [*] Loop control variable `i` not used in loop "
            "body\n"
            "app/workers/cleanup.py:42:9: SIM108 Use ternary operator instead of if-else\n"
            "Found 3 errors.\n"
            "[*] 2 fixable with the `--fix` option.\n",
        ),
        _a(
            "Ruff found 3 issues in `app/`:\n"
            "- `app/api/users.py:14` — unused `Optional` import (auto-fixable).\n"
            "- `app/services/email.py:88` — unused loop variable `i` (auto-fixable).\n"
            "- `app/workers/cleanup.py:42` — could use a ternary; cosmetic.\n\n"
            "Two of the three are autofixable with `ruff check --fix app/`."
        ),
    ]


# ---------------------------------------------------------------------------
# Samples 31-42: multi-turn (2-3 tool calls each)
# ---------------------------------------------------------------------------


def s31() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Find every place we call `send_email` and read the one in app/workers/digest.py."),
        _a(
            "First locating the call sites with grep.",
            _tc("c0", "grep_search", pattern=r"send_email\(", path="app/"),
        ),
        _t(
            "c0",
            "app/api/users.py:74:    send_email(user.email, 'welcome', ctx)\n"
            "app/workers/digest.py:118:    send_email(sub.email, 'weekly_digest', ctx)\n"
            "app/services/billing.py:201:    send_email(acct.billing_email, 'invoice_ready', ctx)\n"
            "(3 matches)",
        ),
        _a(
            "Three call sites: signup welcome, weekly digest, and invoice notification. "
            "Reading the digest worker now.",
            _tc("c1", "file_read", file_path="app/workers/digest.py"),
        ),
        _t(
            "c1",
            "from app.services.email import send_email\n"
            "from app.db import get_session\n"
            "from app.models import Subscriber\n\n\n"
            "def run_weekly_digest() -> None:\n"
            "    with get_session() as s:\n"
            "        for subscriber in s.query(Subscriber).filter_by(active=True):\n"
            "            ctx = build_digest_context(subscriber)\n"
            "            send_email(subscriber.email, 'weekly_digest', ctx)\n",
        ),
        _a(
            "`run_weekly_digest()` walks all active subscribers, builds a per-user context, "
            "and sends a `weekly_digest`-templated email each. No batching or rate-limiting "
            "— if your subscriber count is large, that loop will hammer the email provider; "
            "consider chunking with a small `time.sleep` or moving each send to a Celery task."
        ),
    ]


def s32() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Find the latest Alembic migration, read it, and add a new column "
            "`archived_at TIMESTAMPTZ NULL` to the `orders` table in a follow-up migration."
        ),
        _a(
            "First listing all Alembic versions to find the head.",
            _tc("c0", "glob_search", pattern="alembic/versions/*.py"),
        ),
        _t(
            "c0",
            "alembic/versions/0001_initial.py\n"
            "alembic/versions/0002_add_users_table.py\n"
            "alembic/versions/0003_add_orders.py\n"
            "alembic/versions/0004_add_index_orders_user.py\n"
            "alembic/versions/0005_add_status_to_orders.py\n"
            "(5 matches)",
        ),
        _a(
            "Head is `0005`. Reading it to grab its revision id.",
            _tc("c1", "file_read", file_path="alembic/versions/0005_add_status_to_orders.py"),
        ),
        _t(
            "c1",
            '"""add status to orders\n\n'
            "Revision ID: a1b2c3d4e5f6\n"
            "Revises: 9988aabbccdd\n"
            'Create Date: 2026-03-30 10:14:00\n"""\n'
            "from alembic import op\n"
            "import sqlalchemy as sa\n\n"
            "revision = 'a1b2c3d4e5f6'\n"
            "down_revision = '9988aabbccdd'\n\n\n"
            "def upgrade() -> None:\n"
            "    op.add_column('orders', sa.Column('status', sa.String(32), nullable=False, "
            "server_default='pending'))\n\n\n"
            "def downgrade() -> None:\n"
            "    op.drop_column('orders', 'status')\n",
        ),
        _a(
            "Got it — head revision is `a1b2c3d4e5f6`. Writing the new migration that adds "
            "`archived_at` and chains off that head.",
            _tc(
                "c2",
                "file_write",
                file_path="alembic/versions/0006_add_archived_at_to_orders.py",
                content=(
                    '"""add archived_at to orders\n\n'
                    "Revision ID: b7c8d9e0f1a2\n"
                    "Revises: a1b2c3d4e5f6\n"
                    'Create Date: 2026-04-17 18:00:00\n"""\n'
                    "from alembic import op\n"
                    "import sqlalchemy as sa\n\n"
                    "revision = 'b7c8d9e0f1a2'\n"
                    "down_revision = 'a1b2c3d4e5f6'\n\n\n"
                    "def upgrade() -> None:\n"
                    "    op.add_column(\n"
                    "        'orders',\n"
                    "        sa.Column('archived_at', sa.DateTime(timezone=True), "
                    "nullable=True),\n"
                    "    )\n\n\n"
                    "def downgrade() -> None:\n"
                    "    op.drop_column('orders', 'archived_at')\n"
                ),
            ),
        ),
        _t(
            "c2",
            "wrote alembic/versions/0006_add_archived_at_to_orders.py (16 lines, 552 bytes)",
        ),
        _a(
            "Migration `0006_add_archived_at_to_orders.py` is in place, chained off "
            "`a1b2c3d4e5f6`. `archived_at` is nullable (no backfill needed). Run "
            "`alembic upgrade head` when you're ready to apply it."
        ),
    ]


def s33() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "I'm done with the payment retry refactor. Stage the modified files and commit "
            "them with a sensible message."
        ),
        _a(
            "First viewing what's changed.",
            _tc("c0", "git", action="diff", args="--stat"),
        ),
        _t(
            "c0",
            "app/services/payment.py        | 64 ++++++++++++++++++++++++++++++++--\n"
            "app/services/retry.py          | 48 +++++++++++++++++++++++++\n"
            "tests/services/test_payment.py | 22 +++++++++++\n"
            "tests/services/test_retry.py   | 31 ++++++++++++++++++\n"
            "4 files changed, 162 insertions(+), 3 deletions(-)\n",
        ),
        _a(
            "Four files: the service rework, the new RetryPolicy module, and tests for both. "
            "Staging them.",
            _tc(
                "c1",
                "git",
                action="add",
                args=(
                    "app/services/payment.py app/services/retry.py "
                    "tests/services/test_payment.py tests/services/test_retry.py"
                ),
            ),
        ),
        _t("c1", "staged 4 paths"),
        _a(
            "Committing.",
            _tc(
                "c2",
                "git",
                action="commit",
                message=(
                    "refactor(payment): extract RetryPolicy and add exponential backoff\n\n"
                    "Move retry logic out of PaymentService into a frozen RetryPolicy\n"
                    "dataclass with configurable max attempts, base/max delay, and jitter.\n"
                    "Adds unit tests for the policy and updates payment tests to cover the\n"
                    "503-retry-after path."
                ),
            ),
        ),
        _t(
            "c2",
            "[feat/payment-retries 7d9e2af] refactor(payment): extract RetryPolicy and add "
            "exponential backoff\n"
            " 4 files changed, 162 insertions(+), 3 deletions(-)\n"
            " create mode 100644 app/services/retry.py\n"
            " create mode 100644 tests/services/test_retry.py\n",
        ),
        _a(
            "Committed as `7d9e2af` on `feat/payment-retries`. The commit message follows "
            "your repo's conventional-commit style with scope and a body explaining the "
            "extraction. Push when you're ready for review."
        ),
    ]


def s34() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "I'm new to this codebase. Give me the high-level layout, then dig into how "
            "authentication is wired."
        ),
        _a(
            "Starting with a top-level repo map.",
            _tc("c0", "repo_map", max_depth=2),
        ),
        _t(
            "c0",
            "app/\n"
            "  api/         # FastAPI routers (auth, billing, users, ...)\n"
            "  core/        # settings, dependency injection, security primitives\n"
            "  db/          # SQLAlchemy models + Alembic\n"
            "  services/    # domain logic\n"
            "  workers/     # Celery tasks\n"
            "tests/\n"
            "alembic/\n"
            "pyproject.toml | Dockerfile | docker-compose.yml\n",
        ),
        _a(
            "Standard FastAPI + SQLAlchemy layout. Authentication should be in `app/api/` "
            "(routes), `app/core/` (security primitives), and `app/services/`. Searching "
            "for the actual entry points.",
            _tc("c1", "code_search", query="JWT authentication FastAPI dependency"),
        ),
        _t(
            "c1",
            "## app/core/security.py (lines 22-58, score=0.88)\n"
            "async def get_current_user(\n"
            "    token: Annotated[str, Depends(oauth2_scheme)],\n"
            "    db: Annotated[Session, Depends(get_db)],\n"
            ") -> User:\n"
            "    payload = jwt.decode(token, settings.jwt_secret, algorithms=['HS256'])\n"
            "    user = db.query(User).get(payload['sub'])\n"
            "    if not user:\n"
            "        raise HTTPException(401, 'invalid token')\n"
            "    return user\n\n"
            "## app/api/users.py (lines 14-22, score=0.74)\n"
            "@router.get('/me')\n"
            "def me(user: User = Depends(get_current_user)) -> UserOut:\n"
            "    return UserOut.model_validate(user)\n",
        ),
        _a(
            "Auth flow:\n"
            "1. **`app/core/security.py`** owns the JWT decode + user lookup, exposed as the "
            "`get_current_user` FastAPI dependency.\n"
            "2. **Any route that needs auth** declares `user: User = Depends(get_current_user)` "
            "— example in `app/api/users.py:14` (the `/me` endpoint).\n\n"
            "Adding a new protected endpoint = same one-line `Depends`. Token issuance lives "
            "in a separate `auth_routes.py` module (search for `mint_access` if you want to "
            "see it)."
        ),
    ]


def s35() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Fetch the OpenAPI schema at https://api.example.com/openapi.json and save it "
            "to schemas/upstream_api.json so we can codegen a client from it."
        ),
        _a(
            "Fetching the schema.",
            _tc("c0", "web_fetch", url="https://api.example.com/openapi.json"),
        ),
        _t(
            "c0",
            '{"openapi":"3.1.0","info":{"title":"Example API","version":"2.4.0"},'
            '"paths":{"/customers":{"get":{"operationId":"listCustomers","responses":'
            '{"200":{"description":"ok"}}}},"/customers/{id}":{"get":{"operationId":'
            '"getCustomer","parameters":[{"name":"id","in":"path","required":true,'
            '"schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}}},'
            '"components":{"schemas":{"Customer":{"type":"object","properties":'
            '{"id":{"type":"string"},"email":{"type":"string","format":"email"}}}}}}',
        ),
        _a(
            "Got it — OpenAPI 3.1 spec, version 2.4.0, two endpoints (`listCustomers`, "
            "`getCustomer`) and a `Customer` schema. Saving it.",
            _tc(
                "c1",
                "file_write",
                file_path="schemas/upstream_api.json",
                content=(
                    '{"openapi":"3.1.0","info":{"title":"Example API","version":"2.4.0"},'
                    '"paths":{"/customers":{"get":{"operationId":"listCustomers",'
                    '"responses":{"200":{"description":"ok"}}}},'
                    '"/customers/{id}":{"get":{"operationId":"getCustomer",'
                    '"parameters":[{"name":"id","in":"path","required":true,'
                    '"schema":{"type":"string"}}],"responses":{"200":{"description":"ok"}}}}},'
                    '"components":{"schemas":{"Customer":{"type":"object","properties":'
                    '{"id":{"type":"string"},"email":{"type":"string","format":"email"}}}}}}'
                ),
            ),
        ),
        _t("c1", "wrote schemas/upstream_api.json (1 line, 552 bytes)"),
        _a(
            "Saved to `schemas/upstream_api.json`. You can now run "
            "`openapi-python-client generate --path schemas/upstream_api.json` (or your "
            "codegen of choice) to scaffold the client. Worth pinning the version in a "
            "comment so future bumps are visible in PR diffs."
        ),
    ]


def s36() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Install the new dev dependencies in the background, watch the install, and "
            "then run the unit tests once it finishes."
        ),
        _a(
            "Kicking off `uv sync` in the background.",
            _tc("c0", "shell", command="uv sync --all-extras --frozen", background=True),
        ),
        _t(
            "c0",
            "started background task install_42\n"
            "  command: uv sync --all-extras --frozen\n"
            "  pid: 18234\n",
        ),
        _a(
            "Running. Polling status.",
            _tc("c1", "background_check", task_id="install_42"),
        ),
        _t(
            "c1",
            "task_id: install_42\n"
            "status: completed\n"
            "exit_code: 0\n"
            "elapsed: 18.4s\n"
            "tail: 'Installed 14 packages in 4.21s'\n",
        ),
        _a(
            "Install finished cleanly (exit 0, 14 new packages, 18.4s). Running the unit suite.",
            _tc("c2", "test_runner", path="tests/unit/"),
        ),
        _t(
            "c2",
            "============================= test session starts =============================\n"
            "collected 142 items\n\n"
            "tests/unit/ ........................................................   [100%]\n\n"
            "============================ 142 passed in 3.14s ==============================\n",
        ),
        _a(
            "All 142 unit tests pass in 3.14s after the dependency bump. Safe to commit "
            "the lockfile changes."
        ),
    ]


def s37() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Look at PR #412 and post a review comment asking the author to add a test "
            "for the case where the queue is empty on startup."
        ),
        _a(
            "Pulling the PR first.",
            _tc("c0", "github", action="get_pr", number=412),
        ),
        _t(
            "c0",
            "#412 — feat: add payment retry queue\n"
            "state: open  base: main  head: feat/payment-retries  author: @tremayne\n"
            "files_changed: 4  +162/-3  reviewers: @maria (approved)\n\n"
            "Body: 'Adds a Redis-backed retry queue with exponential backoff. Worker drains "
            "every 30s. Tests cover the retry-after-503 path. Closes #389.'\n",
        ),
        _a(
            "Got the PR. Posting the review comment.",
            _tc(
                "c1",
                "github",
                action="comment_pr",
                number=412,
                body=(
                    "Could you add a test for the case where the queue is empty when the "
                    "worker boots? Right now the retry-after-503 path is covered, but not "
                    "the 'cold start, nothing to drain' path — easy regression to introduce "
                    "and a one-liner to test."
                ),
            ),
        ),
        _t("c1", "posted comment to PR #412 (id: 9821447)"),
        _a("Comment posted on #412 asking for the empty-queue startup test."),
    ]


def s38() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Read papers/lora_2021.pdf and write a one-paragraph summary to notes/papers/lora.md."),
        _a(
            "Reading the LoRA paper.",
            _tc("c0", "pdf_read", file_path="papers/lora_2021.pdf", pages="1-2"),
        ),
        _t(
            "c0",
            "[Page 1] LoRA: Low-Rank Adaptation of Large Language Models. Hu et al., 2021.\n"
            "Abstract. We propose Low-Rank Adaptation (LoRA), which freezes pre-trained\n"
            "model weights and injects trainable rank-decomposition matrices into each\n"
            "layer of the Transformer architecture, greatly reducing the number of\n"
            "trainable parameters for downstream tasks. Compared to full fine-tuning,\n"
            "LoRA reduces trainable parameters by 10,000x and GPU memory by 3x while\n"
            "matching or exceeding full-FT quality on RoBERTa, DeBERTa, GPT-2, and GPT-3.\n",
        ),
        _a(
            "Writing the summary.",
            _tc(
                "c1",
                "file_write",
                file_path="notes/papers/lora.md",
                content=(
                    "# LoRA — Low-Rank Adaptation of LLMs (Hu et al., 2021)\n\n"
                    "LoRA freezes the pretrained transformer weights and injects two small "
                    "trainable rank-decomposition matrices (A, B) into each attention "
                    "projection. Only A and B are updated during fine-tuning, so trainable "
                    "params drop ~10,000x and GPU memory ~3x vs. full fine-tuning, while "
                    "downstream quality on RoBERTa / DeBERTa / GPT-2 / GPT-3 matches or "
                    "exceeds the full-FT baseline. The rank `r` is the main knob — typical "
                    "values are 4-16. Adapter weights can be merged back into the base "
                    "model at inference time for zero added latency.\n"
                ),
            ),
        ),
        _t("c1", "wrote notes/papers/lora.md (4 lines, 681 bytes)"),
        _a(
            "Summary saved to `notes/papers/lora.md`. The 10,000x param reduction and "
            "merge-at-inference property are the bits worth remembering."
        ),
    ]


def s39() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Have a researcher subagent figure out whether asyncio.TaskGroup or "
            "anyio.create_task_group fits this codebase better, then update "
            "app/workers/runner.py to use the chosen one."
        ),
        _a(
            "Delegating the comparison to a subagent.",
            _tc(
                "c0",
                "spawn_agent",
                agent="researcher",
                task=(
                    "Compare asyncio.TaskGroup (PEP 654, Py3.11+) and anyio.create_task_group "
                    "for a FastAPI codebase already using anyio under the hood. Cover: "
                    "exception semantics, cancellation, library compatibility. Recommend one."
                ),
            ),
        ),
        _t(
            "c0",
            "[researcher subagent finished]\n"
            "Recommendation: anyio.create_task_group.\n"
            "Reasoning:\n"
            " - FastAPI already runs on anyio, so adding asyncio-specific primitives is\n"
            "   inconsistent and would mix the two cancellation models.\n"
            " - anyio task groups have the same structured-concurrency semantics as\n"
            "   asyncio.TaskGroup (ExceptionGroup on aggregate failure) but also work\n"
            "   under trio if the project ever switches.\n"
            " - asyncio.TaskGroup requires Python 3.11+; anyio works on 3.8+.\n"
            "Sources: anyio.readthedocs.io/en/stable/tasks.html, peps.python.org/pep-0654/.\n",
        ),
        _a(
            "Subagent picked **anyio.create_task_group** for consistency with FastAPI's "
            "anyio backend. Updating `runner.py`.",
            _tc(
                "c1",
                "file_edit",
                file_path="app/workers/runner.py",
                old_string=(
                    "import asyncio\n\n\n"
                    "async def run_jobs(jobs: list[Job]) -> None:\n"
                    "    async with asyncio.TaskGroup() as tg:\n"
                    "        for job in jobs:\n"
                    "            tg.create_task(_run_one(job))"
                ),
                new_string=(
                    "import anyio\n\n\n"
                    "async def run_jobs(jobs: list[Job]) -> None:\n"
                    "    async with anyio.create_task_group() as tg:\n"
                    "        for job in jobs:\n"
                    "            tg.start_soon(_run_one, job)"
                ),
            ),
        ),
        _t("c1", "edited app/workers/runner.py: 1 replacement"),
        _a(
            "Switched to `anyio.create_task_group()` and the matching `tg.start_soon(...)` "
            "API (note: anyio takes the callable + args, not an awaitable like asyncio). "
            "Aggregate failures still surface as `ExceptionGroup`."
        ),
    ]


def s40() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Open mockups/login_v2.png and update web/src/styles/login.css to match the "
            "header background color shown in the mockup."
        ),
        _a(
            "Inspecting the mockup.",
            _tc("c0", "image_read", file_path="mockups/login_v2.png"),
        ),
        _t(
            "c0",
            "[image: 1440x900 PNG]\n"
            "Description (vision model output):\n"
            "Login screen redesign. Header bar spans the full width with a dark navy "
            "background — looks like #0F172A (slate-900). White wordmark on the left, no "
            "icons on the right. Body below uses a subtle gray (#F8FAFC) background, "
            "centered card with email/password fields and a primary indigo button.\n",
        ),
        _a(
            "Header background should be `#0F172A` (Tailwind slate-900). Updating the CSS.",
            _tc(
                "c1",
                "file_edit",
                file_path="web/src/styles/login.css",
                old_string=".login-header {\n  background: #1E293B;\n}",
                new_string=".login-header {\n  background: #0F172A;\n}",
            ),
        ),
        _t("c1", "edited web/src/styles/login.css: 1 replacement"),
        _a(
            "`.login-header` now uses `#0F172A` to match the mockup. The body background "
            "(`#F8FAFC`) and indigo CTA color also visible in the mockup look like they're "
            "already correct, but check side-by-side after a hot-reload."
        ),
    ]


def s41() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Add a task for migrating the legacy webhook handler off requests, then list "
            "everything that's pending."
        ),
        _a(
            "Adding the new task.",
            _tc(
                "c0",
                "tasks",
                action="add",
                title="Migrate legacy webhook handler from requests to httpx",
                priority="normal",
            ),
        ),
        _t(
            "c0",
            "added task #11\n"
            "  title: Migrate legacy webhook handler from requests to httpx\n"
            "  status: pending  priority: normal\n",
        ),
        _a(
            "Done — task #11. Now listing all pending tasks.",
            _tc("c1", "tasks", action="list", status="pending"),
        ),
        _t(
            "c1",
            "#7   Write integration tests for payment retry queue        normal\n"
            "#9   Add Sentry breadcrumbs to billing service               low\n"
            "#11  Migrate legacy webhook handler from requests to httpx   normal\n"
            "(3 pending)",
        ),
        _a(
            "Three pending tasks: #7 (integration tests), #9 (Sentry breadcrumbs), and the "
            "new #11 (webhook migration). Want me to bump #11's priority?"
        ),
    ]


def s42() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "In notebooks/training.ipynb: insert a new markdown cell at position 2 with the "
            "title '## Hyperparameters', then update cell 5's `lr=1e-4` to `lr=2e-5`."
        ),
        _a(
            "Inserting the markdown cell first.",
            _tc(
                "c0",
                "notebook_edit",
                notebook_path="notebooks/training.ipynb",
                cell_id="2",
                cell_type="markdown",
                edit_mode="insert",
                new_source="## Hyperparameters",
            ),
        ),
        _t("c0", "inserted markdown cell at position 2 in notebooks/training.ipynb"),
        _a(
            "Now updating the learning rate in cell 5.",
            _tc(
                "c1",
                "notebook_edit",
                notebook_path="notebooks/training.ipynb",
                cell_id="5",
                new_source=(
                    "trainer = Trainer(\n"
                    "    model=model,\n"
                    "    args=TrainingArguments(\n"
                    "        output_dir='out/',\n"
                    "        learning_rate=2e-5,\n"
                    "        per_device_train_batch_size=8,\n"
                    "        num_train_epochs=3,\n"
                    "    ),\n"
                    "    train_dataset=train_ds,\n"
                    ")"
                ),
            ),
        ),
        _t("c1", "edited cell 5 (code) in notebooks/training.ipynb"),
        _a(
            "Both edits applied: new `## Hyperparameters` markdown header at cell 2, and "
            "cell 5 now uses `learning_rate=2e-5`. Re-run from cell 2 onward to make sure "
            "nothing depends on the old value."
        ),
    ]


# ---------------------------------------------------------------------------
# Samples 43-45: no-tool conversational
# ---------------------------------------------------------------------------


def s43() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "I have a list of users called `active_users`. What's a good name for the "
            "filtered subset of users who haven't logged in for 30 days?"
        ),
        _a(
            "A few options ranked by how clearly they convey intent:\n"
            "1. **`stale_users`** — short, common idiom for 'active but dormant'.\n"
            "2. **`inactive_for_30d`** — explicit about the threshold; good if 30 days is "
            "load-bearing in the surrounding code.\n"
            "3. **`reengagement_candidates`** — domain-flavored; great if you're going to "
            "act on the list (e.g., send a re-engagement email).\n\n"
            "I'd pick `stale_users` for general use, or `reengagement_candidates` if the "
            "next thing you do with the list is in fact a marketing email — names that "
            "preview the use site age well."
        ),
    ]


def s44() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Walk me through the mental model of Python's async/await."),
        _a(
            "Three concepts make the whole thing click:\n\n"
            "**1. Coroutines are pause-able functions.**\n"
            "Calling an `async def` function returns a coroutine *object* — it hasn't "
            "started running. You hand it to an event loop (via `await` or `asyncio.run`) "
            "to actually execute it.\n\n"
            "**2. `await` is the cooperative yield point.**\n"
            "When code hits `await some_io()`, the coroutine suspends and gives control "
            "back to the event loop, which is free to run other ready coroutines. When the "
            "I/O completes, the loop resumes the original coroutine right where it left off.\n\n"
            "**3. The event loop is single-threaded.**\n"
            "Only one coroutine runs at a time. 'Concurrency' here means *interleaving* "
            "lots of waiting tasks, not parallel CPU work. CPU-bound work in a coroutine "
            "blocks every other coroutine until it returns — that's why you offload heavy "
            "compute to a thread pool (`asyncio.to_thread`) or process pool.\n\n"
            "If you remember 'coroutines are paused functions, `await` is the pause, the "
            "loop interleaves the paused ones', the rest of the API (TaskGroups, gather, "
            "queues) is just plumbing on top."
        ),
    ]


def s45() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("What's the practical difference between bf16 and fp16 for training?"),
        _a(
            "Both are 16-bit, but they trade precision for range very differently:\n\n"
            "| | fp16 | bf16 |\n"
            "|---|---|---|\n"
            "| Sign bits | 1 | 1 |\n"
            "| Exponent bits | 5 | **8** |\n"
            "| Mantissa bits | 10 | 7 |\n"
            "| Dynamic range | ~6e-5 to 65504 | same as fp32 |\n\n"
            "**fp16** has more mantissa precision but a tiny range — large gradients "
            "overflow, tiny gradients underflow. That's why fp16 training needs a *gradient "
            "scaler*: multiply the loss by ~2¹⁶ before backprop, divide back before the "
            "optimizer step.\n\n"
            "**bf16** has fp32's exponent (same range), so no scaler needed. The cost is "
            "less mantissa precision, which usually doesn't matter for deep nets but can "
            "show up in things like loss curves drifting more.\n\n"
            "**Practical rule**: on any Ampere or newer GPU (A100, H100, RTX 30/40/50), "
            "use bf16. The training is more numerically stable and you skip the scaler "
            "machinery entirely. fp16 only makes sense on older hardware (V100, T4) that "
            "lacks bf16 support."
        ),
    ]


# ---------------------------------------------------------------------------
# Samples 46-50: error-recovery (tool returns an error, assistant adapts)
# ---------------------------------------------------------------------------


def s46() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Read the changelog so I can see what shipped in 1.4.0."),
        _a(
            "Reading the changelog.",
            _tc("c0", "file_read", file_path="CHANGELOG"),
        ),
        _t("c0", "Error: file not found: CHANGELOG"),
        _a(
            "No bare `CHANGELOG` — convention here is probably `CHANGELOG.md`. Trying that.",
            _tc("c1", "file_read", file_path="CHANGELOG.md"),
        ),
        _t(
            "c1",
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "## [1.4.0] - 2026-04-10\n"
            "### Added\n"
            "- Payment retry queue with exponential backoff (#412)\n"
            "- Sentry breadcrumbs in billing service (#398)\n"
            "### Fixed\n"
            "- Off-by-one in invoice line totals (#404)\n"
            "### Changed\n"
            "- Bumped pydantic to 2.10.x\n",
        ),
        _a(
            "1.4.0 (released 2026-04-10) shipped: payment retry queue (#412), Sentry "
            "breadcrumbs in billing (#398), invoice off-by-one fix (#404), and a pydantic "
            "2.10 bump. The `[Unreleased]` section is currently empty."
        ),
    ]


def s47() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Show me which version of jq is on this box."),
        _a(
            "Checking jq's version.",
            _tc("c0", "shell", command="jq --version"),
        ),
        _t(
            "c0",
            "exit_code: 127\nstderr: bash: jq: command not found\n",
        ),
        _a(
            "`jq` isn't installed. I won't install it without your say-so, but the install "
            "would be `sudo apt-get install -y jq` on Debian/Ubuntu or `brew install jq` "
            "on macOS. Let me see if there's a Python equivalent we already have so you "
            "don't need it.",
            _tc("c1", "shell", command="python -c 'import json; print(json.__name__)'"),
        ),
        _t("c1", "json\n"),
        _a(
            "Python's stdlib `json` is available — for one-off pipelines you can use "
            "`python -m json.tool` (pretty-print) or pipe through `python -c 'import json,sys; "
            'print(json.load(sys.stdin)["key"])\'`. If you need real query power, install '
            "jq with the apt/brew command above."
        ),
    ]


def s48() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Type-check app/services/email.py and fix anything mypy flags."),
        _a(
            "Running mypy.",
            _tc("c0", "verify", file_path="app/services/email.py", check_type="mypy"),
        ),
        _t(
            "c0",
            'app/services/email.py:42: error: Argument 1 to "send" of "SmtpClient" '
            'has incompatible type "list[str]"; expected "str"  [arg-type]\n'
            "Found 1 error in 1 file (checked 1 source file)\n",
        ),
        _a(
            "L42 is passing a `list[str]` of recipients into `SmtpClient.send` which "
            "wants a single `str`. Looking at how recipients are handled elsewhere, the "
            "fix is to join them with commas (RFC 5322 envelope-To list).",
            _tc(
                "c1",
                "file_edit",
                file_path="app/services/email.py",
                old_string="    client.send(recipients, subject, body)",
                new_string='    client.send(", ".join(recipients), subject, body)',
            ),
        ),
        _t("c1", "edited app/services/email.py: 1 replacement"),
        _a(
            "Re-running mypy to confirm.",
            _tc("c2", "verify", file_path="app/services/email.py", check_type="mypy"),
        ),
        _t("c2", "Success: no issues found in 1 source file\n"),
        _a(
            "Clean. Recipients are now comma-joined before being passed to "
            "`SmtpClient.send`. Worth checking the `SmtpClient` API though — if it has a "
            "list-accepting overload, that's cleaner than string-joining."
        ),
    ]


def s49() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u("Run the cart tests; if they fail, find the offending code and fix it."),
        _a(
            "Running the cart tests.",
            _tc("c0", "test_runner", path="tests/services/test_cart.py"),
        ),
        _t(
            "c0",
            "============================= test session starts =============================\n"
            "collected 9 items\n\n"
            "tests/services/test_cart.py ........F                                    [100%]\n\n"
            "================================== FAILURES ===================================\n"
            "____________________ test_cart_total_with_zero_quantity _______________________\n"
            "    def test_cart_total_with_zero_quantity():\n"
            "        cart = Cart()\n"
            "        cart.add(Item(price=Decimal('5.00'), quantity=0))\n"
            ">       assert cart.total() == Decimal('0.00')\n"
            "E       decimal.InvalidOperation: [<class 'decimal.DivisionByZero'>]\n"
            "app/services/cart.py:24: InvalidOperation\n\n"
            "========================= 1 failed, 8 passed in 0.42s =========================\n",
        ),
        _a(
            "One failure: `cart.total()` divides by zero when an item has quantity=0. "
            "Finding the relevant code.",
            _tc("c1", "grep_search", pattern=r"def total", path="app/services/cart.py"),
        ),
        _t(
            "c1",
            "app/services/cart.py:20:    def total(self) -> Decimal:\n"
            "app/services/cart.py:21:        if not self.items:\n"
            "app/services/cart.py:22:            return Decimal('0.00')\n"
            "app/services/cart.py:23:        subtotal = sum("
            "(i.price * i.quantity for i in self.items), Decimal('0'))\n"
            "app/services/cart.py:24:        return subtotal / "
            "sum(i.quantity for i in self.items)\n",
        ),
        _a(
            "Found it. L24 divides by total quantity to compute… an *average* price, not "
            "the cart total. The function name and the test both expect the simple "
            "subtotal. Fixing.",
            _tc(
                "c2",
                "file_edit",
                file_path="app/services/cart.py",
                old_string=(
                    "        subtotal = sum("
                    "(i.price * i.quantity for i in self.items), Decimal('0'))\n"
                    "        return subtotal / "
                    "sum(i.quantity for i in self.items)"
                ),
                new_string=(
                    "        return sum((i.price * i.quantity for i in self.items), Decimal('0'))"
                ),
            ),
        ),
        _t("c2", "edited app/services/cart.py: 1 replacement"),
        _a(
            "`total()` now returns the sum of `price * quantity` directly. The bug looks "
            "like a copy/paste from a separate `average_price()` helper. Re-run the cart "
            "tests to confirm — should be 9/9 green now."
        ),
    ]


def s50() -> list[dict[str, Any]]:
    return [
        _sys(),
        _u(
            "Apply this hunk:\n"
            "--- a/app/api/health.py\n"
            "+++ b/app/api/health.py\n"
            "@@ -8,3 +8,4 @@\n"
            " def health():\n"
            "     return {'ok': True}\n"
            "+    # touched by deploy script"
        ),
        _a(
            "Applying the patch.",
            _tc(
                "c0",
                "diff_apply",
                diff=(
                    "--- a/app/api/health.py\n"
                    "+++ b/app/api/health.py\n"
                    "@@ -8,3 +8,4 @@\n"
                    " def health():\n"
                    "     return {'ok': True}\n"
                    "+    # touched by deploy script\n"
                ),
            ),
        ),
        _t(
            "c0",
            "Error: failed to apply hunk @@ -8,3 +8,4 @@ to app/api/health.py: "
            "context mismatch — file has different content at line 8.\n",
        ),
        _a(
            "Hunk header offsets are stale. Reading the file to find the real line numbers.",
            _tc("c1", "file_read", file_path="app/api/health.py"),
        ),
        _t(
            "c1",
            "from fastapi import APIRouter\n\n"
            "router = APIRouter()\n\n\n"
            "@router.get('/health')\n"
            "def health():\n"
            "    return {'ok': True}\n",
        ),
        _a(
            "`health()` is at line 6, returning at line 7 — not line 8 as the original "
            "patch claimed. Rewriting the hunk with corrected offsets.",
            _tc(
                "c2",
                "diff_apply",
                diff=(
                    "--- a/app/api/health.py\n"
                    "+++ b/app/api/health.py\n"
                    "@@ -6,3 +6,4 @@\n"
                    " @router.get('/health')\n"
                    " def health():\n"
                    "     return {'ok': True}\n"
                    "+    # touched by deploy script\n"
                ),
            ),
        ),
        _t("c2", "applied 1 hunk to app/api/health.py (1+, 0-)"),
        _a(
            "Patch applied with corrected hunk offsets. The original diff was generated "
            "against an older version of the file — when patches drift like this, easiest "
            "fix is to read the file, find the real anchor lines, and rewrite the `@@` "
            "header."
        ),
    ]


# ---------------------------------------------------------------------------
# Assembly + CLI
# ---------------------------------------------------------------------------


_BUILDERS = (
    s01,
    s02,
    s03,
    s04,
    s05,
    s06,
    s07,
    s08,
    s09,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    s16,
    s17,
    s18,
    s19,
    s20,
    s21,
    s22,
    s23,
    s24,
    s25,
    s26,
    s27,
    s28,
    s29,
    s30,
    s31,
    s32,
    s33,
    s34,
    s35,
    s36,
    s37,
    s38,
    s39,
    s40,
    s41,
    s42,
    s43,
    s44,
    s45,
    s46,
    s47,
    s48,
    s49,
    s50,
)


def build_anchor_samples() -> list[dict[str, Any]]:
    """Build the 50 ``{messages, tools}`` records.

    The tools schema is loaded once and shared across every record (they all
    advertise the same 21 canonical tools).
    """
    tools = get_tool_schemas()
    return [{"messages": fn(), "tools": tools} for fn in _BUILDERS]


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-tool usage counts and category counts for sanity-logging."""
    tool_usage: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for rec in samples:
        n_calls = 0
        for msg in rec["messages"]:
            for tc in msg.get("tool_calls") or []:
                name = tc.get("function", {}).get("name")
                if name:
                    tool_usage[name] += 1
                    n_calls += 1
        if n_calls == 0:
            category_counts["no_tool"] += 1
        elif n_calls == 1:
            category_counts["single_tool"] += 1
        else:
            category_counts["multi_turn"] += 1
    return {
        "samples": len(samples),
        "tool_usage": dict(tool_usage),
        "categories": dict(category_counts),
        "missing_tools": [t for t in ALL_TOOLS if tool_usage[t] == 0],
    }


def write_anchor_jsonl(output_path: Path) -> dict[str, Any]:
    samples = build_anchor_samples()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for rec in samples:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return _summarize(samples)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Emit the Opus-authored anchor JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase_a1/data/anchor_opus_50.jsonl"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    summary = write_anchor_jsonl(args.output)
    logger.info("wrote %d anchor samples to %s", summary["samples"], args.output)
    logger.info("category mix: %s", summary["categories"])
    logger.info(
        "tool coverage (sorted): %s",
        dict(sorted(summary["tool_usage"].items(), key=lambda kv: -kv[1])),
    )
    if summary["missing_tools"]:
        logger.error("MISSING tools (no coverage): %s", summary["missing_tools"])
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
