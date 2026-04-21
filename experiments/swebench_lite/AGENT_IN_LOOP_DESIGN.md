# Agent-in-loop Docker — design

**Goal:** let the agent run the failing SWE-Bench test inside Docker, mid-session, before declaring its fix complete. Directly attacks the "right file, wrong fix" failure mode (~40% of unresolved instances in the 2026-04-20 analysis).

## Current flow (no in-loop)

```
run.py --> subprocess "godspeed run" --> agent edits files --> agent stops --> run.py captures git diff
                                                                                        |
                                                           (dev/test: submit to sb-cli)
                                                           (optional: --verify-retry loops once)
```

The agent never sees whether its fix works. `--verify-retry` bolts on one retry *after* the agent is done, but that's post-hoc: the agent is already in its follow-up turn, reasoning from failure output rather than iterating naturally.

## Proposed flow (in-loop)

```
run.py --> agent_loop() (direct import) --> registry has swebench_verify_patch tool
                                               |
                                        agent iterates: edit -> verify_patch -> read test output -> edit
                                               |
                                     agent stops when test passes (or max-iter)
                                               |
                                         run.py captures final git diff
```

Agent gains a new tool `swebench_verify_patch` that:
1. Captures the current workspace diff via `git diff`.
2. Invokes `verify_patch.py`'s WSL-Docker path, passing the diff.
3. Returns `resolved: bool, test_output: tail-1000-chars`.

Cost per call: ~60–90s (full swebench harness container spinup per invocation). Agents will call it 3–8x per instance. Total wall-clock per instance ≈ 5–15 min, up from ~2–5 min today. Budget acceptable because we reclaim ≥10–15% of instances.

## Components

### 1. Custom tool — `experiments/swebench_lite/docker_test_tool.py`

```python
class SWEBenchVerifyTool(Tool):
    name = "swebench_verify_patch"
    description = "Run the SWE-Bench test harness on the current working tree..."

    def __init__(self, instance_id: str, model_name: str, workdir: Path):
        self.instance_id = instance_id
        self.model_name = model_name
        self.workdir = workdir

    async def execute(self, arguments, context):
        diff = run_git_diff(context.cwd)  # capture current state
        resolved, test_output = verify_patch(
            instance_id=self.instance_id,
            model_name=self.model_name,
            model_patch=diff,
            workdir=self.workdir,
        )
        return ToolResult.success(
            f"resolved={resolved}\n\n{test_output[-1000:]}"
        )
```

### 2. New runner entrypoint — `experiments/swebench_lite/run_in_loop.py`

Replicates `run.py`'s per-instance setup but uses `agent_loop()` directly instead of CLI subprocess. Pseudocode:

```python
from godspeed.agent.loop import agent_loop
from godspeed.tools.registry import default_tool_registry  # or manual build

for row in instances:
    workspace = prepare_repo(...)
    registry = default_tool_registry()
    registry.register(SWEBenchVerifyTool(row["instance_id"], model, workdir))
    conversation = Conversation(system_prompt=PROMPT)
    await agent_loop(
        user_input=prompt,
        conversation=conversation,
        llm_client=build_llm_client(model),
        tool_registry=registry,
        tool_context=ToolContext(cwd=workspace),
        max_iterations=40,
    )
    patch = capture_diff(workspace)
    write_prediction(row["instance_id"], patch)
```

### 3. Opt-in flag on existing runner

`run.py` gains `--agent-in-loop` flag. When set, dispatches to `run_in_loop.py` internals instead of CLI subprocess path. Preserves backward compat.

## Risks / unknowns

- **ToolRegistry factory**: need to confirm godspeed exposes a "build default registry" function, or we reconstruct it manually.
- **LLMClient factory from CLI model string**: `nvidia_nim/moonshotai/kimi-k2.5` is a LiteLLM string. Need the same resolver Godspeed CLI uses.
- **Per-call container cost**: 60–90s is painful. Possible mitigations (V2):
  - Keep one container alive per instance, agent does `docker exec` per test.
  - Use a lighter test runner (pytest directly against a pre-built env) when we trust the env.
- **Prompt hints**: The system prompt must explicitly tell the agent the tool exists and when to call it. Otherwise models ignore it (especially non-thinking drivers).

## Validation plan

1. **Smoke**: run one dev instance (`sqlfluff-2419`) with Kimi K2.5 + in-loop tool. Confirm the tool is callable, harness runs, output returns to agent.
2. **Single dev run**: 23 instances dev-split, Kimi K2.5. Measure delta vs non-in-loop Kimi baseline (34.8%).
3. **Gate for test**: ≥+5pp delta on dev (≥39.8%) → run test-50. Below that, iterate on prompt + tool return format.

## Not in this commit

This doc + the scaffolding file only. Wiring into `run.py` and the `run_in_loop.py` entrypoint is a follow-up commit requiring:
- User sign-off on the new flag name and semantics
- Tests for the tool class (happy path, empty-diff, harness-failure)
- Decision on whether to share/diverge the system prompt vs today's `DEFAULT_PROMPT_TEMPLATE`
