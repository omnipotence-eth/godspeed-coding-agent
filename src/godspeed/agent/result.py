"""Agent loop result, metrics sink, and exit code contract.

The agent loop is driven both interactively (TUI) and unattended (headless/CI).
Interactive callers only need the final text. Unattended callers need
structured outcomes so an orchestrator can decide what to do next — retry on
LLM error, bail on budget, escalate on permission denial, etc.

`AgentMetrics` is a mutable accumulator the loop populates as it runs.
`ExitCode` is the headless exit-code contract exposed on `godspeed run`.
"""

from __future__ import annotations

import dataclasses
import enum
import time


class ExitCode(enum.IntEnum):
    """Headless exit codes. Documented and stable across minor versions.

    An orchestrator script can switch on the exit code to decide retry
    behavior. Never repurpose an existing code; add new ones only.
    """

    SUCCESS = 0
    TOOL_ERROR = 1
    MAX_ITERATIONS = 2
    BUDGET_EXCEEDED = 3
    LLM_ERROR = 4
    INVALID_INPUT = 5
    TIMEOUT = 6
    INTERRUPTED = 130


class ExitReason(enum.StrEnum):
    """Human-readable counterpart to ExitCode, emitted in JSON output."""

    STOPPED = "stopped"
    TOOL_ERROR = "tool_error"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXCEEDED = "budget_exceeded"
    LLM_ERROR = "llm_error"
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"


EXIT_REASON_TO_CODE: dict[ExitReason, ExitCode] = {
    ExitReason.STOPPED: ExitCode.SUCCESS,
    ExitReason.TOOL_ERROR: ExitCode.TOOL_ERROR,
    ExitReason.MAX_ITERATIONS: ExitCode.MAX_ITERATIONS,
    ExitReason.BUDGET_EXCEEDED: ExitCode.BUDGET_EXCEEDED,
    ExitReason.LLM_ERROR: ExitCode.LLM_ERROR,
    ExitReason.INVALID_INPUT: ExitCode.INVALID_INPUT,
    ExitReason.TIMEOUT: ExitCode.TIMEOUT,
    ExitReason.INTERRUPTED: ExitCode.INTERRUPTED,
}


class AgentCancelled(Exception):
    """Raised inside the agent loop when an external cancel_event is set.

    Distinct from KeyboardInterrupt so that:
      - Callers can catch AgentCancelled specifically without catching
        unrelated KeyboardInterrupts (prompt-toolkit, stdin reads, etc).
      - The loop can unwind cleanly: stop the current streaming call,
        record partial assistant text, finalize metrics with
        ExitReason.INTERRUPTED, and return.
      - The TUI's "first Ctrl+C cancels the turn; second Ctrl+C exits"
        UX works without the signal racing with prompt-toolkit reads.
    """


@dataclasses.dataclass(slots=True)
class ToolCallRecord:
    """One tool invocation, as seen by the loop."""

    name: str
    is_error: bool


@dataclasses.dataclass(slots=True)
class AgentMetrics:
    """Mutable accumulator populated by the agent loop as it runs.

    Interactive callers can ignore this. Headless/CI callers pass an instance
    in and read it after the loop returns to build the JSON result.
    """

    iterations_used: int = 0
    exit_reason: ExitReason = ExitReason.STOPPED
    tool_calls: list[ToolCallRecord] = dataclasses.field(default_factory=list)
    must_fix_injections: int = 0
    start_time: float = dataclasses.field(default_factory=time.monotonic)
    end_time: float | None = None

    def record_tool_call(self, name: str, is_error: bool) -> None:
        self.tool_calls.append(ToolCallRecord(name=name, is_error=is_error))

    def record_must_fix_injection(self) -> None:
        """Increment when the MUST-FIX gate injects a fix-required message.

        Training signal: agents that trigger many MUST-FIX injections are
        less efficient per unit of successful work. GRPO can penalize on
        this counter.
        """
        self.must_fix_injections += 1

    def finalize(self, reason: ExitReason) -> None:
        self.exit_reason = reason
        self.end_time = time.monotonic()

    @property
    def duration_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.monotonic()
        return end - self.start_time

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def tool_error_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_error)

    @property
    def exit_code(self) -> ExitCode:
        return EXIT_REASON_TO_CODE[self.exit_reason]
