"""Conversation compaction — summarize history when approaching context limit.

Follows Anthropic's context engineering guidance:
- Preserve architectural decisions
- Preserve file paths modified
- Preserve unresolved issues
- Discard redundant tool outputs

Model-aware: small-context models get aggressive compaction prompts,
frontier models get detailed preservation prompts.
"""

from __future__ import annotations

import logging

from godspeed.agent.conversation import Conversation
from godspeed.config import get_model_context_window
from godspeed.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Threshold for "small" vs "large" vs "ultra-large" context models
SMALL_CONTEXT_THRESHOLD = 32_768
LARGE_CONTEXT_THRESHOLD = 100_000
ULTRA_LARGE_CONTEXT_THRESHOLD = 500_000  # Claude Opus 4.6, GPT-5, Gemini 2

COMPACTION_PROMPT_SMALL = """\
Aggressively summarize this coding agent conversation.

Keep ONLY:
- The current task and its status
- File paths modified (just the paths, not contents)
- The last error encountered (if any)
- Key user instructions

Discard everything else. Be extremely brief — under 500 words.
The summary replaces the full history, so omit anything non-essential.
"""

COMPACTION_PROMPT_MEDIUM = """\
Summarize the following conversation between a user and a coding agent.

You MUST preserve:
- Architectural decisions made
- File paths that were modified or created
- Unresolved issues or errors
- The current task state and what was accomplished
- Any user preferences or instructions given

You MUST discard:
- Redundant tool outputs (e.g., full file contents already edited)
- Repeated attempts that were superseded
- Verbose error tracebacks (keep the error message, not the full trace)

Be concise but complete. The summary will replace the conversation history,
so anything not in the summary is lost forever.
"""

COMPACTION_PROMPT_LARGE = """\
Summarize this coding agent conversation, preserving maximum context.

You MUST preserve:
- Architectural decisions and rationale
- All file paths modified or created, with a brief note on each change
- Unresolved issues, errors, and their context
- Current task state — what was accomplished, what remains
- User preferences, instructions, and corrections given
- Last 3 tool results (summarized, not verbatim)
- Key code patterns or conventions discovered

You may discard:
- Verbatim file contents that were read but not modified
- Redundant intermediate tool outputs
- Full stack traces (keep the error message and location)

Be thorough. With a large context window, it is better to preserve
too much than too little.
"""

COMPACTION_PROMPT_ULTRA_LARGE = """\
You are summarizing a coding agent conversation with a 1M+ token context window.

Preserve EVERYTHING important. With this large context, be exhaustive:
- All architectural decisions, file changes, errors, and task state
- Complete file path history with change notes
- User instructions and corrections verbatim when possible
- All unresolved issues with full context
- Last 10 tool results (summarized)
- Code conventions and patterns discovered
- Session goals and progress toward them

Do NOT discard anything that might be relevant. Better to preserve
extra context than lose important information. This summary will allow
a fresh agent to continue exactly where this one left off.
"""

# Keep the old name as an alias for backwards compatibility in agent/loop.py
COMPACTION_SYSTEM_PROMPT = COMPACTION_PROMPT_MEDIUM


def get_compaction_prompt(model: str) -> str:
    """Select the compaction prompt based on model context window size.

    - Small (≤32K): aggressive — keep only essentials
    - Medium (32K-100K): balanced - standard preservation
    - Large (>100K, ≤500K): detailed — preserve maximum context
    - Ultra-Large (>500K, 1M+): exhaustive — preserve everything
    """
    context_size = get_model_context_window(model)

    if context_size <= SMALL_CONTEXT_THRESHOLD:
        logger.debug("Using small compaction prompt model=%s context=%d", model, context_size)
        return COMPACTION_PROMPT_SMALL
    if context_size > ULTRA_LARGE_CONTEXT_THRESHOLD:
        logger.debug("Using ultra-large compaction prompt model=%s context=%d", model, context_size)
        return COMPACTION_PROMPT_ULTRA_LARGE
    if context_size > LARGE_CONTEXT_THRESHOLD:
        logger.debug("Using large compaction prompt model=%s context=%d", model, context_size)
        return COMPACTION_PROMPT_LARGE

    logger.debug("Using medium compaction prompt model=%s context=%d", model, context_size)
    return COMPACTION_PROMPT_MEDIUM


async def compact_if_needed(
    conversation: Conversation,
    llm_client: LLMClient,
    model: str | None = None,
) -> bool:
    """Check if compaction is needed and perform it.

    Args:
        conversation: The conversation to compact.
        llm_client: LLM client for the summarization call.
        model: Model name for selecting the compaction prompt.
            Falls back to llm_client.model if not provided.

    Returns True if compaction was performed.
    """
    if not conversation.is_near_limit:
        return False

    model_name = model or getattr(llm_client, "model", "")
    prompt = get_compaction_prompt(model_name) if model_name else COMPACTION_PROMPT_MEDIUM

    logger.info(
        "Compaction triggered tokens=%d threshold=%d model=%s",
        conversation.token_count,
        int(conversation.max_tokens * conversation.compaction_threshold),
        model_name,
    )

    context = conversation.get_compaction_context()
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    try:
        response = await llm_client.chat(messages=summary_messages)
        conversation.compact(response.content)
        logger.info("Compaction complete new_tokens=%d", conversation.token_count)
        return True
    except Exception as exc:
        logger.error("Compaction failed: %s", exc, exc_info=True)
        return False
