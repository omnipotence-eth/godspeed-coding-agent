"""Conversation compaction — summarize history when approaching context limit.

Follows Anthropic's context engineering guidance:
- Preserve architectural decisions
- Preserve file paths modified
- Preserve unresolved issues
- Discard redundant tool outputs
"""

from __future__ import annotations

import logging

from godspeed.agent.conversation import Conversation
from godspeed.llm.client import LLMClient

logger = logging.getLogger(__name__)

COMPACTION_SYSTEM_PROMPT = """\
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


async def compact_if_needed(
    conversation: Conversation,
    llm_client: LLMClient,
) -> bool:
    """Check if compaction is needed and perform it.

    Returns True if compaction was performed.
    """
    if not conversation.is_near_limit:
        return False

    logger.info(
        "Compaction triggered tokens=%d threshold=%d",
        conversation.token_count,
        int(conversation.max_tokens * conversation.compaction_threshold),
    )

    context = conversation.get_compaction_context()
    summary_messages = [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
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
