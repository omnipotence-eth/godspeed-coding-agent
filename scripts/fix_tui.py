"""Fix TUI to be minimal and modern."""

import pathlib

p = pathlib.Path("src/godspeed/tui/output.py")
text = p.read_text(encoding="utf-8")

# Minimal format_status_hud
old = '''def format_status_hud(
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    model: str,
    turns: int,
    budget_usd: float = 0.0,
    max_iterations: int = 0,
    context_pct: float = 0.0,
    permission_mode: str = "",
    preset: str = "",
) -> None:
    """Print a compact one-line session HUD after each completed turn.

    Example rendering:
        | 1,234 in + 567 out (1,801) | ctx 34% | $0.0024 | qwen3.5-397b | 3/50 turns

    Shows context window usage percentage, cost, model with preset tag,
    iteration progress, and active permission mode.
    """
    total_tokens = input_tokens + output_tokens
    tokens_text = styled(f"{input_tokens:,} in + {output_tokens:,} out ({total_tokens:,})", DIM)

    if budget_usd > 0:
        remaining = max(0.0, budget_usd - cost_usd)
        near_limit = remaining < budget_usd * 0.2
        cost_style = WARNING if near_limit else DIM
        cost_text = styled(f"${cost_usd:.4f} / ${budget_usd:.2f}", cost_style)
    else:
        cost_text = styled(f"${cost_usd:.4f}", DIM)

    # Short model label — drop provider prefix for readability
    model_short = model.split("/", 1)[-1] if "/" in model else model
    if preset:
        model_text = styled(f"{model_short} [{preset}]", NEUTRAL)
    else:
        model_text = styled(model_short, NEUTRAL)

    # Turn progress
    if max_iterations > 0:
        turns_text = styled(f"{turns}/{max_iterations} turns", DIM)
    else:
        turns_text = styled(f"{turns} turn{'s' if turns != 1 else ''}", DIM)

    parts = [tokens_text]

    # Context window usage
    if context_pct > 0:
        if context_pct >= 90:
            ctx_style = CTX_CRITICAL
        elif context_pct >= 70:
            ctx_style = CTX_WARN
        else:
            ctx_style = CTX_OK
        ctx_text = styled(f"ctx {context_pct:.0f}%", ctx_style)
        parts.append(ctx_text)

    parts.extend([cost_text, model_text, turns_text])

    # Permission mode indicator
    if permission_mode == "yolo":
        parts.append(styled("YOLO", BOLD_WARNING))
    elif permission_mode == "strict":
        parts.append(styled("strict", WARNING))
    elif permission_mode == "plan":
        parts.append(styled("plan", PRIMARY))

    sep = styled(SEPARATOR_DOT, NEUTRAL)
    console.print(f"  {sep} {' | '.join(parts)}")'''

new = '''def format_status_hud(
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    model: str,
    turns: int,
    budget_usd: float = 0.0,
    max_iterations: int = 0,
    context_pct: float = 0.0,
    permission_mode: str = "",
    preset: str = "",
) -> None:
    """Print a minimal one-line session HUD after each completed turn."""
    parts: list[str] = []

    # Token count
    total = input_tokens + output_tokens
    parts.append(styled(f"{total:,} tokens", DIM))

    # Context window
    if context_pct > 0:
        if context_pct >= 90:
            ctx_style = CTX_CRITICAL
        elif context_pct >= 70:
            ctx_style = CTX_WARN
        else:
            ctx_style = CTX_OK
        parts.append(styled(f"ctx {context_pct:.0f}%", ctx_style))

    # Cost
    if budget_usd > 0:
        remaining = max(0.0, budget_usd - cost_usd)
        near_limit = remaining < budget_usd * 0.2
        cost_style = WARNING if near_limit else DIM
        parts.append(styled(f"${cost_usd:.4f} / ${budget_usd:.2f}", cost_style))
    elif cost_usd > 0:
        parts.append(styled(f"${cost_usd:.4f}", DIM))

    # Model (short)
    model_short = model.split("/", 1)[-1] if "/" in model else model
    if preset:
        model_short = f"{model_short} [{preset}]"
    parts.append(styled(model_short, NEUTRAL))

    # Turn count
    if max_iterations > 0:
        parts.append(styled(f"{turns}/{max_iterations}", DIM))
    else:
        parts.append(styled(f"turn {turns}", DIM))

    # Permission mode
    if permission_mode == "yolo":
        parts.append(styled("YOLO", BOLD_WARNING))
    elif permission_mode == "strict":
        parts.append(styled("strict", WARNING))
    elif permission_mode == "plan":
        parts.append(styled("plan", PRIMARY))

    sep = styled(SEPARATOR_DOT, NEUTRAL)
    console.print(f"  {' | '.join(parts)}")'''

if old in text:
    text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    print("Replaced format_status_hud")
else:
    print("NOT FOUND")
    # Debug: show what's around format_status_hud
    idx = text.find("def format_status_hud")
    if idx >= 0:
        print(repr(text[idx : idx + 200]))
