"""Minimal TUI rewrite — strip welcome/summary bloat."""

import pathlib

p = pathlib.Path("src/godspeed/tui/output.py")
text = p.read_text(encoding="utf-8")

# ---- 1. Replace format_status_hud (minimal) ----
old_hud = text[text.find("def format_status_hud(") :]
next_def = old_hud.find("\ndef ")
if next_def >= 0:
    old_hud = old_hud[:next_def]

new_hud = """def format_status_hud(
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
    \"\"\"Print a minimal one-line session HUD after each completed turn.\"\"\"
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
    console.print(f"  {' | '.join(parts)}")

"""

if old_hud in text:
    text = text.replace(old_hud, new_hud, 1)
    print("Replaced format_status_hud")
else:
    print(f"Could not find format_status_hud. First 100 chars: {old_hud[:100]!r}")

# ---- 2. Replace format_welcome (minimal) ----
old_welcome = text[text.find("def format_welcome(") :]
next_def = old_welcome.find("\ndef ")
if next_def >= 0:
    old_welcome = old_welcome[:next_def]

new_welcome = """def format_welcome(
    model: str,
    project_dir: str,
    tools: list[str] | None = None,
    deny_rules: list[str] | None = None,
    audit_enabled: bool = True,
    permission_mode: str = "normal",
    preset: str = "",
) -> None:
    \"\"\"Display minimal welcome line with model and mode only.\"\"\"
    from godspeed import __version__

    console.print()

    # Branded header — one line
    header = f"  {PROMPT_ICON} {brand(__version__)}"
    console.print(header)

    # Model + mode on one line
    model_short = model.split("/", 1)[-1] if "/" in model else model
    if preset:
        model_short = f"{model_short} [{preset}]"
    mode_map = {"normal": "", "strict": "strict", "yolo": "YOLO", "plan": "plan"}
    mode_str = mode_map.get(permission_mode, permission_mode)
    line = f"  model: {styled(model_short, NEUTRAL)}"
    if mode_str:
        line += f"  {styled(f'[{mode_str}]', DIM)}"
    console.print(line)

    # Tools count (minimal)
    if tools:
        console.print(f"  {len(tools)} tools  {styled(f'(/help)', DIM)}")
    console.print()

"""

if old_welcome in text:
    text = text.replace(old_welcome, new_welcome, 1)
    print("Replaced format_welcome")
else:
    print(f"Could not find format_welcome. First 100 chars: {old_welcome[:100]!r}")

# ---- 3. Replace format_session_summary (minimal) ----
old_summary = text[text.find("def format_session_summary(") :]
next_def = old_summary.find("\ndef ")
if next_def >= 0:
    old_summary = old_summary[:next_def]

new_summary = """def format_session_summary(
    duration_secs: float,
    input_tokens: int,
    output_tokens: int,
    cost: float | None = None,
    tool_calls: int = 0,
    tool_errors: int = 0,
    tool_denied: int = 0,
    model: str = "",
    session_id: str = "",
) -> None:
    \"\"\"Display minimal session summary on quit.\"\"\"
    console.print()

    # Duration + tokens on one line
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)
    dur = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    total = input_tokens + output_tokens
    line = f"  {styled(dur, DIM)}  {styled(f'{total:,} tokens', NEUTRAL)}"
    if cost is not None and cost > 0:
        line += f"  {styled(f'${cost:.4f}', DIM)}"
    console.print(line)

    # Tool summary (one line)
    if tool_calls > 0:
        success = tool_calls - tool_errors - tool_denied
        parts = [f"{success} {MARKER_SUCCESS}"]
        if tool_errors > 0:
            parts.append(f"{tool_errors} {MARKER_ERROR}")
        if tool_denied > 0:
            parts.append(f"{tool_denied} denied")
        summary = f" {' | '.join(parts)}"
        console.print(f"  {tool_calls} calls  {styled(f'({summary})', DIM)}")

    # Compact sign-off
    console.print(f"  {styled(PROMPT_ICON, BOLD_PRIMARY)} {styled('Godspeed', BOLD_PRIMARY)}")

"""

if old_summary in text:
    text = text.replace(old_summary, new_summary, 1)
    print("Replaced format_session_summary")
else:
    print(f"Could not find format_session_summary. First 100 chars: {old_summary[:100]!r}")

p.write_text(text, encoding="utf-8")
print("Done writing output.py")
