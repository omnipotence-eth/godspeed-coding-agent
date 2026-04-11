"""Headless smoke test — verify Godspeed works end-to-end with a real LLM.

Usage:
    uv run python scripts/smoke_test.py [model]

Default model: ollama/qwen3:4b
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()


async def smoke_test(model: str) -> bool:
    """Run a minimal agent loop with a real LLM and real tools."""
    from godspeed.agent.conversation import Conversation
    from godspeed.agent.loop import agent_loop
    from godspeed.llm.client import LLMClient
    from godspeed.security.permissions import PermissionEngine
    from godspeed.tools.base import ToolContext
    from godspeed.tools.file_read import FileReadTool
    from godspeed.tools.file_write import FileWriteTool
    from godspeed.tools.glob_search import GlobSearchTool
    from godspeed.tools.registry import ToolRegistry
    from godspeed.tools.shell import ShellTool

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Seed a file for the agent to find
        (tmp_path / "hello.py").write_text('print("Hello from Godspeed!")\n')

        # Wire up components
        registry = ToolRegistry()
        for tool in [FileReadTool(), FileWriteTool(), GlobSearchTool(), ShellTool()]:
            registry.register(tool)

        risk_levels = {t.name: t.risk_level for t in registry.list_tools()}
        engine = PermissionEngine(
            allow_patterns=["file_read(*)", "file_write(*)", "glob_search(*)", "shell(*)"],
            tool_risk_levels=risk_levels,
        )

        ctx = ToolContext(cwd=tmp_path, session_id="smoke-test", permissions=engine)
        conv = Conversation(
            system_prompt=(
                "You are a coding agent. You have tools to read, write, and search files. "
                "Use tools to complete tasks. Be concise."
            ),
            model=model,
            max_tokens=8000,
        )
        client = LLMClient(model=model, timeout=60)

        console.print(f"\n[bold cyan]Smoke Test[/bold cyan] — model: {model}")
        console.print(f"  Project dir: {tmp_path}")
        console.print(f"  Tools: {[t.name for t in registry.list_tools()]}")
        console.print()

        # Test 1: Ask agent to read a file
        console.print("[bold]Test 1:[/bold] Read hello.py")
        result = await agent_loop(
            "Read the file hello.py and tell me what it prints.",
            conv,
            client,
            registry,
            ctx,
            on_tool_call=lambda name, args: console.print(f"  [dim]> tool: {name}({args})[/dim]"),
            on_assistant_chunk=lambda chunk: print(chunk, end="", flush=True),  # noqa: T201
        )
        console.print(f"\n  [green]Response:[/green] {result[:200]}")

        passed = "hello" in result.lower() or "godspeed" in result.lower()
        if passed:
            console.print("  [bold green]PASS[/bold green]")
        else:
            console.print("  [bold red]FAIL[/bold red] — expected mention of 'hello' or 'godspeed'")

        # Test 2: Ask agent to create a file
        console.print("\n[bold]Test 2:[/bold] Write a new file")
        conv2 = Conversation(
            system_prompt=(
                "You are a coding agent. You have tools to read, write, and search files. "
                "Use tools to complete tasks. Be concise."
            ),
            model=model,
            max_tokens=8000,
        )
        result2 = await agent_loop(
            "Create a file called 'test.txt' with the content 'smoke test passed'.",
            conv2,
            client,
            registry,
            ctx,
            on_tool_call=lambda name, args: console.print(f"  [dim]> tool: {name}({args})[/dim]"),
            on_assistant_chunk=lambda chunk: print(chunk, end="", flush=True),  # noqa: T201
        )
        console.print(f"\n  [green]Response:[/green] {result2[:200]}")

        test_file = tmp_path / "test.txt"
        if test_file.exists():
            content = test_file.read_text()
            console.print(f"  File content: {content!r}")
            passed2 = "smoke" in content.lower() or "passed" in content.lower()
        else:
            console.print("  [red]File not created[/red]")
            passed2 = False

        if passed2:
            console.print("  [bold green]PASS[/bold green]")
        else:
            console.print("  [bold red]FAIL[/bold red]")

        console.print()
        if passed and passed2:
            console.print("[bold green]All smoke tests passed![/bold green]")
            return True
        else:
            console.print("[bold red]Some smoke tests failed.[/bold red]")
            return False


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "ollama/qwen3:4b"
    success = asyncio.run(smoke_test(model))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
