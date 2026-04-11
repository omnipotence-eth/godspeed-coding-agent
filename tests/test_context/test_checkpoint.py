"""Tests for checkpoint save/restore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.context.checkpoint import (
    delete_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from godspeed.tui.commands import Commands


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestSaveCheckpoint:
    """Test checkpoint saving."""

    def test_save_creates_file(self, project_dir: Path) -> None:
        path = save_checkpoint(
            name="test-cp",
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            token_count=42,
            project_dir=project_dir,
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_preserves_data(self, project_dir: Path) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        save_checkpoint(
            name="data-test",
            system_prompt="system",
            messages=msgs,
            model="gpt-4",
            token_count=100,
            project_dir=project_dir,
        )
        data = load_checkpoint("data-test", project_dir)
        assert data is not None
        assert data["name"] == "data-test"
        assert data["model"] == "gpt-4"
        assert data["token_count"] == 100
        assert data["system_prompt"] == "system"
        assert len(data["messages"]) == 2

    def test_save_sanitizes_name(self, project_dir: Path) -> None:
        path = save_checkpoint(
            name="my checkpoint/v1",
            system_prompt="sys",
            messages=[],
            model="test",
            token_count=0,
            project_dir=project_dir,
        )
        assert path.exists()
        # Slashes should be sanitized
        assert "/" not in path.stem
        assert "\\" not in path.stem

    def test_save_overwrites_existing(self, project_dir: Path) -> None:
        save_checkpoint("dup", "sys", [{"role": "user", "content": "v1"}], "m", 10, project_dir)
        save_checkpoint("dup", "sys", [{"role": "user", "content": "v2"}], "m", 20, project_dir)
        data = load_checkpoint("dup", project_dir)
        assert data is not None
        assert data["token_count"] == 20
        assert data["messages"][0]["content"] == "v2"


class TestLoadCheckpoint:
    """Test checkpoint loading."""

    def test_load_nonexistent_returns_none(self, project_dir: Path) -> None:
        assert load_checkpoint("nonexistent", project_dir) is None

    def test_load_corrupt_file_returns_none(self, project_dir: Path) -> None:
        cp_dir = project_dir / ".godspeed" / "checkpoints"
        cp_dir.mkdir(parents=True)
        (cp_dir / "corrupt.checkpoint.json").write_text("not json{{{")
        assert load_checkpoint("corrupt", project_dir) is None


class TestListCheckpoints:
    """Test checkpoint listing."""

    def test_list_empty(self, project_dir: Path) -> None:
        result = list_checkpoints(project_dir)
        assert result == []

    def test_list_multiple(self, project_dir: Path) -> None:
        save_checkpoint("alpha", "sys", [], "m1", 10, project_dir)
        save_checkpoint("beta", "sys", [{"role": "user", "content": "hi"}], "m2", 20, project_dir)
        result = list_checkpoints(project_dir)
        assert len(result) == 2
        names = {cp["name"] for cp in result}
        assert names == {"alpha", "beta"}

    def test_list_includes_metadata(self, project_dir: Path) -> None:
        save_checkpoint("meta", "sys", [{"role": "user", "content": "x"}], "gpt-4", 50, project_dir)
        result = list_checkpoints(project_dir)
        assert len(result) == 1
        cp = result[0]
        assert cp["name"] == "meta"
        assert cp["model"] == "gpt-4"
        assert cp["token_count"] == 50
        assert cp["message_count"] == 1
        assert cp["timestamp"] > 0


class TestDeleteCheckpoint:
    """Test checkpoint deletion."""

    def test_delete_existing(self, project_dir: Path) -> None:
        save_checkpoint("to-delete", "sys", [], "m", 0, project_dir)
        assert delete_checkpoint("to-delete", project_dir) is True
        assert load_checkpoint("to-delete", project_dir) is None

    def test_delete_nonexistent(self, project_dir: Path) -> None:
        assert delete_checkpoint("nope", project_dir) is False


class TestCheckpointCommands:
    """Test /checkpoint and /restore slash commands."""

    @pytest.fixture
    def commands(self, tmp_path: Path) -> Commands:
        conversation = Conversation("System prompt.", max_tokens=100_000)
        llm_client = MagicMock()
        llm_client.model = "test-model"
        llm_client.fallback_models = []
        llm_client.total_input_tokens = 0
        llm_client.total_output_tokens = 0
        return Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=None,
            audit_trail=None,
            session_id="test-session",
            cwd=tmp_path,
        )

    def test_checkpoint_save(self, commands: Commands) -> None:
        commands._conversation.add_user_message("hello")
        commands._conversation.add_assistant_message("hi there")
        result = commands.dispatch("/checkpoint my-save")
        assert result is not None
        assert result.handled

        # Verify checkpoint was saved
        checkpoints = list_checkpoints(commands._cwd)
        assert len(checkpoints) == 1
        assert checkpoints[0]["name"] == "my-save"

    def test_checkpoint_list_empty(self, commands: Commands) -> None:
        result = commands.dispatch("/checkpoint")
        assert result is not None
        assert result.handled

    def test_checkpoint_list_with_entries(self, commands: Commands) -> None:
        commands.dispatch("/checkpoint save1")
        commands.dispatch("/checkpoint save2")
        result = commands.dispatch("/checkpoint list")
        assert result is not None
        assert result.handled

    def test_restore(self, commands: Commands) -> None:
        # Add messages and checkpoint
        commands._conversation.add_user_message("original message")
        commands._conversation.add_assistant_message("original response")
        commands.dispatch("/checkpoint before-clear")

        # Clear conversation
        commands.dispatch("/clear")
        assert len(commands._conversation.messages) == 1  # only system prompt

        # Restore
        result = commands.dispatch("/restore before-clear")
        assert result is not None
        assert result.handled

        # Verify messages were restored
        msgs = commands._conversation.messages
        assert len(msgs) == 3  # system + user + assistant
        assert msgs[1]["content"] == "original message"
        assert msgs[2]["content"] == "original response"

    def test_restore_nonexistent(self, commands: Commands) -> None:
        result = commands.dispatch("/restore nonexistent")
        assert result is not None
        assert result.handled

    def test_restore_no_args(self, commands: Commands) -> None:
        result = commands.dispatch("/restore")
        assert result is not None
        assert result.handled
