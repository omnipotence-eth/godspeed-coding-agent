"""Tests for the JSON-markdown tool-call parser."""

from __future__ import annotations

from godspeed.llm.json_markdown_parser import (
    extract_json_markdown_tool_calls,
    looks_like_json_tool_call_output,
)


class TestLooksLikeJsonToolCallOutput:
    def test_empty_string(self):
        assert looks_like_json_tool_call_output("") is False

    def test_plain_text(self):
        assert looks_like_json_tool_call_output("Hello world") is False

    def test_no_code_block(self):
        assert looks_like_json_tool_call_output('{"name": "foo"}') is False

    def test_code_block_without_tool_call(self):
        assert looks_like_json_tool_call_output("```json\n{'a': 1}\n```") is False

    def test_with_tool_call_fingerprint(self):
        assert looks_like_json_tool_call_output('```json\n{"name": "file_write"}\n```') is True


class TestExtractJsonMarkdownToolCalls:
    def test_empty(self):
        assert extract_json_markdown_tool_calls("") == []

    def test_no_blocks(self):
        assert extract_json_markdown_tool_calls("Just text") == []

    def test_single_tool_call(self):
        content = """
I'll create that file for you.

```json
{"name": "file_write", "arguments": {"file_path": "hello.py", "content": "print('hi')"}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_write"
        args = result[0]["function"]["arguments"]
        assert '"file_path": "hello.py"' in args
        assert '"content": "print(\'hi\')"' in args

    def test_multiple_tool_calls_in_list(self):
        content = """
```json
[
  {"name": "file_read", "arguments": {"file_path": "a.py"}},
  {"name": "file_write", "arguments": {"file_path": "b.py", "content": "x"}}
]
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "file_read"
        assert result[1]["function"]["name"] == "file_write"

    def test_arguments_as_json_string(self):
        content = """
```json
{"name": "shell", "arguments": "{\\"command\\": \\"echo hi\\"}"}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        args = result[0]["function"]["arguments"]
        assert '"command": "echo hi"' in args

    def test_alternative_keys(self):
        content = """
```json
{"tool_name": "grep_search", "parameters": {"pattern": "foo"}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "grep_search"

    def test_no_name_skipped(self):
        content = """
```json
{"arguments": {"x": 1}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 0

    def test_multiple_code_blocks(self):
        content = """
```json
{"name": "a", "arguments": {}}
```

Some explanation...

```json
{"name": "b", "arguments": {}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2
        names = [r["function"]["name"] for r in result]
        assert names == ["a", "b"]

    def test_returns_unique_ids(self):
        content = """
```json
{"name": "x", "arguments": {}}
```
```json
{"name": "y", "arguments": {}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("call_") for i in ids)

    def test_mixed_content_ignored(self):
        content = """
Here's my plan:

```json
{"name": "file_write", "arguments": {"file_path": "x", "content": "y"}}
```

That should work!
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_write"
