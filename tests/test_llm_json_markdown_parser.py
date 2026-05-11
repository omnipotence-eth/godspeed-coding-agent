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

    def test_bare_json(self):
        assert looks_like_json_tool_call_output('{"name": "foo"}') is True

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

    def test_bare_json_tool_call(self):
        content = (
            '{"name": "file_write", "arguments": {"file_path": "test.txt", "content": "hello"}}'
        )
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_write"

    def test_bare_json_in_text(self):
        content = """Let me create that file for you.
{"name": "file_write", "arguments": {"file_path": "test.txt", "content": "hello"}}
This should work."""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_write"
        args = result[0]["function"]["arguments"]
        assert '"file_path": "test.txt"' in args

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


class TestParseToolCallItemEdgeCases:
    def test_name_is_none(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        assert _parse_tool_call_item({"arguments": {}}) is None

    def test_name_is_int_not_str(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        assert _parse_tool_call_item({"name": 123, "arguments": {}}) is None

    def test_name_is_empty_string(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        assert _parse_tool_call_item({"name": "", "arguments": {}}) is None

    def test_arguments_as_invalid_json_string(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        result = _parse_tool_call_item({"name": "test", "arguments": "{not valid json}"})
        assert result is not None
        args = result["function"]["arguments"]
        assert "raw" in args

    def test_arguments_as_non_dict(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        result = _parse_tool_call_item({"name": "test", "arguments": 42})
        assert result is not None
        args = result["function"]["arguments"]
        assert '"raw": 42' in args

    def test_arguments_as_list(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        result = _parse_tool_call_item({"name": "test", "arguments": [1, 2, 3]})
        assert result is not None
        args = result["function"]["arguments"]
        assert "raw" in args

    def test_using_tool_name_key(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        result = _parse_tool_call_item({"tool_name": "my_tool", "args": {"x": 1}})
        assert result is not None
        assert result["function"]["name"] == "my_tool"

    def test_using_function_key(self) -> None:
        from godspeed.llm.json_markdown_parser import _parse_tool_call_item

        result = _parse_tool_call_item({"function": "do_thing", "parameters": {"x": 1}})
        assert result is not None
        assert result["function"]["name"] == "do_thing"


class TestParseJsonTextEdgeCases:
    def test_list_with_non_dict_items(self) -> None:
        content = """
```json
[
  {"name": "file_read", "arguments": {"file_path": "a.py"}},
  "not a dict",
  {"name": "file_write", "arguments": {"file_path": "b.py"}}
]
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "file_read"
        assert result[1]["function"]["name"] == "file_write"

    def test_item_with_function_key_as_subfield(self) -> None:
        content = """
```json
{"name": "shell", "arguments": {"command": "ls"}, "function": "ignored"}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "shell"


class TestBareJsonEdgeCases:
    def test_bare_json_with_surrounding_text_and_newline(self) -> None:
        content = """Here is my analysis.

{"name": "file_read", "arguments": {"file_path": "src/main.py"}}

Let me explain further."""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_read"

    def test_bare_json_with_nested_objects_tracks_brace_depth(self) -> None:
        content = """Text before.

{"name": "shell", "arguments": {"command": "echo '{\\"key\\": \\"value\\"}'"}}

Text after."""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "shell"

    def test_multiple_bare_json_objects(self) -> None:
        content = """
{"name": "file_read", "arguments": {"file_path": "a.py"}}

{"name": "file_write", "arguments": {"file_path": "b.py"}}
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2

    def test_bare_json_list_of_objects(self) -> None:
        content = """
Text here.
[
{"name": "a", "arguments": {}},
{"name": "b", "arguments": {}}
]
More text."""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2


class TestMarkdownBlockEdgeCases:
    def test_empty_code_block(self) -> None:
        content = "```json\n\n```"
        result = extract_json_markdown_tool_calls(content)
        assert result == []

    def test_generic_fenced_block_without_json_tag(self) -> None:
        content = """
```
{"name": "file_write", "arguments": {"file_path": "x"}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_write"

    def test_non_json_code_block_ignored(self) -> None:
        content = """```python
print("hello world")
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert result == []

    def test_mixed_json_and_non_json_blocks(self) -> None:
        content = """Some text before.

```json
{"name": "file_read", "arguments": {"file_path": "test.py"}}
```

Some text after.
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_read"

    def test_json_array_instead_of_object(self) -> None:
        content = """```json
[
  {"name": "tool_a", "arguments": {"x": 1}},
  {"name": "tool_b", "arguments": {"y": 2}}
]
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2

    def test_json_with_comment_style_content(self) -> None:
        content = """```json
{"name": "file_write", "arguments": {"file_path": "//some/path"}}
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_write"

    def test_empty_block_with_valid_block_in_same_content(self) -> None:
        content = """{"name": "valid_tool", "arguments": {}}

```json


```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid_tool"

    def test_bare_json_not_a_valid_tool_call(self) -> None:
        content = """
The tool has "name": "file_write" in its definition.

{"name": 42, "arguments": {}}

End.
"""
        result = extract_json_markdown_tool_calls(content)
        assert result == []

    def test_unbalanced_braces_in_bare_json(self) -> None:
        content = """
{"name": "tool", "arguments": {"key": "val"}

Trailing text."""
        result = extract_json_markdown_tool_calls(content)
        assert result == []

    def test_list_with_item_returning_none(self) -> None:
        content = """```json
[
  {"name": "good_tool", "arguments": {"x": 1}},
  {"arguments": {"y": 2}},
  {"name": "another_good", "arguments": {}}
]
```
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "good_tool"
        assert result[1]["function"]["name"] == "another_good"

    def test_bare_json_candidate_that_is_invalid_json(self) -> None:
        content = """
{"name": "tool", "arguments": {"key": "value"}}

Some trailing text here.
"""
        result = extract_json_markdown_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "tool"
