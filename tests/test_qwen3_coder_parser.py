"""Tests for the Qwen3-Coder tool-call parser shim (v2.9.1)."""

from __future__ import annotations

import json

from godspeed.llm.qwen3_coder_parser import (
    _coerce_parameter_value,
    extract_qwen3_coder_tool_calls,
    looks_like_qwen3_coder_output,
)


class TestLooksLikeQwen3CoderOutput:
    def test_detects_function_tag(self) -> None:
        assert looks_like_qwen3_coder_output("<function=file_read>...")

    def test_empty_returns_false(self) -> None:
        assert not looks_like_qwen3_coder_output("")

    def test_plain_text_returns_false(self) -> None:
        assert not looks_like_qwen3_coder_output("Here is the answer: 42")


class TestCoerceParameterValue:
    def test_string(self) -> None:
        assert _coerce_parameter_value("pyproject.toml") == "pyproject.toml"

    def test_boolean_lowercase(self) -> None:
        assert _coerce_parameter_value("true") is True
        assert _coerce_parameter_value("false") is False

    def test_boolean_mixed_case(self) -> None:
        assert _coerce_parameter_value("True") is True
        assert _coerce_parameter_value("FALSE") is False

    def test_null(self) -> None:
        assert _coerce_parameter_value("null") is None

    def test_int(self) -> None:
        assert _coerce_parameter_value("42") == 42
        assert _coerce_parameter_value("-7") == -7

    def test_float(self) -> None:
        assert _coerce_parameter_value("3.14") == 3.14

    def test_json_object(self) -> None:
        assert _coerce_parameter_value('{"key": "value"}') == {"key": "value"}

    def test_json_array(self) -> None:
        assert _coerce_parameter_value("[1, 2, 3]") == [1, 2, 3]

    def test_whitespace_stripped(self) -> None:
        assert _coerce_parameter_value("  hello  \n") == "hello"

    def test_malformed_json_falls_back_to_string(self) -> None:
        assert _coerce_parameter_value("{not json}") == "{not json}"


class TestExtractQwen3CoderToolCalls:
    def test_returns_empty_on_no_match(self) -> None:
        assert extract_qwen3_coder_tool_calls("plain assistant reply") == []

    def test_single_call_with_one_parameter(self) -> None:
        content = (
            "<function=file_read>\n<parameter=file_path>\npyproject.toml\n</parameter>\n</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "file_read"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"file_path": "pyproject.toml"}
        assert calls[0]["id"].startswith("call_")

    def test_multiple_parameters(self) -> None:
        content = (
            "<function=file_edit>\n"
            "<parameter=file_path>\nsrc/x.py\n</parameter>\n"
            "<parameter=old_string>\nfoo\n</parameter>\n"
            "<parameter=new_string>\nbar\n</parameter>\n"
            "</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"file_path": "src/x.py", "old_string": "foo", "new_string": "bar"}

    def test_multiple_function_calls_in_one_response(self) -> None:
        content = (
            "<function=file_read>\n<parameter=file_path>\na.py\n</parameter>\n</function>\n"
            "some text\n"
            "<function=file_read>\n<parameter=file_path>\nb.py\n</parameter>\n</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 2
        assert json.loads(calls[0]["function"]["arguments"])["file_path"] == "a.py"
        assert json.loads(calls[1]["function"]["arguments"])["file_path"] == "b.py"

    def test_boolean_parameter_coerced(self) -> None:
        content = (
            "<function=file_read>\n"
            "<parameter=file_path>\nx.py\n</parameter>\n"
            "<parameter=recursive>\ntrue\n</parameter>\n"
            "</function>"
        )
        args = json.loads(extract_qwen3_coder_tool_calls(content)[0]["function"]["arguments"])
        assert args["recursive"] is True

    def test_malformed_missing_close_tag_is_ignored(self) -> None:
        content = "<function=file_read>\n<parameter=file_path>\nx.py\n"
        assert extract_qwen3_coder_tool_calls(content) == []

    def test_ids_are_unique_across_calls(self) -> None:
        content = (
            "<function=shell>\n<parameter=command>\nls\n</parameter>\n</function>\n"
            "<function=shell>\n<parameter=command>\npwd\n</parameter>\n</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        assert calls[0]["id"] != calls[1]["id"]
