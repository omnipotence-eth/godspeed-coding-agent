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


class TestExtractQwen3EdgeCases:
    def test_empty_function_name_skipped(self) -> None:
        content = "<function=>\n<parameter=key>\nvalue\n</parameter>\n</function>"
        calls = extract_qwen3_coder_tool_calls(content)
        assert calls == []

    def test_whitespace_only_function_name_skipped(self) -> None:
        content = "<function=   >\n<parameter=key>\nvalue\n</parameter>\n</function>"
        calls = extract_qwen3_coder_tool_calls(content)
        assert calls == []

    def test_empty_parameter_key_skipped(self) -> None:
        content = "<function=tool>\n<parameter=>\nvalue\n</parameter>\n</function>"
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {}

    def test_whitespace_only_parameter_key_skipped(self) -> None:
        content = "<function=tool>\n<parameter=   >\nvalue\n</parameter>\n</function>"
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {}

    def test_valid_and_invalid_params_mixed(self) -> None:
        content = (
            "<function=search>\n"
            "<parameter=>\nignored\n</parameter>\n"
            "<parameter=query>\ntest\n</parameter>\n"
            "<parameter=   >\nalso ignored\n</parameter>\n"
            "</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"query": "test"}

    def test_text_interleaved_with_tool_calls(self) -> None:
        content = (
            "Let me think about this...\n"
            "<function=file_read>\n<parameter=file_path>\nconfig.yaml\n</parameter>\n</function>\n"
            "Now let me check another file...\n"
            "<function=file_read>\n<parameter=file_path>\nsettings.json\n</parameter>\n</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 2
        assert json.loads(calls[0]["function"]["arguments"])["file_path"] == "config.yaml"
        assert json.loads(calls[1]["function"]["arguments"])["file_path"] == "settings.json"

    def test_no_function_tags_returns_empty(self) -> None:
        content = "Just a plain text response with no tool calls whatsoever."
        assert extract_qwen3_coder_tool_calls(content) == []

    def test_malformed_function_with_valid_mixed(self) -> None:
        content = (
            "<function=bad>\n"
            "missing close tag...\n"
            "<function=file_read>\n<parameter=file_path>\ngood.py\n</parameter>\n</function>"
        )
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bad"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"file_path": "good.py"}

    def test_function_name_with_special_chars(self) -> None:
        content = "<function=my_tool_123>\n<parameter=arg>\nvalue\n</parameter>\n</function>"
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "my_tool_123"

    def test_parameter_value_with_whitespace_only(self) -> None:
        content = "<function=test>\n<parameter=key>\n   \n</parameter>\n</function>"
        calls = extract_qwen3_coder_tool_calls(content)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"key": ""}


class TestCoerceParameterValueEdgeCases:
    def test_empty_value_returns_empty_string(self) -> None:
        assert _coerce_parameter_value("") == ""
        assert _coerce_parameter_value("   ") == ""
        assert _coerce_parameter_value("\n\t") == ""

    def test_json_string_value(self) -> None:
        assert _coerce_parameter_value('"hello world"') == "hello world"

    def test_json_string_value_malformed(self) -> None:
        result = _coerce_parameter_value('"unclosed')
        assert result == '"unclosed'

    def test_number_with_exponent(self) -> None:
        assert _coerce_parameter_value("1e10") == 1e10
        assert _coerce_parameter_value("-2.5E-3") == -2.5e-3

    def test_number_parsing_fallthrough(self) -> None:
        result = _coerce_parameter_value("99999999999999999999999999999999999999")
        assert isinstance(result, (int, str))

    def test_integer_overflow_string_fallback(self) -> None:
        result = _coerce_parameter_value("99999999999999999999999x")
        assert result == "99999999999999999999999x"

    def test_null_case_insensitive(self) -> None:
        assert _coerce_parameter_value("NULL") is None
        assert _coerce_parameter_value("Null") is None

    def test_boolean_case_insensitive_variants(self) -> None:
        assert _coerce_parameter_value("TRUE") is True
        assert _coerce_parameter_value("False") is False

    def test_number_like_but_not_full_number(self) -> None:
        result = _coerce_parameter_value("123abc")
        assert result == "123abc"

    def test_array_with_nested_content(self) -> None:
        result = _coerce_parameter_value('[1, "two", {"three": 3}]')
        assert result == [1, "two", {"three": 3}]

    def test_nested_json_object(self) -> None:
        result = _coerce_parameter_value('{"outer": {"inner": [1, 2, 3]}}')
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_partial_number_parse_fallback(self) -> None:
        result = _coerce_parameter_value("-")
        assert result == "-"
