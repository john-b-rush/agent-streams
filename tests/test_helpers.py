from pathlib import Path

import agent_streams as mod


def test_extract_text_from_content_and_message() -> None:
    assert mod._extract_text({"content": [{"type": "text", "text": "hello"}]}) == "hello"
    assert (
        mod._extract_text({"message": {"content": [{"type": "text", "text": "world"}]}})
        == "world"
    )
    assert mod._extract_text({"completion": "done"}) == "done"
    assert mod._extract_text({"type": "text", "text": "ok"}) == "ok"
    assert (
        mod._extract_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        == "ab"
    )


def test_extract_text_empty() -> None:
    assert mod._extract_text({}) == ""
    assert mod._extract_text(None) == ""


def test_extract_usage_variants() -> None:
    usage = mod._extract_usage({"usage": {"input_tokens": 100, "output_tokens": 50}})
    assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    usage = mod._extract_usage({"usage": {"prompt_tokens": "7", "completion_tokens": "3"}})
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_load_json_maybe_variants() -> None:
    assert mod._load_json_maybe("") is None
    assert mod._load_json_maybe("not json") is None
    assert mod._load_json_maybe('{"key": "value"}') == {"key": "value"}
    assert mod._load_json_maybe('prefix {"key": "value"} suffix') == {"key": "value"}
    assert mod._load_json_maybe("nope\n{\"key\": \"value\"}\n") == {"key": "value"}


def test_repo_slug_generation() -> None:
    repo_root = Path("/path/to/repo")
    slug1 = mod._repo_slug(repo_root)
    slug2 = mod._repo_slug(repo_root)
    assert slug1 == slug2
    assert slug1.startswith("repo-")
