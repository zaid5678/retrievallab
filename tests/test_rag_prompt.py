
import pytest

from src.retrievallab.rag import build_messages


def test_prompt_includes_question_evidence_and_source():
    results = [
        {
            "source": "security.md",
            "text": "Access is restricted to authorised personnel.",
        }
    ]

    messages = build_messages(
        "Who can access customer records?",
        results,
    )

    assert messages[1]["role"] == "user"
    assert "Who can access customer records?" in messages[1]["content"]
    assert "Access is restricted to authorised personnel." in messages[1]["content"]
    assert "[Source: security.md]" in messages[1]["content"]


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        build_messages("  ", [{"source": "security.md", "text": "Example"}])


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError):
        build_messages("What is the policy?", [])
