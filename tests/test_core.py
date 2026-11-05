"""Offline tests for extraction, checks, calendar export and chat. The Claude client is faked."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.chat import answer_question  # noqa: E402
from api.models import ChatMessage  # noqa: E402
from src.demo_extract import (  # noqa: E402
    TOOL_NAME,
    VALIDATION_TOOL_NAME,
    GradingWeight,
    ImportantDate,
    SyllabusExtraction,
    build_calendar,
    extract_syllabus,
    validate_extraction,
)


class FakeClient:
    """Hands back the queued responses in order and records every call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def tool_response(name, payload, tool_id="toolu_1"):
    block = SimpleNamespace(type="tool_use", name=name, id=tool_id, input=payload)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


BAD_DATE = {"important_dates": [{"label": "Final", "date_iso": "2025-02-30"}]}
TEXT_BLOCKS = [{"type": "text", "text": "CS 101 syllabus"}]


def test_impossible_date_is_rejected():
    with pytest.raises(ValidationError):
        ImportantDate(label="Final", date_iso="2025-02-30")
    assert ImportantDate(label="Final", date_iso="2025-12-10").date_iso == "2025-12-10"


def test_extraction_retries_once_and_tells_the_model_what_failed():
    client = FakeClient(
        tool_response(TOOL_NAME, BAD_DATE, tool_id="toolu_1"),
        tool_response(TOOL_NAME, {"course_code": "CS 101"}, tool_id="toolu_2"),
    )
    result = extract_syllabus(TEXT_BLOCKS, client=client, debug=False)

    assert result.course_code == "CS 101"
    assert len(client.calls) == 2
    assert len(client.calls[0]["messages"]) == 1
    feedback = client.calls[1]["messages"][-1]["content"][0]
    assert feedback["type"] == "tool_result"
    assert feedback["tool_use_id"] == "toolu_1"
    assert feedback["is_error"] is True


def test_extraction_gives_up_after_one_retry():
    client = FakeClient(
        tool_response(TOOL_NAME, BAD_DATE),
        tool_response(TOOL_NAME, BAD_DATE, tool_id="toolu_2"),
    )
    with pytest.raises(RuntimeError):
        extract_syllabus(TEXT_BLOCKS, client=client, debug=False)
    assert len(client.calls) == 2


def _grading(*percents):
    return SyllabusExtraction(
        grading_weights=[GradingWeight(component=f"Part {i}", percent=p) for i, p in enumerate(percents)]
    )


def test_grading_total_is_added_up_in_python():
    client = FakeClient(tool_response(VALIDATION_TOOL_NAME, {"warnings": []}))
    result = asyncio.run(validate_extraction(_grading(50, 30), client=client, debug=False))

    assert result.grading_total == 80
    assert result.grading_sums_to_100 is False
    assert any("80" in w.message for w in result.warnings)


def test_grading_that_adds_up_gives_no_warning():
    client = FakeClient(tool_response(VALIDATION_TOOL_NAME, {"warnings": []}))
    result = asyncio.run(validate_extraction(_grading(60, 40), client=client, debug=False))

    assert result.grading_sums_to_100 is True
    assert result.warnings == []


def test_rounded_thirds_still_count_as_100():
    client = FakeClient(tool_response(VALIDATION_TOOL_NAME, {"warnings": []}))
    result = asyncio.run(validate_extraction(_grading(33.3, 33.3, 33.3), client=client, debug=False))

    assert result.grading_sums_to_100 is True


def test_date_warnings_from_the_model_are_passed_through():
    warning = {"field": "important_dates", "message": "Withdrawal is after the final.", "severity": "medium"}
    client = FakeClient(tool_response(VALIDATION_TOOL_NAME, {"warnings": [warning]}))
    result = asyncio.run(validate_extraction(_grading(100), client=client, debug=False))

    assert [w.message for w in result.warnings] == ["Withdrawal is after the final."]


def test_calendar_events_have_unique_uid_and_dtstamp():
    dates = [
        ImportantDate(label="Midterm", date_iso="2025-10-15"),
        ImportantDate(label="Final", date_iso="2025-12-10"),
        ImportantDate(label="Quiz", raw_text="Week 8"),
    ]
    ics = build_calendar(dates, "CS 101").to_ical().decode().replace("\r\n ", "")

    assert ics.count("BEGIN:VEVENT") == 2
    assert "DTSTART;VALUE=DATE:20251015" in ics
    assert "DTEND;VALUE=DATE:20251016" in ics
    assert ics.count("DTSTAMP:") == 2
    uids = [line for line in ics.splitlines() if line.startswith("UID:")]
    assert len(uids) == 2 and len(set(uids)) == 2


def test_chat_returns_answer_and_extends_history():
    client = FakeClient(SimpleNamespace(content=[SimpleNamespace(type="text", text="10% per day.")]))
    extraction = SyllabusExtraction(course_code="CS 101", policy_bullets=["Late work loses 10% per day."])
    history = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]

    answer, updated = answer_question(extraction, history, "Late policy?", client=client)

    assert answer == "10% per day."
    assert [m.role for m in updated] == ["user", "assistant", "user", "assistant"]
    sent = client.calls[0]
    assert "Late work loses 10% per day." in sent["system"]
    assert sent["messages"][-1] == {"role": "user", "content": "Late policy?"}


def test_chat_raises_when_the_model_returns_no_text():
    client = FakeClient(SimpleNamespace(content=[]))
    with pytest.raises(RuntimeError):
        answer_question(SyllabusExtraction(), [], "anything", client=client)
