from __future__ import annotations
import argparse
import base64
import datetime
import functools
import json
import os
import sys
from typing import Any, Final, TypeAlias
import asyncio
from anthropic import Anthropic
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from icalendar import Calendar, Event

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv not installed; set ANTHROPIC_API_KEY directly
    load_dotenv = lambda: None  # noqa: E731

# Pinned snapshot ID (weights fixed for this ID). See README for aliases / upgrades. Feel free to change it into a different model for your own use
MODEL_ID: Final[str] = "claude-haiku-4-5-20251001"

DEFAULT_MAX_CHARS: Final[int] = 50_000
MAX_OUTPUT_TOKENS: Final[int] = 4096
MAX_BINARY_BYTES: Final[int] = 32 * 1024 * 1024
TOOL_NAME: Final[str] = "submit_syllabus_extraction"

ContentBlock: TypeAlias = dict[str, Any]

_EXT_MAP: Final[dict[str, tuple[str, str]]] = {
    ".pdf": ("document", "application/pdf"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"),
}

class GradingWeight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str = Field(..., description="Name of graded component, e.g. Midterm, Homework")
    percent: float = Field(..., description="Weight as percent (0-100)")

class GradeThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: str = Field(..., description="Letter grade, e.g. 'A', 'A-', 'B+'")
    min_percent: float | None = Field(
        default=None,
        description=(
            "Minimum overall percentage (0-100) needed to earn this letter grade. "
            "Use null if the syllabus lists the letter without a numeric cutoff."
        ),
    )

class ImportantDate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(..., description="Human-readable label, e.g. Final exam")
    date_iso: str | None = Field(
        default=None,
        description="ISO-8601 date if known, else null",
    )
    raw_text: str = Field(
        default="",
        description="Verbatim date/time phrase from syllabus if ISO not inferable",
    )

    @field_validator("date_iso", mode="before")
    @classmethod
    def _validate_iso_date(cls, v: object) -> object:
        if v is None or v == "":
            return v
        try:
            datetime.date.fromisoformat(str(v))
        except ValueError as exc:
            raise ValueError(
                f"date_iso must be a valid ISO-8601 date (YYYY-MM-DD), got: {v!r}"
            ) from exc
        return v

# Structured syllabus extraction target returned via native tool use.
class SyllabusExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course_code: str | None = Field(default=None, description="Course code if present, e.g. CS 101")
    instructor_email: str | None = Field(default=None, description="Primary instructor email if present")
    grading_weights: list[GradingWeight] = Field(default_factory=list)
    important_dates: list[ImportantDate] = Field(default_factory=list)
    policy_bullets: list[str] = Field(default_factory=list)
    grade_scale: list[GradeThreshold] = Field(
        default_factory=list,
        description="Letter-grade cutoffs / distribution if present in the syllabus",
    )

class ValidationWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Which field the warning relates to, e.g. grading_weights")
    message: str = Field(..., description="Plain-English description of the issue found")
    severity: str = Field(..., description="'low', 'medium', or 'high'")

class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warnings: list[ValidationWarning] = Field(default_factory=list)
    grading_sums_to_100: bool = Field(..., description="True if grading weights sum to 100%")
    grading_total: float = Field(..., description="Actual sum of grading weights")

class PolicyFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: str = Field(..., description="The original policy text being evaluated")
    severity: str = Field(..., description="'low', 'medium', or 'high'")
    reason: str = Field(..., description="Plain-English explanation of the severity rating")

class PolicyFlagResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flags: list[PolicyFlag] = Field(default_factory=list)
    overall_severity: str = Field(
        ...,
        description="Highest severity level across all policies: 'low', 'medium', or 'high'"
    )

def _debug_stderr(debug: bool, message: str) -> None:
    if debug:
        print(message, file=sys.stderr)

# returns the lowercase extension (e.g. '.pdf') or '.txt' for everything else
def _detect_input_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ext if ext in _EXT_MAP else ".txt"

# read a PDF or image and return a single API content block
def _binary_blocks_from_file(path: str) -> list[ContentBlock]:
    ext = os.path.splitext(path)[1].lower()
    block_type, media_type = _EXT_MAP[ext]
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) > MAX_BINARY_BYTES:
        raise ValueError(
            f"File is {len(raw) // (1024 * 1024)} MB — "
            f"exceeds the {MAX_BINARY_BYTES // (1024 * 1024)} MB cap. "
            "Reduce the file size or extract the text manually first."
        )
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return [
        {
            "type": block_type,
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }
    ]

@functools.lru_cache(maxsize=1)
def _build_tool() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Submit structured syllabus fields extracted from the document.",
        "input_schema": SyllabusExtraction.model_json_schema(),
    }

VALIDATION_TOOL_NAME: Final[str] = "report_validation"

@functools.lru_cache(maxsize=1)
def _build_validation_tool() -> dict[str, Any]:
    return {
        "name": VALIDATION_TOOL_NAME,
        "description": (
            "Report logical consistency issues found in extracted syllabus data. "
            "Always call this tool even if no warnings are found."
        ),
        "input_schema": ValidationResult.model_json_schema(),
    }

@functools.lru_cache(maxsize=1)
def _build_policy_tool() -> dict[str, object]:
    schema = PolicyFlagResult.model_json_schema()
    return {
        "name": "report_policy_flags",
        "description": (
            "Report student-friendliness ratings for each policy in a syllabus. "
            "Always call this tool even if all policies are low severity."
        ),
        "input_schema": schema,
    }

@functools.lru_cache(maxsize=1)
def _build_system_prompt() -> str:
    return (
        "You extract syllabus information into structured data. "
        "Use null for unknown scalar fields; use empty arrays when no items apply. "
        "Do not echo the source document."
    )

# Append the extraction instruction after the document block(s)
def _build_user_content(doc_blocks: list[ContentBlock]) -> list[ContentBlock]:
    instruction: ContentBlock = {
        "type": "text",
        "text": (
            "Extract all syllabus fields from the document above. "
            "Include the letter-grade scale in grade_scale: one entry per letter "
            "with its minimum overall percentage when the syllabus states cutoffs. "
            f"Call the {TOOL_NAME} tool with the structured data."
        ),
    }
    return doc_blocks + [instruction]

def _call_model(client: Anthropic, *, messages: list[dict[str, Any]], debug: bool) -> Message:
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=_build_system_prompt(),
        messages=messages,
        tools=[_build_tool()],
        tool_choice={"type": "tool", "name": TOOL_NAME},
    )
    _debug_stderr(debug, f"stop_reason={response.stop_reason}")
    return response

def _parse_tool_use(message: Message) -> SyllabusExtraction:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            return SyllabusExtraction.model_validate(block.input)
    raise RuntimeError(
        f"Model did not return a {TOOL_NAME!r} tool_use block."
    )

# single API call; model returns structured data via tool_use
def extract_syllabus(doc_blocks: list[ContentBlock], *, client: Anthropic, debug: bool) -> SyllabusExtraction:
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_content(doc_blocks)}]
    response = _call_model(client, messages=messages, debug=debug)
    try:
        return _parse_tool_use(response)
    except ValidationError as exc:
        raise RuntimeError(
            "Tool input did not pass Pydantic validation."
        ) from exc

def _build_validation_system_prompt() -> str:
    return (
        "You are a syllabus data validator. "
        "You will receive extracted syllabus data as JSON and must check it for logical issues. "
        f"Always call the {VALIDATION_TOOL_NAME} tool, even when no issues were found."
    )

# Second chained call: checks extracted data for logical inconsistencies.
async def validate_extraction(
    extraction: SyllabusExtraction,
    *,
    client: Anthropic,
    debug: bool,
) -> ValidationResult:
    payload = json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False)
    user_content = (
        "Check this extracted syllabus data for logical inconsistencies:\n\n"
        f"{payload}\n\n"
        "Specifically:\n"
        "1. Do the grading_weights percentages sum to 100? Set grading_total to their sum "
        "and grading_sums_to_100 accordingly. Add a warning if they do not sum to 100.\n"
        "2. Are any important_dates logically inconsistent with each other "
        "(e.g. final before midterm, withdrawal deadline after final exam)? "
        "Add a warning for each issue found.\n"
        f"Report your findings via the {VALIDATION_TOOL_NAME} tool."
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=_build_validation_system_prompt(),
            tools=[_build_validation_tool()],
            tool_choice={"type": "tool", "name": VALIDATION_TOOL_NAME},
            messages=[{"role": "user", "content": user_content}],
        ),
    )
    _debug_stderr(debug, f"validation stop_reason={response.stop_reason}")

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == VALIDATION_TOOL_NAME:
            try:
                return ValidationResult.model_validate(block.input)
            except ValidationError as exc:
                raise RuntimeError(
                    "Validation tool input did not pass Pydantic validation."
                ) from exc
    raise RuntimeError(
        f"Validation call did not return a {VALIDATION_TOOL_NAME!r} tool_use block. "
        f"Content types: {[getattr(b, 'type', '?') for b in response.content]}"
    )

# the third chained call to score each policy bullet for harshness.
# Anthropic SDK is synchronous, and client.messages.create() is a blocking call.
# we will use loop.run_in_executor to run the call asynchronously because asyncio is not enough
# None argument means to use the default thread pool executor.
async def flag_policies(
    extraction: SyllabusExtraction,
    *,
    client: Anthropic,
    debug: bool,
) -> PolicyFlagResult:
    if not extraction.policy_bullets:
        return PolicyFlagResult(flags=[], overall_severity="low")

    bullets = "\n".join(f"- {p}" for p in extraction.policy_bullets)

    system = (
        "You are a student-advocate syllabus reviewer. "
        "Rate each policy for how strict or punishing it is from a student's perspective. "
        "Always call the report_policy_flags tool."
    )
    user_content = (
        "Rate each of these syllabus policies for student-friendliness:\n\n"
        f"{bullets}\n\n"
        "Consider late-penalty severity, attendance strictness, accommodation "
        "flexibility, and academic-dishonesty consequences when assigning severity.\n\n"
        "Severity guide:\n"
        "  low    — standard, reasonable, no cause for concern\n"
        "  medium — notably strict, students should be aware\n"
        "  high   — unusually punishing or restrictive\n\n"
        "Return one flag per policy with a short reason. "
        "Set overall_severity to the highest severity level found across all policies."
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            tools=[_build_policy_tool()],
            tool_choice={"type": "tool", "name": "report_policy_flags"},
            messages=[{"role": "user", "content": user_content}],
        ),
    )
    _debug_stderr(debug, f"policy stop_reason={response.stop_reason}")

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "report_policy_flags":
            return PolicyFlagResult.model_validate(block.input)

    raise RuntimeError(
        "Policy flagging call did not return a report_policy_flags tool_use block. "
        f"Content types: {[getattr(b, 'type', '?') for b in response.content]}"
    )

# run validation and policy flagging concurrently.
# I decided to use asyncio.gather to run the two calls concurrently because it
# takes any number of awaitables and runs them concurrently, returning all results 
# once every one of them finishes. Both API calls are now concurrent
async def run_analysis(
    extraction: SyllabusExtraction,
    *,
    client: Anthropic,
    debug: bool,
) -> tuple[ValidationResult, PolicyFlagResult]:
    validation, policy = await asyncio.gather(
        validate_extraction(extraction, client=client, debug=debug),
        flag_policies(extraction, client=client, debug=debug),
    )
    return validation, policy

# Convert a list of ImportantDate objects into an icalendar Calendar
def build_calendar(
    dates: list[ImportantDate],
    course_code: str | None,
) -> Calendar:
    cal = Calendar()
    cal.add("prodid", "-//Conversational Syllabus Scanner//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", course_code or "Syllabus")

    for item in dates:
        if not item.date_iso:
            continue

        start = datetime.date.fromisoformat(item.date_iso)
        event = Event()
        event.add("summary", item.label)
        event.add("dtstart", start)
        # All-day events use an exclusive end date (day after start), per RFC 5545.
        event.add("dtend", start + datetime.timedelta(days=1))
        event.add("description", item.raw_text or item.label)
        cal.add_component(event)

    return cal

# serialise a Calendar object and write it to a .ics file.
def write_calendar(cal: Calendar, path: str) -> None:
    with open(path, "wb") as f:
        f.write(cal.to_ical())
    print(f"Calendar written to {path}", file=sys.stderr)

# return content blocks representing the user's document
def _read_input(args: argparse.Namespace, *, debug: bool) -> list[ContentBlock]:
    if args.file is None:
        return [{"type": "text", "text": sys.stdin.read()}]

    path = os.path.abspath(args.file)
    input_type = _detect_input_type(path)

    if input_type == ".txt":
        with open(path, encoding="utf-8", errors="replace") as f:
            return [{"type": "text", "text": f.read()}]

    if args.max_chars != DEFAULT_MAX_CHARS:
        _debug_stderr(debug, "warning: --max-chars has no effect on PDF/image input")

    _debug_stderr(debug, f"input_type={input_type[1:]} path={path}")
    return _binary_blocks_from_file(path)

# truncate text blocks only; binary blocks (PDF, image) pass through unchanged
def _truncate_content(blocks: list[ContentBlock], max_chars: int, *, debug: bool) -> list[ContentBlock]:
    result = []
    for block in blocks:
        if block.get("type") == "text":
            text = block["text"]
            if len(text) > max_chars:
                _debug_stderr(debug, f"input_truncated length={len(text)} max_chars={max_chars}")
                block = {**block, "text": text[:max_chars]}
        result.append(block)
    return result

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract syllabus-oriented JSON from text using Anthropic Messages API.",
    )
    p.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Syllabus file — .txt (default: stdin), .pdf, .jpg, or .png",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        metavar="N",
        help=f"max characters for text input (default: {DEFAULT_MAX_CHARS}). No effect on PDF/image.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print minimal diagnostics to stderr (never prints full user document or model JSON).",
    )
    p.add_argument(
        "--export-calendar",
        metavar="PATH",
        default=None,
        help="Write extracted dates to an .ics iCalendar file at this path.",
    )
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not str(api_key).strip():
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2
    try:
        blocks = _read_input(args, debug=args.debug)
    except (OSError, ValueError) as e:
        print(f"error: could not read input: {e}", file=sys.stderr)
        return 2
    blocks = _truncate_content(blocks, args.max_chars, debug=args.debug)
    # Guard against empty text input (binary blocks are never empty)
    if all(b.get("type") == "text" and not b["text"].strip() for b in blocks):
        print("error: input is empty after trimming.", file=sys.stderr)
        return 2

    client = Anthropic(api_key=api_key)
    try:
        result = extract_syllabus(blocks, client=client, debug=args.debug)
    except Exception as e:
        print(f"error: extraction failed: {e}", file=sys.stderr)
        return 1
    try:
        validation, policy = asyncio.run(
            run_analysis(result, client = client, debug = args.debug),
        )
    except Exception as e:
        print(f"error: analysis failed: {e}", file=sys.stderr)
        return 1
    output = {
        "extraction": result.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "policy_flags" : policy.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.export_calendar:
        if not any(d.date_iso for d in result.important_dates):
            print(
                "warning: no dates with ISO values found; calendar will be empty",
                file=sys.stderr,
            )
        cal = build_calendar(result.important_dates, result.course_code)
        write_calendar(cal, args.export_calendar)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())