from __future__ import annotations
import asyncio
import base64
import logging
import os
from pathlib import Path
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from api.models import ExtractionResponse, HealthResponse, InfoResponse
from src.demo_extract import (
    MAX_BINARY_BYTES,
    ContentBlock,
    build_calendar,
    extract_syllabus,
    run_analysis,
)
from api.chat import answer_question
from api.models import CalendarRequest, ChatRequest, ChatResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = MAX_BINARY_BYTES

log = logging.getLogger(__name__)

class MissingAPIKey(RuntimeError):
    pass

app = FastAPI(
    title="Conversational Syllabus Scanner API",
    description="Extract structured data from course syllabi using Claude.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/app", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

def _get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKey("ANTHROPIC_API_KEY is not set.")
    return Anthropic(api_key=api_key)

# Map errors to HTTP codes without sending raw SDK or internal error text to the client.
def _to_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MissingAPIKey):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, anthropic.APIError):
        log.exception("Anthropic API error")
        return HTTPException(status_code=502, detail="The Anthropic API returned an error.")
    if isinstance(exc, RuntimeError):
        # these messages are written in demo_extract.py, so they are safe to show
        return HTTPException(status_code=502, detail=str(exc))
    log.exception("Unexpected error")
    return HTTPException(status_code=500, detail="Something went wrong. Check the server logs.")

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=302)

@app.get("/api", response_model=InfoResponse)
async def api_info() -> InfoResponse:
    return InfoResponse(
        name="Conversational Syllabus Scanner API",
        description="Extract structured data from course syllabi using Claude.",
        endpoints=[
            "GET  /app            — web UI",
            "GET  /api            — API info",
            "GET  /health         — health check",
            "POST /extract        — extract from text or file upload",
            "POST /extract/calendar — extract and return .ics file",
            "POST /calendar       — .ics from extracted JSON (no Claude)",
            "POST /chat           — syllabus Q&A (multi-turn)",
        ],
    )

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")

# Accept either a file upload PDF/PNG/JPEG/TXT or raw text via form field
# will return extraction, validation, and policy flag results
@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
) -> ExtractionResponse:
    if file is None and not text:
        raise HTTPException(
            status_code=422,
            detail="Provide either a file upload or a 'text' form field.",
        )

    blocks: list[ContentBlock] = []

    if file is not None:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        content_type = (file.content_type or "").lower()
        ext = os.path.splitext(file.filename or "")[1].lower()

        if "pdf" in content_type or ext == ".pdf":
            media_type, block_type = "application/pdf", "document"
        elif "png" in content_type or ext == ".png":
            media_type, block_type = "image/png", "image"
        elif "jpeg" in content_type or "jpg" in content_type or ext in {".jpg", ".jpeg"}:
            media_type, block_type = "image/jpeg", "image"
        else:
            blocks = [{"type": "text", "text": raw.decode("utf-8", errors="replace")}]
            media_type, block_type = None, None

        if media_type and block_type:
            b64 = base64.standard_b64encode(raw).decode("ascii")
            blocks = [
                {
                    "type": block_type,
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                }
            ]
    else:
        blocks = [{"type": "text", "text": text}]

    try:
        client = _get_client()
        # extract_syllabus is a blocking SDK call, so keep it off the event loop
        result = await asyncio.to_thread(extract_syllabus, blocks, client=client, debug=False)
        validation, policy = await run_analysis(result, client=client, debug=False)
    except Exception as e:
        raise _to_http_error(e) from e

    return ExtractionResponse(
        extraction=result,
        validation=validation,
        policy_flags=policy,
    )

# Same as POST /extract but returns an .ics iCalendar file for download
@app.post("/extract/calendar")
async def extract_calendar(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
) -> Response:
    extraction_response = await extract(file=file, text=text)
    result = extraction_response.extraction

    cal = build_calendar(result.important_dates, result.course_code)
    ics_bytes = cal.to_ical()

    filename = f"{result.course_code or 'syllabus'}.ics".replace(" ", "_")

    return Response(
        content=ics_bytes,
        media_type="text/calendar",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

# generates an .ics file from already-extracted syllabus data, no extra call needed
@app.post("/calendar")
async def generate_calendar(request: CalendarRequest) -> Response:
    cal = build_calendar(
        request.extraction.important_dates,
        request.extraction.course_code,
    )
    filename = f"{request.extraction.course_code or 'syllabus'}.ics".replace(" ", "_")
    return Response(
        content=cal.to_ical(),
        media_type="text/calendar",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

# answer a student question using the extracted syllabus data as context
# then send the full conversation history with every request
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        client = _get_client()
        answer, updated_history = await asyncio.to_thread(
            answer_question,
            request.extraction,
            request.history,
            request.message,
            client=client,
        )
    except Exception as e:
        raise _to_http_error(e) from e

    return ChatResponse(answer=answer, history=updated_history)
