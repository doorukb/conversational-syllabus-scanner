# Conversational Syllabus Scanner

[![CI](https://github.com/doorukb/conversational-syllabus-scanner/actions/workflows/ci.yml/badge.svg)](https://github.com/doorukb/conversational-syllabus-scanner/actions/workflows/ci.yml)

Upload a course syllabus (PDF, PNG, JPG or TXT) and get the grading weights, important dates, policies and grade scale back as JSON. You can also export the dates as a calendar file and ask questions about the syllabus in a chat box.

This is a small personal project. It calls the Anthropic API (Claude Haiku 4.5), so you need your own API key. I have not measured how accurate the extraction is, so check the output against your syllabus.

<p align="center">
<img width="400" alt="Example chat" src="https://github.com/user-attachments/assets/14144567-6cb8-4490-99be-76ec96b97bbd" />
</p>

### How it works

Step 1 - Extraction : one Claude call reads the file and fills a fixed schema (tool use). The result is checked with Pydantic. If the check fails, the call is retried once and the error is sent back to the model. If it fails again, the request returns an error.

Step 2 - Checks : two calls run at the same time with asyncio.gather. One looks for date conflicts, like a withdrawal deadline after the final exam. The other rates each policy as low, medium or high. The grading total is added up in Python, not by the model. These checks only see the extracted data, not the original file, so they cannot tell you the extraction itself is wrong.

Step 3 - Chat : the extracted JSON goes into the system prompt and the model is told to answer only from it. The chat history is kept in the browser and sent back with every message.

A normal upload is 3 Claude calls (4 if the extraction is retried). Each chat message is 1 call.

### Run it

You need an Anthropic API key (https://console.anthropic.com/settings/keys).

    git clone https://github.com/doorukb/conversational-syllabus-scanner.git
    cd conversational-syllabus-scanner
    cp .env.example .env        # Windows: copy .env.example .env

Put your key in `.env`. Never commit that file.

With Docker:

    docker compose up --build

Without Docker (Python 3.12):

    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn api.main:app

Open http://localhost:8000. You can try it with the `syllabus.txt` in this repo.

### CLI and API

    python src/demo_extract.py --file syllabus.txt
    python src/demo_extract.py --file syllabus.txt --export-calendar out.ics

The REST API docs are at http://localhost:8000/docs. This is the shape of the response (the values are placeholders):

    {
      "extraction": {
        "course_code": "...",
        "instructor_email": "...",
        "grading_weights": [{ "component": "...", "percent": 30.0 }],
        "important_dates": [{ "label": "...", "date_iso": "2025-10-15", "raw_text": "" }],
        "policy_bullets": ["..."],
        "grade_scale": [{ "letter": "A", "min_percent": 93.0 }]
      },
      "validation": { "warnings": [], "grading_sums_to_100": true, "grading_total": 100.0 },
      "policy_flags": {
        "flags": [{ "policy": "...", "severity": "low", "reason": "..." }],
        "overall_severity": "low"
      }
    }

### Limitations

- Accuracy is not measured. The only sample in the repo is `syllabus.txt`, an invented example. The model can miss things or get dates wrong, especially on scans and photos.
- The calendar export only has events for dates that were turned into a specific day. A syllabus that says "Week 8" gives an empty calendar.
- The chat is told to answer only from the extracted data. I have not tested how well that holds, and the model can get grade math wrong.
- The text of the syllabus goes into the prompts as is, so a syllabus that contains instructions can change the output.
- There is no login and no rate limit, so run it locally. Do not put it on the internet with your API key in it. Uploads over 20 MB are rejected.

### Tests

    pip install -r requirements.txt -r requirements-dev.txt
    python -m pytest tests -q

The tests use a fake Claude client, so they run without an API key. They cover the code around the model (retry, grading sum, calendar, API errors), not how good the model's answers are.

MIT license, see LICENSE.
