from __future__ import annotations
import json
from anthropic import Anthropic
from src.demo_extract import MODEL_ID, MAX_OUTPUT_TOKENS, SyllabusExtraction
from api.models import ChatMessage

# this will inject the extracted syllabus data as the authoritative source of truth for the conversation
# claude answers only from this data.
def build_chat_system_prompt(extraction: SyllabusExtraction) -> str:
    data = json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False)
    return (
        "You are a helpful assistant that answers student questions about their course syllabus.\n"
        "Answer only using the structured syllabus data provided below.\n"
        "If the answer is not present in the data, say so clearly — do not guess or use outside knowledge.\n"
        "Be concise and direct. Use plain language a student would appreciate.\n"
        "Use short paragraphs separated by a blank line.\n"
        "Use **text** only for key terms or numbers you want emphasized.\n"
        "When the data includes grade_scale, you may compute a student's overall "
        "percentage from grading_weights and map it to the matching letter grade.\n\n"
        f"SYLLABUS DATA:\n{data}"
    )

# Send the full conversation history plus the new message to Claude.
# will return the answer text and the updated history.
def answer_question(
    extraction: SyllabusExtraction,
    history: list[ChatMessage],
    message: str,
    *,
    client: Anthropic,
) -> tuple[str, list[ChatMessage]]:
    # Build the messages list from history + new user message
    messages = [
        {"role": m.role, "content": m.content}
        for m in history
    ]
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=build_chat_system_prompt(extraction),
        messages=messages,
    )

    texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not texts:
        raise RuntimeError("The model returned no text for this message.")
    answer = "\n".join(texts)

    # Build updated history including this turn
    updated_history = list(history) + [
        ChatMessage(role="user", content=message),
        ChatMessage(role="assistant", content=answer),
    ]

    return answer, updated_history