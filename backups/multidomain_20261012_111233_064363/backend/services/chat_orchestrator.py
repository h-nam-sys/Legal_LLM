"""Chat coordination extracted from the existing route. Public contracts unchanged."""
import ast
import json
import random
import re
import unicodedata
import difflib
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from services.llm_service import get_legal_response_stream
from services.answer_policy import resolve_mode, needs_retrieval
from services.llm_service import get_friendly_response_stream
from services.mcp_service import should_trigger_mcp
from services.db_service import (get_conversation, add_message, get_conversation_history,
    update_conversation_title, update_conversation_state, log_interaction)

def parse_sources(sources_data):
    if not sources_data:
        return []
    if isinstance(sources_data, list):
        return sources_data
    if isinstance(sources_data, str):
        try:
            parsed = json.loads(sources_data)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = ast.literal_eval(sources_data)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    return []

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def fuzzy_match_any(query: str, normalized_vocab_list: list[str], threshold: float = 0.85) -> bool:
    norm_query = normalize_text(query)
    for vocab in normalized_vocab_list:
        if vocab in norm_query:
            return True
    query_words = norm_query.split()
    for vocab in normalized_vocab_list:
        vocab_word_count = len(vocab.split())
        if vocab_word_count == 0:
            continue
        for i in range(len(query_words) - vocab_word_count + 1):
            ngram = " ".join(query_words[i:i + vocab_word_count])
            similarity = difflib.SequenceMatcher(None, vocab, ngram).ratio()
            if similarity >= threshold:
                return True
    return False

RAW_FORM_TRIGGERS = ["giấy tờ", "làm giấy", "cần gì", "hồ sơ", "biểu mẫu", "tờ khai"]
NORMALIZED_FORM_TRIGGERS = [normalize_text(w) for w in RAW_FORM_TRIGGERS if normalize_text(w)]

def is_form_trigger(query: str) -> bool:
    return fuzzy_match_any(query, NORMALIZED_FORM_TRIGGERS)

def active_procedures(messages, legacy_title=""):
    """Use the most recent local answer's source titles; no schema migration."""
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        sources = parse_sources(message.sources)
        if not any(isinstance(s, str) and s.startswith("RAG_CONTENT:") for s in sources):
            continue
        titles = list(dict.fromkeys(s for s in sources
            if isinstance(s, str) and s.strip() and not s.startswith("RAG_")))
        if titles:
            return titles
    return [legacy_title] if legacy_title else []


def resolve_search_query(prompt, active_titles):
    """Only plain property follow-ups reuse all active records.

    Explicit procedure/action questions retain the existing contextual search path;
    query preparation gives a clearly named new topic priority.
    """
    from rag.domain_config import load_domain_config
    from rag.query_pipeline import explicit_fields
    text = normalize_text(prompt.replace("đ", "d").replace("Đ", "D"))
    names_new_topic = bool(re.search(
        r"\b(dang ky|khai sinh|khai tu|ket hon|ho kinh doanh|cap lai|cap moi|"
        r"xay dung|chung thuc|chuyen sang|thu tuc khac|hoi ve thu tuc)\b", text))
    # Existing referential follow-ups may contain 'thủ tục này', but no named entity.
    fields = explicit_fields(prompt, load_domain_config())
    followup_words = set("le phi chi tien mat bao nhieu nhieu o dau noi nop dia diem ho so giay to can chuan bi thoi gian lau may ngay lam viec giai quyet trong thi la het ton gom nhung gi nao vay the con cua thu tuc nay do tren vua roi va ca hai tung moi cho toi biet xin hoi muon duoc khong co phai them mot ban chinh sao".split())
    # Do not treat an unrecognized entity name as a property-only follow-up.
    property_only = set(text.split()).issubset(followup_words)
    field_followup = bool(active_titles and fields and property_only and not names_new_topic)
    if field_followup:
        return "; ".join(active_titles) + ". " + prompt.strip(), active_titles
    old_switch = bool(re.search(r"(muốn làm thủ tục khác|chuyển sang|hỏi về thủ tục|xin cấp mới|cấp lại)", prompt.lower()))
    if not active_titles or old_switch:
        return prompt.strip(), []
    return "; ".join(active_titles) + ". " + prompt.strip(), []


async def orchestrate_chat(conversation_id, payload, db):
    if not payload.user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    conversation = await get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await add_message(db, conversation_id, "user", payload.user_prompt)
    context_messages = await get_conversation_history(db, conversation_id)

    if len(context_messages) == 1:
        raw_title = payload.user_prompt.strip()
        new_title = raw_title.capitalize()
        if len(new_title) > 35:
            new_title = new_title[:35] + "..."
        await update_conversation_title(db, conversation_id, new_title)

    # 1. READ STATE DIRECTLY FROM DATABASE
    last_proc_title = conversation.current_procedure or ""
    current_location = conversation.location or ""

    user_prompt_clean = payload.user_prompt.strip().lower()

    # Update location memory if mentioned
    if "tp.hcm" in user_prompt_clean or "hồ chí minh" in user_prompt_clean:
        current_location = "TP.HCM"
        await update_conversation_state(db, conversation_id, location=current_location)

    active_titles = active_procedures(context_messages, last_proc_title)
    from rag.domain_config import load_domain_config
    config = load_domain_config()
    mode = resolve_mode(getattr(payload, "mode", None), config)
    retrieve = mode == "strict" or await needs_retrieval(payload.user_prompt, active_titles, config)
    if retrieve:
        search_query, followup_titles = resolve_search_query(payload.user_prompt, active_titles)
        stream_gen, max_score = await get_legal_response_stream(
            prompt=payload.user_prompt.strip(), search_query=search_query,
            active_procedures=followup_titles,
        )
    else:
        stream_gen = get_friendly_response_stream(payload.user_prompt.strip(), context_messages)
        max_score = None  # No retrieval was performed; never invent a relevance score.

    async def stream_and_save_to_db():
        if max_score == 0.0:
            if should_trigger_mcp(payload.user_prompt):
                ask_msg = "Tôi không tìm thấy thông tin trong dữ liệu địa phương. Bạn có muốn tôi tìm kiếm trực tuyến trên Cổng Dịch vụ công Quốc gia không?"
                await add_message(db, conversation_id, "assistant", ask_msg, [])
                await log_interaction(db, conversation_id, payload.user_prompt, ask_msg, None, False, max_score, "MCP_PROMPT")

                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': ask_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            else:
                nudge_msg = random.choice([
                    "Xin chào! Tôi là trợ lý hướng dẫn thủ tục hành chính. Vui lòng cho tôi biết bạn cần làm thủ tục gì nhé.",
                    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thủ tục hành chính công. Bạn đang cần thực hiện thủ tục nào?"
                ])
                await add_message(db, conversation_id, "assistant", nudge_msg, [])
                await log_interaction(db, conversation_id, payload.user_prompt, nudge_msg, None, False, max_score, "LOW_CONFIDENCE")

                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': nudge_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

        full_ai_answer = ""
        final_sources = []
        extracted_procedure = None

        stream_complete = False
        async for raw_chunk in stream_gen:
            if raw_chunk.strip() == "data: [DONE]":
                stream_complete = True
                continue
            yield raw_chunk
            if raw_chunk.startswith("data: "):
                data_str = raw_chunk.replace("data: ", "").strip()
                if data_str and data_str != "[DONE]":
                    try:
                        parsed = json.loads(data_str)
                        if parsed.get("type") == "metadata":
                            final_sources = parsed.get("sources", [])
                            actual_sources = [s for s in final_sources if isinstance(s, str) and not s.startswith("RAG_")]
                            if actual_sources:
                                extracted_procedure = actual_sources[0]

                        elif parsed.get("type") == "chunk":
                            full_ai_answer += parsed.get("text", "")
                    except json.JSONDecodeError:
                        pass

        db_sources = final_sources.copy()
        if max_score is not None:
            db_sources.append(f"RAG_SCORE:{max_score}")

        if extracted_procedure and stream_complete and is_form_trigger(payload.user_prompt):
            offer_text = "\n\n**Bạn có muốn tải biểu mẫu (file Word) cho thủ tục này không?**"
            full_ai_answer += offer_text
            yield f"data: {json.dumps({'type': 'chunk', 'text': offer_text})}\n\n"

        # 3. SAVE STATE FOR THE NEXT TURN
        if extracted_procedure and stream_complete:
            await update_conversation_state(db, conversation_id, current_procedure=extracted_procedure)

        await add_message(db, conversation_id, "assistant", full_ai_answer, db_sources)
        await log_interaction(db, conversation_id, payload.user_prompt, full_ai_answer, str(final_sources), False, max_score, ("FRIENDLY_DIRECT" if not retrieve else "SUCCESS") if stream_complete else "INCOMPLETE_STREAM")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_and_save_to_db(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )