"""
LEGAL RAG - CONTEXT MANAGER / PHASE 7  (sơ đồ 2.2)

Giữ ngữ cảnh hội thoại để hiểu các câu nối tiếp:

    "khai sinh cần giấy tờ gì"   -> hồ sơ khai sinh
    "còn lệ phí thì sao?"        -> lệ phí khai sinh        (mượn THỦ TỤC câu trước)
    "thế khai tử thì sao?"       -> lệ phí khai tử          (mượn INTENT câu trước)
    "vậy nộp ở đâu?"             -> nơi nộp khai tử

    "lệ phí bao nhiêu"           -> hỏi lại: thủ tục nào?
    "khai sinh"                  -> lệ phí khai sinh        (hoàn tất câu hỏi dang dở)

    "xin cấp lại giấy khai sinh" -> hỏi lại: chỉ có thủ tục đăng ký khai sinh, hỏi thủ tục này?
    "có"                         -> tổng quan thủ tục khai sinh

Ngữ cảnh theo sơ đồ: conversation_summary, recent_messages, structured_context,
long_term_memory, domain_profile (xem to_dict()).

Chỉ dùng luật (không LLM). Planner vẫn không có trạng thái, mọi trạng thái nằm ở đây.
"""
from __future__ import annotations

import re

from .semantic_planner import find_qualifier_mismatch, normalize_text

CONTEXT_VERSION = "v1-2026-10-04"

MAX_RECENT_MESSAGES = 6
MAX_CONTEXT_PROCEDURES = 3
INTENT_TTL_TURNS = 3   # intent câu trước "hết hạn" sau bấy nhiêu lượt không nhắc tới

# "còn ... ", "thế ...", "... thì sao"
FOLLOWUP_START = re.compile(
    r"^(?:còn|thế còn|vậy còn|thế thì|vậy thì|thế|vậy|rồi|à|ừ)\b"
)
FOLLOWUP_END = re.compile(
    r"\b(?:thì sao|thế nào|như thế nào|ra sao|như nào|sao)$"
)

# Chỉ nhận là xác nhận khi KHỚP NGUYÊN CÂU và đang có câu hỏi chờ xác nhận,
# để "có mất tiền không" không bị hiểu nhầm thành "có".
AFFIRM = {
    "có", "có ạ", "có chứ", "có nhé", "ừ", "ừm", "uh", "đúng", "đúng rồi",
    "đúng vậy", "đúng ạ", "phải", "phải rồi", "ok", "oke", "vâng", "vâng ạ",
    "dạ", "dạ có", "dạ vâng", "được", "được rồi", "đồng ý",
}
DENY = {
    "không", "không ạ", "không phải", "sai", "không đúng", "thôi",
    "không cần", "thôi không", "hủy",
}

INTENT_LABEL = {
    "required_documents": "hồ sơ",
    "fee": "lệ phí",
    "location": "nơi nộp",
    "processing_time": "thời gian giải quyết",
    "legal_basis": "căn cứ pháp lý",
    "forms": "biểu mẫu",
    "general_information": "thông tin chung",
}


class ContextManager:

    def __init__(self, domain_profile: dict | None = None) -> None:
        self.domain_profile = domain_profile or {}
        self.reset()

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.turn = 0
        self.recent_messages: list[dict] = []
        self.structured_context: dict = {"procedures": [], "intents": []}
        self.intents_age = 0
        self.pending_intents: list[str] = []
        self.pending_confirmation: dict | None = None
        self.long_term_memory: dict = {}
        self.conversation_summary = ""

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def is_followup(normalized_query: str) -> bool:
        return bool(
            FOLLOWUP_START.search(normalized_query)
            or FOLLOWUP_END.search(normalized_query)
        )

    def _active_intents(self) -> list[str]:
        if self.intents_age > INTENT_TTL_TURNS:
            return []
        return list(self.structured_context["intents"])

    @staticmethod
    def _with_note(task: dict, note: str) -> dict:
        existing = task.get("context_note")
        task["context_note"] = f"{existing}; {note}" if existing else note
        task["from_context"] = True
        return task

    def _apply_procedure(self, task: dict, procedure: dict) -> dict:
        """Gắn một thủ tục lấy từ ngữ cảnh vào task chưa có thủ tục."""
        new_task = dict(task)
        name = procedure["procedure_name"]
        clause = task.get("clause") or task.get("query", "")
        mismatch = find_qualifier_mismatch(clause, name)

        new_task.update(
            {
                "procedure": name,
                "procedure_id": procedure["procedure_id"],
                "procedure_score": 1.0,
                "domain": procedure.get("domain"),
                "candidates": [],
                "ambiguous": False,
                "qualifier_mismatch": mismatch,
                "should_retrieve": True,
                "status": "variant_mismatch" if mismatch else "ready",
                "query": f"{name}. {clause}",
            }
        )

        return self._with_note(
            new_task,
            f"Hiểu theo ngữ cảnh: thủ tục '{name}' (từ câu trước)",
        )

    @staticmethod
    def _blank_task(
        procedure: dict,
        intent: str,
        clause: str,
        note: str,
    ) -> dict:
        name = procedure["procedure_name"]
        return {
            "query": f"{name}. {clause}".strip(),
            "clause": clause or name,
            "normalized_query": normalize_text(f"{name} {clause}"),
            "intent": intent,
            "intent_score": 1.0,
            "procedure": name,
            "procedure_id": procedure["procedure_id"],
            "procedure_score": 1.0,
            "domain": procedure.get("domain"),
            "candidates": [],
            "ambiguous": False,
            "qualifier_mismatch": [],
            "intent_explicit": intent != "general_information",
            "has_subject": True,
            "should_retrieve": True,
            "status": "ready",
            "from_context": True,
            "context_note": note,
        }

    # ------------------------------------------------------------------
    # 1) XÁC NHẬN / TỪ CHỐI câu hỏi dang dở
    # ------------------------------------------------------------------

    def try_confirmation(self, query: str):
        """
        Trả về ("affirm", tasks) / ("deny", None) / None.
        Chỉ hoạt động khi lượt trước đang chờ xác nhận biến thể thủ tục.
        """
        pending = self.pending_confirmation

        if not pending:
            return None

        normalized = normalize_text(query)

        if normalized in AFFIRM:
            intents = pending.get("intents") or ["general_information"]

            tasks = [
                self._blank_task(
                    pending["procedure"],
                    intent,
                    pending.get("clause", ""),
                    "Hiểu theo ngữ cảnh: bạn đồng ý hỏi về thủ tục "
                    f"'{pending['procedure']['procedure_name']}'",
                )
                for intent in intents
            ]
            return "affirm", tasks

        if normalized in DENY:
            return "deny", None

        return None

    # ------------------------------------------------------------------
    # 2) GIẢI QUYẾT câu nối tiếp
    # ------------------------------------------------------------------

    def resolve(self, tasks: list[dict], query: str) -> list[dict]:
        normalized = normalize_text(query)
        followup = self.is_followup(normalized)
        procedures = self.structured_context["procedures"][:MAX_CONTEXT_PROCEDURES]

        # Luật 1: task không có thủ tục VÀ không có chủ đề riêng -> mượn thủ tục
        step1: list[dict] = []

        for task in tasks:

            if (
                task.get("procedure_id") is None
                and not task.get("has_subject", True)
                and procedures
            ):
                for procedure in procedures:
                    step1.append(self._apply_procedure(task, procedure))
                continue

            step1.append(task)

        # Luật 2: task có thủ tục nhưng không nêu intent -> mượn intent
        step2: list[dict] = []

        for task in step1:

            if task.get("procedure_id") and not task.get("intent_explicit", True):

                intents: list[str] = []
                reason = ""

                if self.pending_intents:
                    intents = list(self.pending_intents)
                    reason = "hoàn tất câu hỏi dang dở ở lượt trước"
                elif followup and self._active_intents():
                    intents = self._active_intents()
                    reason = "cùng nội dung hỏi như câu trước"

                if intents:
                    for intent in intents:
                        new_task = dict(task)
                        new_task.update(
                            {
                                "intent": intent,
                                "intent_score": 1.0,
                                "intent_explicit": True,
                            }
                        )
                        step2.append(
                            self._with_note(
                                new_task,
                                f"Hiểu theo ngữ cảnh: {INTENT_LABEL.get(intent, intent)}"
                                f" ({reason})",
                            )
                        )
                    continue

            step2.append(task)

        # Bỏ trùng (thủ tục, intent)
        seen: set[tuple] = set()
        result: list[dict] = []

        for task in step2:
            key = (task.get("procedure_id"), task["intent"])
            if key in seen:
                continue
            seen.add(key)
            result.append(task)

        return result

    # ------------------------------------------------------------------
    # 3) CẬP NHẬT sau mỗi lượt
    # ------------------------------------------------------------------

    def update(
        self,
        original_query: str,
        tasks: list[dict],
        bundles: list[dict],
    ) -> None:

        self.turn += 1

        self.recent_messages.append(
            {"turn": self.turn, "role": "user", "text": original_query}
        )
        self.recent_messages = self.recent_messages[-MAX_RECENT_MESSAGES:]

        procedures: list[dict] = []
        seen: set[str] = set()
        explicit_intents: list[str] = []

        pending_intents: list[str] = []
        pending_confirmation = None

        for task, bundle in zip(tasks, bundles):

            needs_clarification = bundle.get("status") == "needs_clarification"
            explicit = (
                task.get("intent_explicit")
                and task["intent"] != "general_information"
            )

            # Thủ tục đang bị hỏi lại thì chưa chắc đúng -> không lưu làm ngữ cảnh
            if task.get("procedure_id") and not needs_clarification:
                if task["procedure_id"] not in seen:
                    seen.add(task["procedure_id"])
                    procedures.append(
                        {
                            "procedure_id": task["procedure_id"],
                            "procedure_name": task["procedure"],
                            "domain": task.get("domain"),
                        }
                    )

            if explicit and task["intent"] not in explicit_intents:
                explicit_intents.append(task["intent"])

            if needs_clarification:

                if explicit and task["intent"] not in pending_intents:
                    pending_intents.append(task["intent"])

                if task.get("qualifier_mismatch") and task.get("procedure_id"):
                    pending_confirmation = {
                        "procedure": {
                            "procedure_id": task["procedure_id"],
                            "procedure_name": task["procedure"],
                            "domain": task.get("domain"),
                        },
                        "intents": [task["intent"]] if explicit else [],
                        "clause": task.get("clause", ""),
                    }

        # Người dùng nói về một chủ đề không nhận ra được thủ tục nào
        # ("đăng ký hộ khẩu"): bỏ ngữ cảnh cũ để câu sau ("có mất tiền không")
        # không bị gán nhầm sang thủ tục trước đó.
        unknown_topic = any(
            task.get("procedure_id") is None and task.get("has_subject", True)
            for task in tasks
        )

        if procedures:
            self.structured_context["procedures"] = procedures[:MAX_CONTEXT_PROCEDURES]
        elif unknown_topic:
            self.structured_context["procedures"] = []

        if explicit_intents:
            self.structured_context["intents"] = explicit_intents
            self.intents_age = 0
        else:
            self.intents_age += 1

        self.pending_intents = pending_intents
        self.pending_confirmation = pending_confirmation
        self.conversation_summary = self._summarize()

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    def _summarize(self) -> str:
        names = [p["procedure_name"] for p in self.structured_context["procedures"]]
        intents = [
            INTENT_LABEL.get(i, i) for i in self._active_intents()
        ]

        if not names:
            return ""

        text = "Đang hỏi về: " + "; ".join(names)

        if intents:
            text += " (" + ", ".join(intents) + ")"

        return text

    def to_dict(self) -> dict:
        """Context Object theo sơ đồ 2.2."""
        return {
            "conversation_summary": self.conversation_summary,
            "recent_messages": list(self.recent_messages),
            "structured_context": {
                "procedures": list(self.structured_context["procedures"]),
                "intents": self._active_intents(),
                "pending_intents": list(self.pending_intents),
                "pending_confirmation": self.pending_confirmation,
            },
            "long_term_memory": dict(self.long_term_memory),
            "domain_profile": {
                "domain": self.domain_profile.get("domain"),
                "entity": self.domain_profile.get("entity"),
            },
        }
