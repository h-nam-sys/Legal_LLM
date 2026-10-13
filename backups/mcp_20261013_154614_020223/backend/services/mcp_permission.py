"""Bind a search approval to a live offer and serialize duplicate clicks."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import HTTPException


def authorize_pending(conversation, current_user_id, owner_id, offer_message_id):
    if owner_id != current_user_id:
        raise HTTPException(403, 'Bạn không có quyền truy cập cuộc trò chuyện này.')
    pending = (conversation.checklist or {}).get('_mcp_pending')
    domain = (conversation.checklist or {}).get('_domain_state', {}).get('name')
    if not conversation.needs_search_permission or not isinstance(pending, dict) or pending.get('domain') != domain or not pending.get('query') or pending.get('message_id') != offer_message_id:
        raise HTTPException(409, 'Không có đề nghị tìm kiếm đang chờ. Hãy gửi câu hỏi mới trước.')
    return dict(pending)

class SearchLocks:
    def __init__(self):
        self.entries = {}

    @asynccontextmanager
    async def hold(self, conversation_id):
        entry = self.entries.setdefault(conversation_id, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if not entry[1]:
                self.entries.pop(conversation_id, None)

search_locks = SearchLocks()
