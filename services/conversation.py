from datetime import datetime, timedelta
from typing import Dict, List
import uuid
from core.exceptions import ConversationNotFoundError

class ConversationService:
    def __init__(self, max_age_hours: int):
        self.conversations: Dict[str, List[dict]] = {}
        self.conversation_times: Dict[str, datetime] = {}
        self.max_age = timedelta(hours=max_age_hours)
    
    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = []
        self.conversation_times[conversation_id] = datetime.now()
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str):
        if conversation_id not in self.conversations:
            conversation_id = self.create_conversation()
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        self.conversation_times[conversation_id] = datetime.now()
        
        return conversation_id
    
    def get_history(self, conversation_id: str) -> List[dict]:
        self.cleanup_old_conversations()
        if conversation_id not in self.conversations:
            raise ConversationNotFoundError()
        return self.conversations[conversation_id]
    
    def cleanup_old_conversations(self):
        now = datetime.now()
        expired = [
            cid for cid, time in self.conversation_times.items()
            if now - time > self.max_age
        ]
        for cid in expired:
            del self.conversations[cid]
            del self.conversation_times[cid]