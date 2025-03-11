import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import uuid
from core.exceptions import ConversationNotFoundError

class ConversationService:
    def __init__(self, max_age_hours: int):
        self.max_age = timedelta(hours=max_age_hours)
        self.data_dir = "data/conversations"  # 儲存對話的目錄
        
        # 確保目錄存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 從檔案載入現有對話
        self.conversations: Dict[str, List[dict]] = {}
        self.conversation_times: Dict[str, datetime] = {}
        self._load_conversations()
    
    def _load_conversations(self):
        """從檔案載入所有對話"""
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                conversation_id = filename[:-5]  # 移除 .json 副檔名
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.conversations[conversation_id] = data['messages']
                        self.conversation_times[conversation_id] = datetime.fromisoformat(data['last_updated'])
                except Exception as e:
                    print(f"Error loading conversation {conversation_id}: {e}")
    
    def _save_conversation(self, conversation_id: str):
        """將對話儲存到檔案"""
        file_path = os.path.join(self.data_dir, f"{conversation_id}.json")
        data = {
            'messages': self.conversations[conversation_id],
            'last_updated': self.conversation_times[conversation_id].isoformat()
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = []
        self.conversation_times[conversation_id] = datetime.now()
        self._save_conversation(conversation_id)
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str):
        if conversation_id not in self.conversations:
            conversation_id = self.create_conversation()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations[conversation_id].append(message)
        self.conversation_times[conversation_id] = datetime.now()
        self._save_conversation(conversation_id)
        
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
            # 刪除記憶體中的對話
            del self.conversations[cid]
            del self.conversation_times[cid]
            
            # 刪除檔案
            file_path = os.path.join(self.data_dir, f"{cid}.json")
            try:
                os.remove(file_path)
            except OSError:
                pass