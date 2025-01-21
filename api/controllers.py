from flask import request
from services.rag import RAGService
from core.exceptions import RAGException

class AskQuestionController:
    def __init__(self):
        self.rag_service = RAGService()
    
    def post(self):
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                raise RAGException("Missing question parameter", 400)
            
            # 取得 workspace，如果沒有提供則使用預設值
            workspace = data.get('workspace', None)
            
            return self.rag_service.ask(data['question'], workspace)
            
        except Exception as e:
            raise RAGException(str(e))