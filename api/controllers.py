from flask import request
from services.rag import RAGService
from services.conversation import ConversationService
from core.exceptions import RAGException
from config.settings import settings

class AskQuestionController:
    def __init__(self):
        self.rag_service = RAGService()
        self.conversation_service = ConversationService(settings.MAX_CONVERSATION_AGE_HOURS)
    
    def post(self):
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                raise RAGException("Missing question parameter", 400)
            
            question = data['question']
            workspace = data.get('workspace', None)
            conversation_id = data.get('conversation_id', None)
            
            # 如果沒有 conversation_id，創建新對話
            if not conversation_id:
                conversation_id = self.conversation_service.create_conversation()
            
            # 保存用戶問題到對話歷史
            self.conversation_service.add_message(conversation_id, "user", question)
            
            # 獲取對話歷史
            history = self.conversation_service.get_history(conversation_id)
            
            # 使用 RAG 服務回答問題
            result = self.rag_service.ask(question, workspace, history)
            
            # 保存 AI 回答到對話歷史
            self.conversation_service.add_message(
                conversation_id, 
                "assistant", 
                result['data']['answer'],
                sources=result['data'].get('sources', [])
            )
            
            # 在回應中加入對話 ID
            result['data']['conversation_id'] = conversation_id
            
            return result
            
        except Exception as e:
            raise RAGException(str(e))

class ConversationController:
    def __init__(self):
        self.conversation_service = ConversationService(settings.MAX_CONVERSATION_AGE_HOURS)
    
    def get_all(self):
        try:
            conversations = self.conversation_service.get_all_conversations()
            return {
                "status": "success",
                "data": {
                    "conversations": conversations
                }
            }
        except Exception as e:
            raise RAGException(str(e))
    
    def get(self, conversation_id):
        try:
            history = self.conversation_service.get_history(conversation_id)
            return {
                "status": "success",
                "data": {
                    "conversation_id": conversation_id,
                    "messages": history
                }
            }
        except ConversationNotFoundError:
            raise RAGException("對話不存在", 404)
        except Exception as e:
            raise RAGException(str(e))