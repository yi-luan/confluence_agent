# core/exceptions.py
class RAGException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DocumentNotFoundError(RAGException):
    def __init__(self, message: str = "沒有找到相關文檔"):
        super().__init__(message, 404)

class ConversationNotFoundError(RAGException):
    def __init__(self, message: str = "對話不存在"):
        super().__init__(message, 404)