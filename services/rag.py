from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from services.vector_store import VectorStoreService
from core.exceptions import RAGException
from config.settings import settings
from typing import Dict

class RAGService:
    def __init__(self):
        self.vector_store_service = VectorStoreService()
        self.llm = ChatOllama(
            model="llama3:8b",  
            temperature=0.7,
            base_url=settings.OLLAMA_BASE_URL,
            num_ctx=2048,    
            num_thread=4,  
            top_k=10,              
            top_p=0.5,         
        )
        
    def _setup_qa_chain(self, workspace=None):
        prompt_template = """你是一個專業的技術顧問，請使用繁體中文回答問題。

        以下是參考文檔內容：
        {context}

        使用者問題：{question}

        回答要求：
        1. 必須使用繁體中文回答，不要使用英文
        2. 只使用參考文檔中的資訊來回答
        3. 如果文檔中沒有相關資訊，請直接說「抱歉，在提供的文檔中找不到相關資訊」
        4. 回答要簡潔、準確、有條理
        5. 如果需要引用原文，請將英文內容翻譯成中文

        請以繁體中文回答："""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain_type_kwargs = {
            "prompt": PROMPT,
            "verbose": True,
            "document_separator": "\n\n",
        }
        
        retriever = self.vector_store_service.get_vectorstore(workspace).as_retriever(
            search_kwargs={
                "k": 3,                # 減少檢索文檔數量，提高相關性
                "score_threshold": 0.7  # 提高相似度閾值
            }
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
    
    def ask(self, question: str, workspace: str = None) -> Dict:
        try:
            self.qa_chain = self._setup_qa_chain(workspace)
            result = self.qa_chain.invoke({"query": question})
            # 檢查是否有找到文檔
            if not result.get('source_documents'):
                return {
                    "status": "success",
                    "data": {
                        "question": question,
                        "answer": "抱歉，我找不到相關的文檔來回答您的問題。",
                        "sources": []
                    }
                }
            
            return {
                "status": "success",
                "data": {
                    "question": question,
                    "answer": result['result'],
                    "sources": [self._process_source(doc) for doc in result['source_documents']]
                }
            }
        except Exception as e:
            raise RAGException(str(e))
    
    def _process_source(self, doc) -> Dict:
        source_path = doc.metadata.get('source', '').lower()
        is_code = any(source_path.endswith(ext) for ext in [
            '.py', '.js', '.java', '.cpp', '.h', '.cs', 
            '.go', '.rs', '.php', '.rb', '.tsx', '.jsx'
        ])
        return {
            "content": doc.page_content[:200],
            "summary": self.generate_summary(doc.page_content, is_code),
            "source": doc.metadata.get('source', 'unknown'),
            "type": "code" if is_code else "document"
        }
    
    def generate_summary(self, content: str, is_code: bool = False) -> str:
        try:
            if is_code:
                summary_prompt = """請為以下程式碼生成一個簡短的摘要：
                程式碼：{content}
                
                請生成100字以內的摘要，說明：
                1. 程式碼的主要功能
                2. 關鍵的實作方法
                3. 重要的依賴或引用
                """
            else:
                summary_prompt = """請為以下文檔內容生成一個簡短的摘要：
                文檔內容：{content}
                
                請生成100字以內的摘要，確保：
                1. 包含關鍵資訊
                2. 清晰易懂
                3. 突出重點
                """
            
            response = self.llm.invoke(summary_prompt.format(content=content))
            # 從 AIMessage 中提取文本內容
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"摘要生成失敗: {str(e)}"