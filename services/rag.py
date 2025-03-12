from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from services.vector_store import VectorStoreService
from core.exceptions import RAGException
from config.settings import settings
from typing import Dict, List
import re

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
                "k": 5,                # 增加返回數量
                "score_threshold": 0.5  # 降低閾值
            }
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
    
    def ask(self, question: str, workspace=None, history: List[dict] = None):
        print(f"\n開始處理問題: '{question}'")
        
        vectorstore = self.vector_store_service.get_vectorstore(workspace)
        if not vectorstore:
            print("錯誤: 向量存儲為空")
            return {"status": "error", "message": "向量存儲未初始化"}
        
        direct_results = vectorstore.similarity_search(question, k=5)
        print(f"直接檢索結果: 找到 {len(direct_results)} 個文檔")
        
        try:
            if not direct_results:
                return {
                    "status": "success",
                    "data": {
                        "question": question,
                        "answer": "抱歉，我找不到相關的文檔來回答您的問題。",
                        "sources": []
                    }
                }
            
            context = "\n\n".join([doc.page_content for doc in direct_results])
            
            # 構建對話歷史字符串
            chat_history = ""
            if history:
                for msg in history[-6:]:  # 只使用最近的 6 條消息
                    chat_history += f"{msg['role']}: {msg['content']}\n"
            
            prompt = f"""你是一個專業的技術顧問，請使用繁體中文回答問題。

            對話歷史：
            {chat_history}

            參考文檔內容：
            {context}

            當前問題：{question}

            回答規則：
            1. 如果問題是關於之前對話的內容（例如：剛剛問了什麼），直接查看對話歷史回答，不要提及文檔內容
            2. 如果是一般問題，則：
               - 只使用參考文檔中的資訊回答
               - 找不到相關資訊時，直接說「抱歉，文檔中沒有相關資訊」
            3. 如果不確定答案，請明確表示不確定，而不是猜測或提供不準確的信息
            4. 直接切入重點，避免贅字
            5. 避免重複使用相同的詞語和句式
            6. 使用精確且多樣的描述方式
            7. 將英文內容翻譯成中文
            8. 使用項目符號列出重點，確保每點表達不同的內容
            9. 如果問題是簡單的「是」或「否」，直接回答「是」或「否」，然後簡要說明
            10. 不要使用「根據你的問題」或類似的開場白，直接回答問題

            請以繁體中文回答："""
            
            response = self.llm.invoke(prompt)
            answer = response.content
            
            return {
                "status": "success",
                "data": {
                    "question": question,
                    "answer": answer,
                    "sources": [self._process_source(doc) for doc in direct_results]
                }
            }
        except Exception as e:
            raise RAGException(str(e))
    
    def _process_source(self, doc) -> Dict:
        # 調試輸出
        print("\n=== 文檔元數據 ===")
        print(f"Metadata: {doc.metadata}")
        
        # 提取原始內容和元數據
        content = doc.page_content
        metadata = doc.metadata
        
        # 構建 Confluence 頁面 URL
        base_url = settings.CONFLUENCE_URL.rstrip('/')
        page_id = metadata.get('page_id', '')
        url = metadata.get('source', '')
        
        # 調試輸出
        print(f"Base URL: {base_url}")
        print(f"Page ID: {page_id}")
        print(f"Final URL: {url}")
        
        # 提取標題
        title = metadata.get('title', '')
        
        # 清理內容（保留原有的清理邏輯）
        cleaned_content = content
        quote_match = re.search(r'"([^"]+)"', content)
        if quote_match:
            cleaned_content = quote_match.group(1)
        
        result = {
            "content": cleaned_content,
            "summary": cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content,
            "source": url or title or "unknown",  # 優先使用 URL
            "title": title
        }
        
        # 調試輸出
        print("\n=== 處理結果 ===")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print("================\n")
        
        return result
    
    def generate_summary(self, content: str, is_code: bool = False) -> str:
        try:
            if is_code:
                summary_prompt = """請為以下程式碼生成一個簡短的摘要：
                程式碼：{content}
                
                請以繁體中文生成100字以內的摘要，說明：
                1. 程式碼的主要功能
                2. 關鍵的實作方法
                3. 重要的依賴或引用
                """
            else:
                summary_prompt = """請為以下文檔內容生成一個簡短的摘要：
                文檔內容：{content}
                
                請以繁體中文生成100字以內的摘要，確保：
                1. 包含關鍵資訊
                2. 清晰易懂
                3. 突出重點
                """
            
            response = self.llm.invoke(summary_prompt.format(content=content))
            print(response.content)
            # 從 AIMessage 中提取文本內容
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"摘要生成失敗: {str(e)}"