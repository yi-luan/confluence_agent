from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from services.vector_store import VectorStoreService
from core.exceptions import RAGException
from config.settings import settings
from typing import Dict
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
    
    def ask(self, question, workspace=None):
        print(f"\n開始處理問題: '{question}'")
        
        # 獲取向量存儲
        vectorstore = self.vector_store_service.get_vectorstore(workspace)
        if not vectorstore:
            print("錯誤: 向量存儲為空")
            return {"status": "error", "message": "向量存儲未初始化"}
        
        # 直接使用向量存儲進行檢索
        print("直接使用向量存儲進行檢索...")
        direct_results = vectorstore.similarity_search(question, k=5)
        print(f"直接檢索結果: 找到 {len(direct_results)} 個文檔")
        for i, doc in enumerate(direct_results):
            print(f"文檔 {i+1}:")
            print(f"  內容: {doc.page_content[:100]}...")
            print(f"  來源: {doc.metadata.get('source', 'unknown')}")
            print(f"  標題: {doc.metadata.get('title', 'unknown')}")
        
        try:
            # 如果沒有找到文檔
            if not direct_results:
                return {
                    "status": "success",
                    "data": {
                        "question": question,
                        "answer": "抱歉，我找不到相關的文檔來回答您的問題。",
                        "sources": []
                    }
                }
            
            # 使用找到的文檔構建上下文
            context = "\n\n".join([doc.page_content for doc in direct_results])
            
            # 構建更詳細的提示
            prompt = f"""你是一個專業的技術顧問，請使用繁體中文回答問題。

            以下是參考文檔內容：
            {context}

            使用者問題：{question}

            回答要求：
            1. 必須使用繁體中文回答，不要使用英文
            2. 只使用參考文檔中的資訊來回答
            3. 提供詳細且全面的回答，至少 1000 字
            4. 如果文檔中沒有相關資訊，請直接說「抱歉，在提供的文檔中找不到相關資訊」
            5. 回答要有條理，可以使用標題、項目符號或編號來組織內容
            6. 如果需要引用原文，請將英文內容翻譯成中文
            7. 盡可能提供具體的步驟、方法或範例
            8. 確保回答的準確性和完整性

            請以繁體中文回答："""
            
            # 直接使用 LLM 生成回答
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
        # 提取原始內容
        content = doc.page_content
        
        # 清理內容 - 移除模型生成的前綴和後綴
        cleaned_content = content
        
        # 嘗試提取引號內的實際摘要 (如果存在)
        quote_match = re.search(r'"([^"]+)"', content)
        if quote_match:
            cleaned_content = quote_match.group(1)
        else:
            # 移除常見的前綴
            prefixes = [
                "Here is a concise summary of the text, retaining key information and professional terminology:",
                "Here is a concise summary:",
                "Summary:",
            ]
            for prefix in prefixes:
                if cleaned_content.startswith(prefix):
                    cleaned_content = cleaned_content[len(prefix):].strip()
            
            # 移除常見的後綴
            suffixes = [
                "Summary length: approximately 15% of the original text.",
                "This summary is about 20% of the original text.",
            ]
            for suffix in suffixes:
                if cleaned_content.endswith(suffix):
                    cleaned_content = cleaned_content[:-len(suffix)].strip()
        
        # 提取來源信息
        source = doc.metadata.get('source', 'unknown')
        title = doc.metadata.get('title', '')
        
        # 使用標題作為顯示名稱 (如果有)
        display_name = title if title else source
        
        # 確保返回與 source_model 匹配的字段
        return {
            "content": cleaned_content,
            "summary": cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content,
            "source": display_name
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
                4. 以繁體中文為輸出語言
                """
            else:
                summary_prompt = """請為以下文檔內容生成一個簡短的摘要：
                文檔內容：{content}
                
                請生成100字以內的摘要，確保：
                1. 包含關鍵資訊
                2. 清晰易懂
                3. 突出重點
                4. 以繁體中文為輸出語言
                """
            
            response = self.llm.invoke(summary_prompt.format(content=content))
            print(response.content)
            # 從 AIMessage 中提取文本內容
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"摘要生成失敗: {str(e)}"