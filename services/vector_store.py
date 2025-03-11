from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
import os
from config.settings import settings

class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstores = {}
        # 初始化摘要模型
        self.summarizer = ChatOllama(
            model="llama3:8b",
            temperature=0.3,
            base_url=settings.OLLAMA_BASE_URL,
        )
        self.load_vectorstore()
    
    def _generate_summary(self, text):
        """生成文檔摘要"""
        # 如果文本太長，先進行截斷
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        prompt = f"""
                    請用繁體中文為以下文本生成一個簡潔的摘要，保留關鍵信息和專業術語。
                    摘要應該不超過原文的20%長度，並且保持原文的主要含義。
                    如果原文是英文，請將摘要翻譯成繁體中文。

                    文本內容：
                    {text}

                    請用繁體中文回答：
                """
        
        try:
            response = self.summarizer.invoke(prompt)
            print(response)
            return response.content
        except Exception as e:
            print(f"摘要生成失敗: {str(e)}")
            # 如果摘要失敗，返回原文的前300個字符作為備用
            return text[:300] + "..."
    
    def get_vectorstore(self, workspace=None):
        workspace = workspace or settings.CONFLUENCE_SPACE
        if workspace not in self.vectorstores:
            self.vectorstores[workspace] = self._create_vectorstore(workspace)
        return self.vectorstores[workspace]
    
    def load_vectorstore(self):
        # 載入預設 workspace   
        self.get_vectorstore()
    
    def _create_vectorstore(self, workspace):
        vector_store_path = os.path.join(settings.VECTOR_STORE_PATH, workspace)
        if os.path.exists(vector_store_path):
            return self._load_existing_vectorstore(vector_store_path)
        
        loader = ConfluenceLoader(
            url=settings.CONFLUENCE_URL,
            token=settings.CONFLUENCE_TOKEN,
            space_key=workspace,
        )
        documents = loader.load()
        print("documents quantity:", len(documents))

        # 添加摘要步驟
        print("開始生成文檔摘要...")
        summarized_documents = []
        for i, doc in enumerate(documents):
            print(f"處理文檔 {i+1}/{len(documents)}")
            summary = self._generate_summary(doc.page_content)
            summarized_doc = Document(
                page_content=summary,
                metadata=doc.metadata
            )
            summarized_documents.append(summarized_doc)
        print("摘要生成完成")
        print(summarized_documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        splits = text_splitter.split_documents(summarized_documents)
        print("splits")
        print(splits)
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        print("vectorstore")
        print(vectorstore)
        # 確保目錄存在
        os.makedirs(vector_store_path, exist_ok=True)
        vectorstore.save_local(vector_store_path)
        return vectorstore
        
    def _load_existing_vectorstore(self, path):
        """載入現有的向量存儲"""
        try:
            return FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"載入向量存儲失敗: {str(e)}")
            return None