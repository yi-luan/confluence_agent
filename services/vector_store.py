from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config.settings import settings

class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstores = {}
        self.load_vectorstore()
    
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
        print("documents quantity:", len(documents));

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # 確保目錄存在
        os.makedirs(vector_store_path, exist_ok=True)
        vectorstore.save_local(vector_store_path)
        return vectorstore
    
    def _load_existing_vectorstore(self, path):
        return FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )