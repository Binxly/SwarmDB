from typing import List, Optional
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    def __init__(self, folder_path: Optional[str] = None):
        self.folder_path = folder_path or settings.documents_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file extension."""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        
        try:
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []

    def load_documents(self) -> List[Document]:
        """Load all supported documents from the specified folder."""
        documents = []
        
        if not os.path.exists(self.folder_path):
            logger.error(f"Documents path does not exist: {self.folder_path}")
            return documents
        
        for filename in os.listdir(self.folder_path):
            if filename.startswith("."):
                continue
                
            file_path = os.path.join(self.folder_path, filename)
            docs = self._load_single_document(file_path)
            documents.extend(docs)
        
        if not documents:
            logger.warning("No documents were loaded from the specified path")
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
