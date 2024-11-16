from typing import List
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    def __init__(self, folder_path: str = None):
        self.folder_path = folder_path or settings.documents_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader
        }
        
        ext = os.path.splitext(file_path)[1]
        if ext not in loaders:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
            
        try:
            return loaders[ext](file_path).load()
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.folder_path):
            logger.error(f"Documents path does not exist: {self.folder_path}")
            return []
        
        documents = []
        for filename in os.listdir(self.folder_path):
            if not filename.startswith("."):
                documents.extend(self._load_single_document(
                    os.path.join(self.folder_path, filename)
                ))
        
        if not documents:
            logger.warning("No documents were loaded from the specified path")
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
