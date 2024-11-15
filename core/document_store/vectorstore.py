from typing import Optional, List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logging import get_logger
from .loader import DocumentLoader

logger = get_logger(__name__)

class VectorStoreHandler:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.loader = DocumentLoader()
        self.vectorstore: Optional[Chroma] = None
        self.llm = ChatOpenAI(model=settings.model_name)
        self._retriever = None
        
    def initialize_vectorstore(self) -> None:
        """Initialize or load the vector store."""
        try:
            documents = self.loader.load_documents()
            if not documents:
                raise ValueError("No documents were loaded")
            
            splits = self.loader.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                collection_name="my_collection",
                documents=splits,
                embedding=self.embedding_function,
                persist_directory=settings.vector_store_path,
            )
            
            self._retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retriever_k}
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    @property
    def retriever(self):
        """Get the retriever, initializing if necessary."""
        if self._retriever is None:
            self.initialize_vectorstore()
        return self._retriever

    def _docs_to_string(self, docs: List[Document]) -> str:
        """Convert a list of documents to a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_and_generate(self, question: str) -> str:
        """Retrieve relevant documents and generate an answer."""
        try:
            if self.retriever is None:
                return "Error: Document retrieval system is not properly initialized."
            
            template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            Answer: """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {
                    "context": self.retriever | self._docs_to_string, 
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            return rag_chain.invoke(question)
            
        except Exception as e:
            error_msg = f"Error in retrieve_and_generate: {str(e)}"
            logger.error(error_msg)
            return error_msg
