from typing import Optional, Any
from swarm import Agent

from config.settings import settings
from utils.logging import get_logger
from ..document_store.vectorstore import VectorStoreHandler
from .base import BaseSwarmAgent

logger = get_logger(__name__)

class RAGAgent(BaseSwarmAgent):
    def __init__(self):
        self.vectorstore_handler = VectorStoreHandler()
        # Initialize the vector store
        self.vectorstore_handler.initialize_vectorstore()
        
        super().__init__(
            name="RAG Agent",
            instructions=(
                "You retrieve and analyze information from academic papers about "
                "transformer architectures and LLM research using the available knowledge base. "
                "If a question is not related to transformer architectures, LLMs, or the research papers "
                "in the knowledge base, use the handle_non_rag_query function to redirect to the coordinator."
            ),
            functions=[self.vectorstore_handler.retrieve_and_generate, self.handle_non_rag_query]
        )
        
    def handle_non_rag_query(self, question: str) -> Any:
        """Handle non-RAG queries by redirecting to appropriate agent."""
        logger.info("Redirecting non-RAG query")
        # Check if the query is database-related
        database_keywords = ["database", "sql", "query", "table", "sales", "albums", "tracks", "artists"]
        if any(keyword in question.lower() for keyword in database_keywords):
            from .sql_agent import SQLAgent
            sql_agent = SQLAgent()
            return sql_agent.handle_query(question)  # Execute the query directly
        # If not database-related, return None to go back to coordinator
        return None
        
    def handle_query(self, query: str) -> Any:
        """Handle an incoming RAG query."""
        try:
            return self.vectorstore_handler.retrieve_and_generate(query)
        except Exception as e:
            error_msg = f"Error handling RAG query: {str(e)}"
            logger.error(error_msg)
            return error_msg
