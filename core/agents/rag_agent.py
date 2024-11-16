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
        self._sql_agent: Optional[SQLAgent] = None
        # Initialize the vector store
        self.vectorstore_handler.initialize_vectorstore()
        
        def transfer_to_sql() -> Agent:
            """Transfer function to SQL Agent."""
            if not self._sql_agent:
                from .sql_agent import SQLAgent
                self._sql_agent = SQLAgent()
            logger.info("Transferring to SQL Agent")
            return self._sql_agent.agent
        
        super().__init__(
            name="RAG Agent",
            instructions=(
                "You retrieve and analyze information from academic papers about "
                "transformer architectures and LLM research using the available knowledge base. "
                "If a question is not related to transformer architectures, LLMs, or the research papers "
                "in the knowledge base, use the transfer_to_sql function for database queries or "
                "return None to go back to the coordinator."
            ),
            functions=[self.vectorstore_handler.retrieve_and_generate, transfer_to_sql]
        )
        
    def handle_non_rag_query(self, question: str) -> Optional[Agent]:
        """Handle non-RAG queries by redirecting to appropriate agent."""
        logger.info("Redirecting non-RAG query")
        # Check if the query is database-related
        database_keywords = ["database", "sql", "query", "table", "sales", "albums", "tracks", "artists"]
        if any(keyword in question.lower() for keyword in database_keywords):
            if not self._sql_agent:
                from .sql_agent import SQLAgent
                self._sql_agent = SQLAgent()
            return self._sql_agent.agent
        # If not database-related, return None to go back to coordinator
        return None
        
    def handle_query(self, query: str) -> Any:
        """Handle an incoming RAG query."""
        try:
            answer, num_docs, snippets = self.vectorstore_handler.retrieve_and_generate(query)
            
            # Format snippets
            snippet_text = "\n\n".join(
                f"📄 Document {i+1}:\n```\n{snippet}\n```"
                for i, snippet in enumerate(snippets)
            )
            
            return f"{answer}\n\n---\n*Retrieved from {num_docs} document(s):*\n\n{snippet_text}"
        except Exception as e:
            error_msg = f"Error handling RAG query: {str(e)}"
            logger.error(error_msg)
            return error_msg
