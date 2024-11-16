from typing import Callable, List, Optional, Any
from swarm import Agent
from utils.logging import get_logger
from .base import BaseSwarmAgent
from .rag_agent import RAGAgent
from .sql_agent import SQLAgent

logger = get_logger(__name__)

class CoordinatorAgent(BaseSwarmAgent):
    def __init__(self):
        super().__init__(
            name="Coordinator",
            instructions=(
                "Route queries to the appropriate specialized agent:\n"
                "- For questions about transformer architectures, LLMs, or related research, "
                "transfer to RAG Agent\n"
                "- For questions about databases, sales data, albums, artists, or any SQL queries, "
                "the RAG Agent will transfer them to SQL Agent\n"
                "- For general questions or clarifications, provide a direct response\n"
                "Always explain your routing decision to the user."
            ),
            functions=[]  # Will be updated after other agents are initialized
        )
        self._sql_agent: Optional[SQLAgent] = None
        self._rag_agent: Optional[RAGAgent] = None
        
    def set_transfer_functions(self, sql_agent: SQLAgent, rag_agent: RAGAgent) -> None:
        """Set up transfer functions with references to other agents."""
        self._sql_agent = sql_agent
        self._rag_agent = rag_agent
        
        def transfer_to_sql() -> Agent:
            logger.info("Transferring to SQL Agent")
            return self._sql_agent.agent
            
        def transfer_to_rag() -> Agent:
            logger.info("Transferring to RAG Agent")
            return self._rag_agent.agent
            
        self.update_functions([transfer_to_sql, transfer_to_rag])
        
    def handle_query(self, query: str) -> Any:
        """Handle incoming queries by routing to appropriate agent."""
        if not self._sql_agent or not self._rag_agent:
            error_msg = "Coordinator not properly initialized with agents"
            logger.error(error_msg)
            return error_msg
            
        return self._agent(query)

coordinator_agent = CoordinatorAgent()
