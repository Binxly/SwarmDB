from typing import Optional, Any
from swarm import Agent

from config.settings import settings
from utils.logging import get_logger
from ..sql.handler import SQLHandler
from .base import BaseSwarmAgent

logger = get_logger(__name__)

class SQLAgent(BaseSwarmAgent):
    def __init__(self):
        self.sql_handler = SQLHandler()
        
        super().__init__(
            name="SQL Agent",
            instructions=(
                "You handle database queries for the Chinook SQLite database. "
                "This database contains tables related to a digital media store, including "
                "artists, albums, tracks, customers, and sales information. "
                "If a question is not related to the Chinook database, use the handle_non_sql_query "
                "function to redirect to the coordinator."
            ),
            functions=[self.sql_handler.generate_response, self.handle_non_sql_query]
        )
        
    def handle_non_sql_query(self, question: str) -> Optional[BaseSwarmAgent]:
        """Handle non-SQL queries by redirecting to coordinator."""
        logger.info("Redirecting non-SQL query to coordinator")
        # Return None to indicate redirect to coordinator
        return None
        
    def handle_query(self, query: str) -> Any:
        """Handle an incoming SQL query."""
        try:
            # For database-related questions, generate SQL response
            if any(keyword in query.lower() for keyword in ["database", "sql", "albums", "tracks", "sales"]):
                return self.sql_handler.generate_response(query)
            # For non-database questions, redirect to coordinator
            else:
                logger.info("Redirecting non-SQL query to coordinator")
                return None
        except Exception as e:
            logger.error(f"Error handling SQL query: {e}")
            return f"Error: {str(e)}"
