from typing import Optional
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

class SQLHandler:
    def __init__(self):
        self.llm = ChatOpenAI(model=settings.model_name)
        self._db: Optional[SQLDatabase] = None
        self._query_tool: Optional[QuerySQLDataBaseTool] = None
    
    @property
    def db(self) -> SQLDatabase:
        if not self._db:
            try:
                self._db = SQLDatabase.from_uri(settings.get_database_uri())
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
        return self._db
    
    @property
    def query_tool(self) -> QuerySQLDataBaseTool:
        if not self._query_tool:
            self._query_tool = QuerySQLDataBaseTool(db=self.db)
        return self._query_tool

    def clean_sql_query(self, markdown_query: str) -> str:
        """Clean SQL query from markdown formatting."""
        lines = markdown_query.strip().split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith("```") or line.strip() == "sql":
                continue
            cleaned_lines.append(line)
        cleaned_query = " ".join(cleaned_lines).strip()
        cleaned_query = cleaned_query.replace("`", "")
        if not cleaned_query.strip().endswith(";"):
            cleaned_query += ";"
        return cleaned_query

    def generate_response(self, question: str) -> str:
        """Generate SQL response for a given question."""
        try:
            # Create prompt template
            sql_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a SQLite expert. Create only the SQL query without explanation. "
                    "The query must be simple and direct, avoiding complex joins unless necessary. "
                    "Table info: {table_info}\n"
                    "Return only the top {top_k} results if the query returns multiple rows."
                ),
                ("human", "{input}"),
            ])

            # Create query chain
            write_query = create_sql_query_chain(
                self.llm,
                self.db,
                sql_prompt,
                k=settings.sql_top_k
            )

            # Generate and clean query
            query = write_query.invoke({"question": question})
            clean_query = self.clean_sql_query(query)
            
            # Execute query
            result = self.query_tool.invoke(clean_query)
            
            # Format response
            response = f"""Query executed: 
{clean_query}

Results:
{result}"""
            return response
        except Exception as e:
            logger.error(f"Failed to generate SQL response: {e}")
            raise
