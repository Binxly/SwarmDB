from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    model_config = {
        "extra": "allow",
        "protected_namespaces": ('settings_',),
        "env_file": ".env"
    }
    
    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = "gpt-4o-mini"
    
    # Path Settings
    documents_path: str = "./data/documents"
    vector_store_path: str = "./data/vector_stores/chroma_db"
    database_path: str = "./data/databases/Chinook.db"
    
    # Vector Store Settings
    chunk_size: int = 4000
    chunk_overlap: int = 500
    embedding_model: str = "all-MiniLM-L6-v2"
    retriever_k: int = 4
    
    # SQL Settings
    sql_top_k: int = 5
    
    # Logging Settings
    log_format: str = " %(name)s - %(message)s"
    log_level: str = "INFO"
    
    # Add these fields to your Settings class
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = ""
    tokenizers_parallelism: bool = False
    log_to_console: bool = False

    def get_database_uri(self) -> str:
        """Get the SQLite database URI."""
        return f"sqlite:///{self.database_path}"

# Create a global settings instance
settings = Settings()
