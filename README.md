# Swarm Testing Platform

A lightweight platform for prototyping and testing swarm-based routing between specialized agents. Built with Python and the OpenAI Swarm library.

## What's This?

This is my sandbox for experimenting with agent routing patterns. It uses a coordinator agent to direct queries to specialized agents (SQL and RAG) based on the query type. Those agents can then call back to the coordinator or themselves,for more complex queries if it's needed. Nothing fancy.

## Key Features

- CLI interface with pretty formatting
- Coordinator-based routing between specialized agents
- SQL Agent for database queries (using the SQLite toy DB, Chinook.db)
- RAG Agent for document retrieval and analysis
- TODO: Basic test coverage with pytest

## Quick Start

1. Clone the repo
2. Set up your environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, fill in your OpenAI API key, and optionally update the LangChain API key for tracing:
   ```
   OPENAI_API_KEY= <YOUR_KEY_HERE>
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY= <YOUR_KEY_HERE>
   LANGCHAIN_PROJECT=openai-swarm
   TOKENIZERS_PARALLELISM=false
   ```

4. Run the CLI:
   ```bash
   python main.py
   ```

## Project Structure

- `core/`: Agent implementations and core logic
- `interfaces/`: CLI and other interfaces (TODO)
- `tests/`: Test suite (TODO)
- `config/`: Configuration and settings
- `utils/`: Shared utilities

## Notes

I don't really plan on turning this into something production-ready - I'm mainly using it to explore different routing patterns and agent interactions. Feel free to use it as a starting point for your own experiments or Swarm projects.

## License

MIT