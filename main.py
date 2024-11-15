import os
from operator import itemgetter
from typing import List

from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# imported from github repo
from swarm import Agent, Swarm

# Add these imports at the top
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import print as rprint
from rich.spinner import Spinner

load_dotenv()

# LLM selection
llm = ChatOpenAI(model="gpt-4o")

# path to the folder containing documents for RAG
ragPath = "./documents"


def load_documents(folder_path: str) -> List[Document]:
    documents = []
    files = os.listdir(folder_path)
    
    for filename in files:
        if filename.startswith("."):
            continue
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    
    return documents


def setup_vectorstore(folder_path: str):
    documents = load_documents(folder_path)
    if not documents:
        raise ValueError("No documents were loaded from the specified path")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)

    # embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        collection_name="my_collection",
        documents=splits,
        embedding=embedding_function,
        persist_directory="./chroma_db",
    )
    return vectorstore


def clean_sql_query(markdown_query):
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


def retrieve_and_generate(question):
    if retriever is None:
        return "Error: Document retrieval system is not properly initialized."
    
    retrieved_docs = retriever.invoke(question)
    
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer: """
    prompt = ChatPromptTemplate.from_template(template)

    def docs2str(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | docs2str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

# TODO: Add caching for the database queries, look at LangChain docs for that and prompt template libraries.
def sql_response_gen(question):
    try:
        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    except Exception as e:
        return f"Error connecting to database: {e}"

    execute_query = QuerySQLDataBaseTool(db=db)

    # Updated prompt template to include top_k
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

    # Add top_k parameter when creating the chain
    write_query = create_sql_query_chain(
        llm, 
        db, 
        sql_prompt,
        k=5  # This sets the default value for top_k
    )
    
    try:
        # Get the query
        query = write_query.invoke({"question": question})
        clean_query = clean_sql_query(query)
        
        # Execute the query
        result = execute_query.invoke(clean_query)
        
        # Format the response with the actual query for transparency
        response = f"""Query executed: 
```sql
{clean_query}
```

Result: {result}"""
        
        return response
        
    except Exception as e:
        return f"Error executing query: {str(e)}"


def handle_non_sql_query(question):
    print("This appears to be a non-SQL query. Redirecting to coordinator...")
    return coordinator_agent

def handle_non_rag_query(question):
    print("This appears to be unrelated to LLM research. Redirecting to coordinator...")
    return coordinator_agent

def transfer_to_sql():
    print("Handing off to SQL Agent")
    return sql_agent

def transfer_to_rag():
    print("Handing off to RAG Agent")
    return rag_agent

sql_agent = Agent(
    name="SQL Agent",
    instructions=(
        "You handle database queries for the Chinook SQLite database. "
        "This database contains tables related to a digital media store, including "
        "artists, albums, tracks, customers, and sales information. "
        "If a question is not related to the Chinook database, use the handle_non_sql_query function "
        "to redirect to the coordinator."
    ),
    functions=[sql_response_gen, handle_non_sql_query],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions=(
        "You retrieve and analyze information from academic papers about "
        "transformer architectures and LLM research using the available knowledge base. "
        "If a question is not related to transformer architectures, LLMs, or the research papers "
        "in the knowledge base, use the handle_non_rag_query function to redirect to the coordinator."
    ),
    functions=[retrieve_and_generate, handle_non_rag_query],
)

coordinator_agent = Agent(
    name="Coordinator",
    instructions=(
        "Route queries to the appropriate specialized agent:\n"
        "- For questions about the Chinook database or any data related to the "
        "Chinook database, transfer to SQL Agent\n"
        "- For questions about transformer architectures, LLMs, or related research, "
        "transfer to RAG Agent\n"
        "- For general questions or clarifications, provide a direct response\n"
        "Always explain your routing decision to the user."
    ),
    functions=[transfer_to_sql, transfer_to_rag],
)

print("\n=== Initializing RAG System ===")
try:
    print(f"Attempting to load documents from: {os.path.abspath(ragPath)}")
    vectorstore = setup_vectorstore(ragPath)
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 5,
        }
    )
    print("Successfully initialized vector store and retriever")
except Exception as e:
    print(f"Error setting up vector store: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
    raise


def pretty_print_messages(messages):
    console = Console()
    
    # Group messages by conversation turn
    current_turn = []
    
    for message in messages:
        if message.get("content") is None:
            continue
            
        sender = message.get("sender", message.get("role", "unknown"))
        content = message["content"]
        
        # Start new turn on user message
        if sender == "user":
            if current_turn:
                # Print previous turn with divider
                console.print("─" * 80)
                for turn_message in current_turn:
                    print_message(turn_message, console)
                current_turn = []
            
            # Add user message to new turn
            current_turn.append({"sender": sender, "content": content})
        else:
            current_turn.append({"sender": sender, "content": content})
    
    # Print final turn
    if current_turn:
        console.print("─" * 80)
        for turn_message in current_turn:
            print_message(turn_message, console)

def print_message(message, console):
    sender = message["sender"]
    content = message["content"]
    
    if sender == "user":
        style = "bold blue"
        title = "📝 User"
    elif sender == "Coordinator":
        style = "bold yellow"
        title = "🎯 Coordinator"
    elif sender == "SQL Agent":
        style = "bold green"
        title = "💾 SQL Agent"
    elif sender == "RAG Agent":
        style = "bold magenta"
        title = "📚 RAG Agent"
    else:
        style = "bold white"
        title = sender
        
    panel = Panel(
        Markdown(content) if sender != "user" else content,
        title=title,
        style=style,
        border_style=style,
        padding=(1, 2)
    )
    console.print(panel)

def main():
    client = Swarm()
    messages = []
    agent = coordinator_agent

    console = Console()
    console.print("[bold magenta]Welcome to the Swarm CLI![/bold magenta]")
    console.print("[bold]Type 'quit' to exit[/bold]")

    while True:
        user_input = console.input("\n[bold blue]Enter your question:[/bold blue] ").strip()

        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        
        with console.status("[bold green]Processing query...", spinner="dots"):
            response = client.run(agent=agent, messages=messages)
            messages = response.messages
            agent = response.agent
        
        pretty_print_messages(messages)


if __name__ == "__main__":
    main()


#TODO Refactor the code to be more modular and reusable.
# - Proper modules for each agent.
# - Proper class structure.
# - Proper dependency injection
# - Unit testing for each module.
#TODO Logging instead of print statements.
#TODO Error handling.
#TODO Input validation.
#TODO Specialize the agents for different types of documents/stores.
#TODO Tool to summarize the documents? Consider adding summarization to the document loader.

#NOTE: Nice to haves:
# UX/UI:
# - Add a loading animation when the system is processing the user's request.
# - Add a way to view the documents that are being used for RAG.
# - Add a way to view the chat history.
# - Consider Next.js or Streamlit for a more interactive UI.

# Performance:
# - Caching for frequently accessed documents.
# - Adjusting batches and chunks based on the size of the documents, 
# trade-off of accuracy and the LLM model.
# - Optimize vector store queries, looking at LangChain docs.
