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


def sql_response_gen(question):
    try:
        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    except Exception as e:
        return f"Error connecting to database: {e}"

    execute_query = QuerySQLDataBaseTool(db=db)

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a SQLite expert. Given an input question, create a "
                "syntactically correct SQL query to run. Unless otherwise "
                "specified.\n\nHere is the relevant table info: {table_info}"
                "\n\n Use max {top_k} rows",
            ),
            ("human", "{input}"),
        ]
    )

    write_query = create_sql_query_chain(llm, db, sql_prompt)
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    chain = (
        RunnablePassthrough.assign(
            query=write_query | RunnableLambda(clean_sql_query)
        ).assign(result=itemgetter("query") | execute_query)
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"question": question})


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
        "artists, albums, tracks, customers, and sales information."
    ),
    functions=[sql_response_gen],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions=(
        "You retrieve and analyze information from academic papers about "
        "transformer architectures and LLM research using the available knowledge base."
    ),
    functions=[retrieve_and_generate],
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


def main():
    client = Swarm()

    print("Welcome to the Swarm CLI!")
    print("Type 'quit' to exit")

    while True:
        user_input = input("\nEnter your question: ").strip()

        if user_input.lower() == "quit":
            break

        messages = [{"role": "user", "content": user_input}]
        response = client.run(agent=coordinator_agent, messages=messages)

        if isinstance(response, Agent):
            selected_agent = response
            result = selected_agent.functions[0](user_input)
            print(f"Response: {result}")
        else:
            print(f"Response: {response.messages[-1]['content']}")


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
# - Optimize vector store queries.
