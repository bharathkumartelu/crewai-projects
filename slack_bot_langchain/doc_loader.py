from langchain_community.document_loaders import GoogleDriveLoader, ConfluenceLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_google_drive_docs():
    """Load documents from Google Drive."""
    try:
        gdrive_loader = GoogleDriveLoader(
            folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
            service_account_key=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
            file_types=("document", "sheet", "pdf", "presentation"),
            recursive=True
        )
        print(f"Loaded Google Drive folder ID: {os.getenv('GOOGLE_DRIVE_FOLDER_ID')}")
        return gdrive_loader.load()
    except Exception as e:
        print(f"Error loading Google Drive documents: {e}")
        return []

def load_confluence_docs():
    """Load documents from Confluence."""
    try:
        confluence_loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USER"),
            api_key=os.getenv("CONFLUENCE_API_TOKEN"),
            space_key=os.getenv("CONFLUENCE_SPACE"),
            max_pages=2000
        )
        return confluence_loader.load()
    except Exception as e:
        print(f"Error loading Confluence documents: {e}")
        return []

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """Create and persist a vector store with the documents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./vector_db"
    )
    vector_store.add_documents(documents=documents)
    print("Vector store created and documents added.")
    return vector_store

def build_knowledge_base():
    """Build the knowledge base by loading, processing, and storing documents."""
    print("Starting to build the knowledge base...")

    # Load documents
    drive_docs = load_google_drive_docs()
    confluence_docs = load_confluence_docs()

    # Combine documents
    all_docs = drive_docs + confluence_docs

    if not all_docs:
        print("No documents loaded. Exiting.")
        return

    # Split documents into chunks
    all_splits = split_documents(all_docs)

    # Create vector store and add documents
    create_vector_store(all_splits)

    print("Knowledge base built successfully.")

if __name__ == "__main__":
    build_knowledge_base()
