from langchain.document_loaders import GoogleDriveLoader, ConfluenceLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

def build_knowledge_base():
    # Load Google Drive
    gdrive_loader = GoogleDriveLoader(
        folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
        service_account_key=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    )
    drive_docs = gdrive_loader.load()
    
    # Load Confluence
    confluence_loader = ConfluenceLoader(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USER"),
        api_key=os.getenv("CONFLUENCE_API_TOKEN")
    )
    confluence_docs = confluence_loader.load(space_key=os.getenv("CONFLUENCE_SPACE"))
    
    # Combine and vectorize
    all_docs = drive_docs + confluence_docs
    Chroma.from_documents(
        documents=all_docs,
        embedding=OpenAIEmbeddings(),
        persist_directory="./vector_db"
    )