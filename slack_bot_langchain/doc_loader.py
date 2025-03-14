from langchain_community.document_loaders import GoogleDriveLoader
from langchain_community.document_loaders import ConfluenceLoader
from langchain_chroma import Chroma
# from langchain_google_community import GoogleDriveLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import os
from dotenv import load_dotenv
load_dotenv()

# def build_knowledge_base():
print("...............................................")
    # Load Google Drive
gdrive_loader = GoogleDriveLoader(
    folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
    service_account_key=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
    file_types=("document", "sheet", "pdf", "presentation"),
    recursive=True
)
print(os.getenv("GOOGLE_DRIVE_FOLDER_ID"))
drive_docs = gdrive_loader.load()
    
# Load Confluence
confluence_loader = ConfluenceLoader(
    url=os.getenv("CONFLUENCE_URL"),
    username=os.getenv("CONFLUENCE_USER"),
    api_key=os.getenv("CONFLUENCE_API_TOKEN"),
    space_key=os.getenv("CONFLUENCE_SPACE"),
    max_pages=2000
)

confluence_docs = confluence_loader.load()

all_docs = drive_docs + confluence_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(all_docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Combine and vectorize
# all_docs = confluence_docs
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./vector_db"
)

vector_store.add_documents(documents=all_splits)
