from langchain_community.document_loaders import GoogleDriveLoader
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import os
from dotenv import load_dotenv
load_dotenv()

# def build_knowledge_base():
print("...............................................")
    # Load Google Drive
    # gdrive_loader = GoogleDriveLoader(
    #     folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
    #     service_account_key=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    # )
    # drive_docs = gdrive_loader.load()
    
    # Load Confluence
confluence_loader = ConfluenceLoader(
    url=os.getenv("CONFLUENCE_URL"),
    username=os.getenv("CONFLUENCE_USER"),
    api_key=os.getenv("CONFLUENCE_API_TOKEN"),
    space_key=os.getenv("CONFLUENCE_SPACE")
)
print(os.getenv("CONFLUENCE_URL"))

confluence_docs = confluence_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(confluence_docs)

    
    # Combine and vectorize
# all_docs = confluence_docs
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./vector_db"
)

uuids = [str(uuid4()) for _ in range(len(all_splits))]

vector_store.add_documents(documents=all_splits, ids=uuids)