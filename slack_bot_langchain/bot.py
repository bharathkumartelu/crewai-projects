from slack_bolt import App
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Initialize
# app = App(
#     token=os.getenv("SLACK_BOT_TOKEN"),
#     signing_secret=os.getenv("SLACK_SIGNING_SECRET")
# )

# Load knowledge base
vector_store = Chroma(
    persist_directory="./vector_db",
    embedding_function=OpenAIEmbeddings()
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm=ChatOpenAI()
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retriever=vector_store.as_retriever()

print(retriever)

qa = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

print(qa.invoke({"input": "summary of software development space", }))
# Handle messages
# @app.event("app_mention")
# def handle_mention(event, say):
# response = qa.run(event["text"])
# say(f"🤖: {response}")

# if __name__ == "__main__":
#     app.start(port=3000)