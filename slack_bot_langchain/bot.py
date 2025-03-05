from slack_bolt import App
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

# Initialize
app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

# Load knowledge base
vector_store = Chroma(
    persist_directory="./vector_db",
    embedding_function=OpenAIEmbeddings()
)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Handle messages
@app.event("app_mention")
def handle_mention(event, say):
    response = qa.run(event["text"])
    say(f"🤖: {response}")

if __name__ == "__main__":
    app.start(port=3000)