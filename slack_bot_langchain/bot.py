from typing import List, TypedDict
from slack_bolt import App
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.documents import Document
import streamlit as st
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Initialize
# app = App(
#     token=os.getenv("SLACK_BOT_TOKEN"),
#     signing_secret=os.getenv("SLACK_SIGNING_SECRET")
# )

template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.
Always say "thanks for asking!" at the end of the answer.

Context: {context}

Question: {question}

Helpful Answer:"""
retrieval_qa_chat_prompt = PromptTemplate.from_template(template)
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = retrieval_qa_chat_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
# Load knowledge base
vector_store = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

st.title("slack bot langchain")

input_text=st.text_input("ask a question")

if input_text:
    st.write(graph.invoke({"question": input_text, })["answer"])
# Handle messages
# @app.event("app_mention")
# def handle_mention(event, say):
# response = qa.run(event["text"])
# say(f"🤖: {response}")

# if __name__ == "__main__":
#     app.start(port=3000)
