from typing import List, TypedDict
from slack_bolt import App
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Initialize Slack app (commented out for now)
# app = App(
#     token=os.getenv("SLACK_BOT_TOKEN"),
#     signing_secret=os.getenv("SLACK_SIGNING_SECRET")
# )

# Define the prompt template
template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.
Always say "thanks for asking!" at the end of the answer.

Context: {context}

Question: {question}

Helpful Answer:"""
retrieval_qa_chat_prompt = PromptTemplate.from_template(template)

# Initialize the LLM and Embeddings
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Define state for the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    try:
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    except Exception as e:
        st.error(f"Error during document retrieval: {e}")
        return {"context": []}

def generate(state: State):
    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = retrieval_qa_chat_prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        st.error(f"Error during answer generation: {e}")
        return {"answer": "Sorry, I couldn't generate an answer."}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Load knowledge base
vector_store = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

# Streamlit frontend
st.title("Slack Bot using LangChain")

input_text = st.text_input("Ask a question")

if input_text:
    state = {"question": input_text}
    answer = graph.invoke(state)["answer"]
    st.write(answer)

# Handle Slack messages (commented out for now)
# @app.event("app_mention")
# def handle_mention(event, say):
#     response = graph.invoke({"question": event["text"]})["answer"]
#     say(f"🤖: {response}")

# if __name__ == "__main__":
#     app.start(port=3000)
