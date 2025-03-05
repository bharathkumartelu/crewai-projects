# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

prompt=ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is Carl."),
    ("user", "Question:{question}")
])


llm = init_chat_model("llama3-8b-8192", model_provider="groq")

st.title("slack bot langchain")

input_text=st.text_input("ask a question")

output_parser =  StrOutputParser()

chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))



