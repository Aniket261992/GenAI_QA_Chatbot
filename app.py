import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if "store" not in st.session_state:
       st.session_state.store={}
       

def get_session_id(session_id:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]

def get_chat_response(chain,input):
       with_chat_history = RunnableWithMessageHistory(
                chain,
                get_session_history=get_session_id
            )
       response = with_chat_history.invoke(
                {"input":input},
                config=config
            )
            
       return response.content

    

config = {"configurable":{"session_id":"user1"}}

model_openai=["gpt-3.5-turbo","gpt-4o"]
model_groq=["gemma2-9b-it","llama3-70b-8192","llama3-8b-8192"]
all_models = model_openai+model_groq

prompt = ChatPromptTemplate(
    [
        ("system","You are a hepfult assistant, please answer the questions politely"),
        ("human","{input}")
    ]
)

st.title("Streamlit ChatBot App")

st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter the API key,use OPENAI API Key for OpenAI models and GROQ API Key for others",type="password")

llm_name = st.sidebar.selectbox("Select your model",all_models)

if api_key == GROQ_API_KEY:
            llm = ChatGroq(model=llm_name)
            chain = prompt|llm

elif api_key == OPENAI_API_KEY:
            llm = ChatOpenAI(model=llm_name)
            chain = prompt|llm
else:
     st.sidebar.write("Enter Valid API Key")

text_input = st.text_input("Enter your question:")

if text_input:
    output = get_chat_response(chain,text_input)
    st.write(output)

