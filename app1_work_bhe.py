from io import BytesIO
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain import LLMChain, PromptTemplate
from openai import OpenAI
from langchain.agents import Tool
from bhe1 import bhe_model  # Import the model instance

# To run this code-- use the command python -m streamlit run app1.py 

#chroma_client = chromadb.Client()
#collection = chroma_client.create_collection(name="my_collection")
# Initialize OpenAI API (replace with your API key)
#import open AI api key.
import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
import pickle
from langchain.agents import initialize_agent

system_message = "You are a helpful assistant."
k = 4

# Keys for session state and environment variables
conversation_key = "conversation"
human_message_key = "human"
os.environ['OPENAI_API_KEY']=""
answer=""
# Streamlit app layout
flag1=0
# Document upload
# Main function to setup Streamlit UI and run the conversation
st.set_page_config(page_title="Conversation Buffer Window Memory", page_icon=":robot:")
st.title("RAG Assistant")
st.markdown(f"System Message: {system_message}")
st.header(f"Buffer Window Memory k={k}")


pdf_r=[]
pdf = st.file_uploader("Upload your PDF", type='pdf',accept_multiple_files=True)
for uploaded_file in pdf:
    pdf_r.append(PdfReader(uploaded_file))
if len(pdf_r)>0:
    text = ""
    for i in pdf_r:
     for page in i.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
    chunks = text_splitter.split_text(text=text)
    embeddings = bhe_model.model_norm

    persist_directory = "chroma_db_bhe1"
    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()

    #from langchain.agents.react.agent import create_react_agent
    # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=4,
        return_messages=True
    )
    # passing the topic into gpt.
    #j_chain1 = ChatPromptTemplate.from_template("gather more information and present examples on the {query}, if there is no information available reply saying no information is available") | llm
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        #retriever=faiss_index_Dental
        retriever=vectordb.as_retriever()
    )


    tools_db = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering any questions or queries to get more information about the query, if you do not find any relatable information, reply saying no inofrmation is available'
            )
        )
    ]


    agent_dental = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools_db,
        llm=llm,
        verbose=False,
        max_iterations=3,
        early_stopping_method='generate',
        handle_parsing_errors=True,
        memory=conversational_memory
    )
    
    # Input field for user message
    user_input = st.text_input(label="Enter your message", placeholder="Send a message")

    # Submit user input to the conversation chain
    if len(user_input) > 1:
        agent_dental(user_input)
        #st.write(agent_dental.memory.chat_memory.messages[-1].content)
    placeholder = st.empty()
    # Display conversation messages
    with placeholder.container():
        for index, msg in enumerate(agent_dental.memory.chat_memory.messages):
            if msg.type == human_message_key:
                st.write(f"User: {msg.content}")
            else:
                st.write(f"AI: {msg.content}")




#DAI: Some different types of composite materials include fiberglass,
#tooth enamel and dentin, hybrid, micro, or nanofiller reinforced resins, prosthodontic resins, restorative composites, and core build-up composites.associated reporting to support the University’s strategic plan, and providing advice on institutional performance to the Council and senior executive. 