from langchain.embeddings import HuggingFaceBgeEmbeddings

import os
import pickle
import streamlit as st
from io import BytesIO
from langchain import LLMChain, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI
from bhe1 import bhe_model  # Import the model instance
from dotenv import load_dotenv
import os
import getpass
# Set API key as environment variable
os.environ['OPENAI_API_KEY'] = ""
#hugging face BGE embedding
# Get the OpenAI API key from the environment variable or user input


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_bytes):
    pdf_reader = PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
# Function to create a vector store from text chunks
def mask_presido(pdf_file):
    pass
def csv_reader (csv_file) :
    pass
# Function to create a vector store from text chunks
def create_vector_store(text_chunks, persist_directory="chroma_dbB"):

    embeddings = bhe_model.model_norm
    vectordb1 = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb1.persist()
    return vectordb1

# Function to answer a prompt using a QA chain and a vector store
def answer_from_qa_chain(vectordb, prompt):
    model = ChatOpenAI(temperature=0)
    temp="You are a helpful assistant and only  say what you know. if you did not find a answer reply saying, i cannot find the information for this prompt based on the given context"
    chain = load_qa_chain(model, chain_type="stuff", verbose=True)
    matching_docs = vectordb.similarity_search(prompt)
    return chain.run(input_documents=matching_docs, question=temp+" "+prompt)

# Function to answer a prompt using OpenAI completion
def answer_from_openai(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. answer this " + prompt}
        ]
    )
    return completion.choices[0].message.content

# Main application logic
st.title("AI RAG Assistant")

# File upload
pdf = st.file_uploader("Upload your PDF", type='pdf')

# Prompt input
prompt = st.text_input("Enter your prompt:")

if prompt:
    if pdf:
        text_chunks = RecursiveCharacterTextSplitter().split_text(extract_text_from_pdf(pdf.read()))
        vectordb1 = create_vector_store(text_chunks)
        #rerank_results = co.rerank(query=query, documents=documents, top_n=3, model="rerank-multilingual-v2.0")
        answer = answer_from_qa_chain(vectordb1, prompt)
    else:
        answer = answer_from_openai(prompt)

    st.write("Response:", answer)
