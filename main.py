import streamlit as st
import vercel_ai
import json
import chroma_db
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'chroma_db'

client = vercel_ai.Client()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever()

query = st.text_area("Enter your question here", "Is Akshay a good machine learning engineer?")

matching_docs = vectordb.similarity_search(query)

prompt = f"""You are the helpful, polite and noble assistant of Akshay P Nambiar. Given the excerpt from the resume of Akshay P Nambiar as context, answer the following question. Remember the following things : 
1. Try to be precise and concise. Finish the answer in less than 200 words.
2. Answer in a well structured professional manner

### Question:
{query}

### Context :
{matching_docs[0]}
"""

answer = ""
if st.button('Answer'):
    try:
        for chunk in client.generate("openai:gpt-3.5-turbo", prompt): answer += chunk
        st.success("Answer Generated")
        st.write(answer)  
    except:
        st.exception("API timed out. Please try after a minute.")