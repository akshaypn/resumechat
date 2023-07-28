import streamlit as st
import vercel_ai
import json
import chromadb
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from load_resume import main
from config import models


main()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'chroma_db'

client = vercel_ai.Client()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever()

with st.sidebar:
    st.title('🦜️🔗RESUME-CHAT \n PDF Q/A Retrieval LLM-LANGCHAIN AGENT🤗')
    st.markdown('''
    ## About APP:
    The application can be used to chat with my resume and recieve answers to your queries.\n
    Chromadb is used for storing the vector embeddings which were created using sentence transformer's all-MiniLM-L6-v2.\n
    We use sentence similarity to return the most matching document from the corpus and provide it as context to the LLM.\n
                         
    The app's primary resource is utilised to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [Vercel Playground](https://sdk.vercel.ai/)
    - [Chroma DB](https://www.trychroma.com/)
    - [Sentence Transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

    A big thanks to [ADING](https://github.com/ading2210) whose module [vercel-llm-api](https://pypi.org/project/vercel-llm-api/) makes all of this possible.

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/akshaynambiar7/)
    - [Website](https://akshaypn.github.io/)
    
    ''')
    st.write('💡All about resume-chat, created by [Akshay](https://www.linkedin.com/in/akshaynambiar7/)🤗')
    st.info("The agent may output incorrect, ambiguous and disrespectful answers. The Author does not endorse the facts/statements generated by the models.")



st.title("RESUME CHAT (Q/A)")
st.header("Ask any question !!")
st.info("Write your question, choose the model (gpt-3.5 works best) and hit Answer. The results will be based on my resume")
query = st.text_area("Enter your question here", "Why is Akshay the ideal candidate for the Senior Machine Learning Engineer(NLP) role?\n\n\
Write your answer with 4-5 short and sharp bullet points")

curr_model = st.selectbox("Select the Language Model", models)

model_params = client.model_defaults[curr_model]
model_params['maximumLength'] = 2000


try:
    matching_docs = vectordb.similarity_search(query)

    prompt = f"""You are the helpful, polite and noble assistant of Akshay P Nambiar. Always remember the following things while answering: 
    1. Be precise, concise and helpful.
    2. Answer in a well structured professional manner
    3. Provide well formatted and structured replies.
    4. If the question is derogatory, abusive or contains profanity, refuse to answer.

    Given the excerpt from the resume of Akshay P Nambiar as context, answer the following question. 
    ### Question:
    {query}

    ### Context :
    {matching_docs[0]}
    """
except:
    prompt = f"""You are the helpful, polite and noble assistant of Akshay P Nambiar. Always remember the following things: 
    1. Try to be precise and concise. Finish the answer in less than 200 words.
    2. Answer in a well structured professional manner.
    3. If the input is derogatory, abusive or contains profanity, refuse to answer.

    

    ### Context :
    The question is beyond the scope of this resume. Reply that you need more information. 
    """

answer = ""
if st.button('Answer'):
    try:
        with st.spinner("Shifting Rocks.. Checking nooks and corners... Thinking hard... 🤔"):
            for chunk in client.generate(curr_model, prompt): answer += chunk
            
            st.success("Answer Generated ✅")
            if len(answer) > 1:
                st.write(answer)  
            else:
                st.exception("API timed out. Please try after a minute or choose another model.")
    except:
        st.exception("API timed out. Please try after a minute or choose another model.")