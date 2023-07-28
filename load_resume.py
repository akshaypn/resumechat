import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

current_directory = os.getcwd()

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def main():
    if not os.path.exists(os.path.join(current_directory, 'chroma_db')):
        directory = os.path.join(current_directory,'files')
    
        documents = load_docs(directory)

        docs = split_docs(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embeddings)
        persist_directory = "chroma_db"
        vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

        vectordb.persist()

        return

    else:
       return

