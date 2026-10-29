from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def load_doc():
    load = PyMuPDFLoader("../data/ia_liste_de_verification.pdf")
    document = load.load()
    return document

def chunks_split(doc,chuks_size=500):
    spliters = RecursiveCharacterTextSplitter(chunk_size=chuks_size,chunk_overlap=50)
    chunks = spliters.split_documents(doc)
    return chunks

#Embeding
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def build_vectorstore(chunks,embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="../chroma_db",
        )
    return vectorstore

def main():
    question = "Quelle est la recette de la tarte aux pommes ?"
    chunks =chunks_split(load_doc()) 
    vectorestore = build_vectorstore(chunks,get_embeddings())
    resultats = vectorestore.similarity_search(question, k=3)
    for result in resultats:
        print("Numero de page: ",result.metadata['page'])
        print(result.page_content[:300])

main()



