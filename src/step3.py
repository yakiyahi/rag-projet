from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"


def load_doc():
    loader = PyMuPDFLoader(str(DATA_DIR / "ia_liste_de_verification.pdf"))
    return loader.load()


def chunks_split(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def build_vectorstore(chunks, embeddings):
    # Pas de persist_directory : base en mémoire, recréée à chaque lancement
    return Chroma.from_documents(documents=chunks, embedding=embeddings)