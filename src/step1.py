from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load = PyMuPDFLoader("../data/ia_liste_de_verification.pdf")


document = load.load()

for size in [200, 500, 2000]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
    chunks = splitter.split_documents(document)
    print(len(chunks))
    lengths = [len(c.page_content) for c in chunks]
    print(size, len(chunks), sum(lengths) / len(lengths))

  
