import shutil
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from step3 import load_doc, chunks_split, get_embeddings, build_vectorstore

ROOT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = ROOT_DIR / "chroma_db"

llm = ChatOllama(model="llama3.2:3b", temperature=0)

prompt = ChatPromptTemplate.from_template("""Tu es un assistant qui répond à des questions sur des documents.

Règles :
- Réponds UNIQUEMENT à partir du contexte ci-dessous.
- Si le contexte ne contient pas la réponse, réponds exactement : "Je ne trouve pas l'information dans les documents."
- N'utilise jamais tes connaissances générales.
- Réponds en français, en quelques phrases.

Contexte :
{context}

Question : {question}

Réponse :""")


def answer(question, vectorstore, k=3, debug=True):
    results = vectorstore.similarity_search_with_score(question, k=k)

    if debug:
        print("\n=== PASSAGES RÉCUPÉRÉS ===")
        for doc, score in results:
            print(f"[page {doc.metadata.get('page')}] distance={score:.3f}")
            print(doc.page_content[:300])
            print("---")

    context = "\n\n".join(doc.page_content for doc, _ in results)
    chain = prompt | llm
    reponse = chain.invoke({"context": context, "question": question})
    return reponse.content


if __name__ == "__main__":
    # Repartir d'une base vide : évite les doublons entre deux lancements
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    chunks = chunks_split(load_doc())
    print(f"{len(chunks)} chunks créés")
    vs = build_vectorstore(chunks, get_embeddings())
    #print("chunks créés :", len(chunks), "| dans Chroma :", vs._collection.count())
    question = "Que dit le document sur les bases légales ?"
    print("\nRÉPONSE :", answer(question, vs))