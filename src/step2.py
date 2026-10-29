from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embedings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

phrases = [
    "Le chat dort sur le canapé.",
    "Un félin se repose sur le sofa.",
    "Le cours de la bourse a chuté hier.",
]
# TODO 1 : vecteur de la phrase 0 (méthode embed_query).
#          Affichez sa longueur (len) et ses 5 premiers nombres.
#vecteur = embedings.embed_query(phrases[0])
#print(len(vecteur))
#print(vecteur[:5])

# TODO 2 : calculez les 3 vecteurs
vecteurs = embedings.embed_documents(phrases)
V1 = vecteurs[0]
V2 = vecteurs[1]
V3 = vecteurs[2]
# TODO 3 : calculez la similarité cosinus entre 0 et 1, puis entre 0 et 2
#          formule : np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
sim_cos_V1_V2 = np.dot(V1,V2)/ (np.linalg.norm(V1) * np.linalg.norm(V2))
sim_cos_V1_V3 = np.dot(V1,V3)/ (np.linalg.norm(V1) * np.linalg.norm(V3))

print("v1_v2",sim_cos_V1_V2)
print("v1_v3",sim_cos_V1_V3)

    
