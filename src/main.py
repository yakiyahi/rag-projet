document = """
Le Data Engineering consiste à collecter,
transformer et mettre à disposition les données.
Apache Spark traite des données distribuées.
Delta Lake ajoute des transactions ACID.
"""

def chunks_text(text,chunks_size=100):
    chunks = []
    for start in range(0,len(text),chunks_size):
        chunk = text[start:start + chunks_size]
        chunks.append(chunk)
    print(chunks)        

chunks_text(document)
