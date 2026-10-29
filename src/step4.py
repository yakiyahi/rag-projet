from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

ollama = ChatOllama(model="llama3.2:3b", temperature=0)