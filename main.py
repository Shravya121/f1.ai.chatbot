import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

CHROMA_DIR = "chroma_db"

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading ChromaDB...")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

print("Loading Groq LLM...")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

prompt_template = """You are an expert F1 historian. Use the context below to help answer the question, but also use your own F1 knowledge to give a complete, interesting answer.

Context from F1 database:
{context}

Question: {question}

Give a detailed, enthusiastic answer like a true F1 fan:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 8}),
    chain_type_kwargs={"prompt": PROMPT}
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=open("index.html", encoding="utf-8").read())
@app.post("/ask")
def ask(body: Question):
    if not body.question.strip():
        return JSONResponse({"answer": "Please ask an F1 question!"})
    result = qa.invoke({"query": body.question})
    return JSONResponse({"answer": result["result"]})

@app.get("/standings")
def get_standings():
    try:
        import requests
        res = requests.get("https://api.openf1.org/v1/drivers?session_key=latest")
        drivers = res.json()
        return JSONResponse({"drivers": drivers})
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    print("Starting F1 Chatbot at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
