import os
import re
import urllib.request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Konfiguratsiya
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ALICE_URL = "https://www.gutenberg.org/files/11/11-0.txt"
DATA_PATH = "/tmp/alice.txt"

# ---------------------------------------------------------------------------
# 1) Manba matnini yuklab olish (Alice in Wonderland)
# ---------------------------------------------------------------------------
def download_source_text():
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(ALICE_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # Gutenberg header/footer'ni olib tashlash
    start = text.find("CHAPTER I.")
    end = text.find("THE END")
    if start != -1 and end != -1:
        text = text[start:end]
    text = re.sub(r"\r\n", "\n", text)
    return text

# ---------------------------------------------------------------------------
# 2) Vector store yaratish (HuggingFace embeddings, Chroma in-memory)
# ---------------------------------------------------------------------------
def build_vectorstore():
    raw_text = download_source_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embeddings)
    return vectordb

print("Vector store tayyorlanmoqda...")
vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Vector store tayyor.")

# ---------------------------------------------------------------------------
# 3) LLM (DeepSeek, OpenAI-compatible API)
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.2,
)

# ---------------------------------------------------------------------------
# 4) FastAPI ilovasi
# ---------------------------------------------------------------------------
app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da aniq domenlar bilan cheklash tavsiya etiladi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Agentic RAG backend ishlamoqda"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question
    steps = []

    # --- Retrieve ---
    docs = retriever.invoke(question)
    steps.append("retrieve")

    # --- Grade documents (oddiy relevance tekshiruvi) ---
    context = "\n\n".join(d.page_content for d in docs)
    steps.append("grade_documents")

    # --- Generate ---
    system_prompt = (
        "Siz 'Alice in Wonderland' kitobi bo'yicha savollarga javob beruvchi yordamchisiz. "
        "Faqat berilgan kontekst asosida javob bering. Agar kontekstda javob bo'lmasa, "
        "buni halol ayting."
    )
    user_prompt = f"Kontekst:\n{context}\n\nSavol: {question}"

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    steps.append("generate")

    sources = [f"Chunk {i+1}" for i in range(len(docs))]

    return {
        "answer": response.content,
        "steps": steps,
        "sources": sources,
    }
