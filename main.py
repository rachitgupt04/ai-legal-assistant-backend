
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

import os
import json
import re

import chromadb
import torch
from sentence_transformers import SentenceTransformer


# =========================================================
# PORTABLE PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "constitution_legal_chunks.json"
)

FUNCTIONS_FILE = os.path.join(
    BASE_DIR,
    "level1_functions.py"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embedding_model"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# =========================================================
# GLOBAL RAG STATE
# =========================================================

rag = {}


# =========================================================
# LOAD RAG SYSTEM
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag

    print("Loading legal knowledge base...")

    # -----------------------------------------
    # Load legal chunks
    # -----------------------------------------

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        articles = json.load(f)

    print(
        f"Loaded {len(articles)} legal chunks."
    )

    # -----------------------------------------
    # Select device
    # -----------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------
    # Load embedding model
    # -----------------------------------------

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_DIR,
        device=device
    )

    print("Embedding model loaded.")

    # -----------------------------------------
    # Load ChromaDB
    # -----------------------------------------

    print("Loading ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection_legal = client.get_collection(
        name="constitution_legal"
    )

    print(
        "ChromaDB records:",
        collection_legal.count()
    )

    # -----------------------------------------
    # Prepare namespace for Level 1 functions
    # -----------------------------------------

    namespace = {
        "__builtins__": __builtins__,
        "re": re,
        "json": json,
        "articles": articles,
        "model": model,
        "collection_legal": collection_legal
    }

    # -----------------------------------------
    # Load saved Level 1 functions
    # -----------------------------------------

    with open(
        FUNCTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        functions_code = f.read()

    exec(
        compile(
            functions_code,
            "level1_functions.py",
            "exec"
        ),
        namespace
    )

    rag = namespace

    print(
        "========================================"
    )

    print(
        "LEGAL ASSISTANT BACKEND READY"
    )

    print(
        "========================================"
    )

    yield

    rag.clear()


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Legal Assistant",
    description="RAG-based Indian Legal Information Assistant",
    version="1.0.0",
    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# =========================================================
# REQUEST MODEL
# =========================================================

class Question(BaseModel):

    query: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    if not rag:
        return {
            "status": "starting"
        }

    return {
        "status": "ok",
        "chunks": len(
            rag["articles"]
        ),
        "database_records": rag[
            "collection_legal"
        ].count()
    }


# =========================================================
# ASK LEGAL QUESTION
# =========================================================

@app.post("/ask")
def ask(question: Question):

    query = question.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # -----------------------------------------
    # RAG retrieval
    # -----------------------------------------

    result = rag[
        "final_legal_retrieve_v2"
    ](query)

    # -----------------------------------------
    # Fundamental Rights query
    # -----------------------------------------

    if rag[
        "is_fundamental_rights_query"
    ](query):

        answer = rag[
            "generate_fundamental_rights_answer"
        ](
            query,
            result["selected_context"]
        )

        sources = [

            {
                "document": "Constitution of India",
                "article": r["article"],
                "page": r["page_start"],
                "title": r["title"]
            }

            for r in result[
                "selected_context"
            ]
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    # -----------------------------------------
    # Normal legal query
    # -----------------------------------------

    response = rag[
        "generate_legal_answer"
    ](
        query,
        result["selected_context"]
    )

    if isinstance(
        response,
        dict
    ):

        return response

    return {
        "answer": str(response),
        "sources": []
    }
