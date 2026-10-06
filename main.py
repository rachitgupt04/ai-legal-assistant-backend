
import os
import json
import numpy as np
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR, "data", "constitution_legal_chunks.json"
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR, "legal_embeddings.npy"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "light_embedding_model"
)

FUNCTIONS_FILE = os.path.join(
    BASE_DIR, "level1_functions_light.py"
)

# --------------------------------------------------
# Load legal chunks
# --------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    LEGAL_CHUNKS = json.load(f)

# --------------------------------------------------
# Load precomputed embeddings
# --------------------------------------------------

LEGAL_EMBEDDINGS = np.load(EMBEDDINGS_FILE)

# --------------------------------------------------
# Load lightweight embedding model
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = SentenceTransformer(
    MODEL_DIR,
    device=DEVICE
)

# --------------------------------------------------
# Load retrieval functions
# --------------------------------------------------

namespace = {}

with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
    exec(f.read(), namespace)

final_lightweight_retrieve = namespace["final_lightweight_retrieve"]
build_rag_context = namespace["build_rag_context"]
generate_legal_answer = namespace["generate_legal_answer"]

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="AI Legal Assistant API",
    description="RAG-based Indian Constitution Legal Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Request model
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(LEGAL_CHUNKS),
        "embedding_shape": list(LEGAL_EMBEDDINGS.shape),
        "device": DEVICE,
        "model": "gomyk/minilm-student-L3_uniform"
    }

# --------------------------------------------------
# Ask
# --------------------------------------------------

@app.post("/ask")
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:
        return {
            "error": "Question cannot be empty."
        }

    results = final_lightweight_retrieve(
        question,
        LEGAL_CHUNKS,
        LEGAL_EMBEDDINGS,
        MODEL,
        top_k=5
    )

    context = build_rag_context(results)

    response = generate_legal_answer(
        question,
        results
    )

    response["query"] = question
    response["retrieved_results"] = len(results)
    response["context_preview"] = context[:1500]

    return response
