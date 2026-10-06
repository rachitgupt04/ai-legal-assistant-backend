
import os
import json
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "constitution_legal_chunks.json"
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "legal_embeddings.npy"
)

TOKENIZER_FILE = os.path.join(
    BASE_DIR,
    "light_embedding_model",
    "tokenizer.json"
)

ONNX_MODEL_FILE = os.path.join(
    BASE_DIR,
    "onnx_model",
    "model.onnx"
)

FUNCTIONS_FILE = os.path.join(
    BASE_DIR,
    "level1_functions_onnx.py"
)


# ============================================================
# LOAD LEGAL DATA
# ============================================================

with open(DATA_FILE, "r", encoding="utf-8") as f:
    LEGAL_CHUNKS = json.load(f)

LEGAL_EMBEDDINGS = np.load(EMBEDDINGS_FILE)


# ============================================================
# LOAD TOKENIZER
# ============================================================

TOKENIZER = Tokenizer.from_file(TOKENIZER_FILE)

# Prevent unexpectedly long requests
TOKENIZER.enable_truncation(max_length=256)
TOKENIZER.enable_padding()


# ============================================================
# LOAD ONNX MODEL
# ============================================================

ONNX_SESSION = ort.InferenceSession(
    ONNX_MODEL_FILE,
    providers=["CPUExecutionProvider"]
)


# ============================================================
# LOAD RETRIEVAL FUNCTIONS
# ============================================================

namespace = {}

with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
    exec(f.read(), namespace)

final_onnx_retrieve = namespace["final_onnx_retrieve"]
build_rag_context = namespace["build_rag_context"]
generate_legal_answer = namespace["generate_legal_answer"]


# ============================================================
# QUERY EMBEDDING
# ============================================================

def encode_query(text):

    encoded = TOKENIZER.encode(text)

    input_ids = np.array(
        [encoded.ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    outputs = ONNX_SESSION.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )

    last_hidden_state = outputs[0]

    # --------------------------------------------------------
    # Mean pooling
    # Matches SentenceTransformer Pooling module
    # --------------------------------------------------------

    mask = attention_mask[:, :, None].astype(np.float32)

    sum_embeddings = np.sum(
        last_hidden_state * mask,
        axis=1
    )

    sum_mask = np.clip(
        mask.sum(axis=1),
        a_min=1e-9,
        a_max=None
    )

    embedding = sum_embeddings / sum_mask

    embedding = embedding[0]

    # L2 normalization
    embedding = embedding / (
        np.linalg.norm(embedding) + 1e-12
    )

    return embedding.astype(np.float32)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Legal Assistant API",
    description="RAG-based Indian Constitution Legal Assistant",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "chunks": len(LEGAL_CHUNKS),
        "embedding_shape": list(LEGAL_EMBEDDINGS.shape),
        "embedding_model": "gomyk/minilm-student-L3_uniform",
        "runtime": "ONNX Runtime CPU",
        "version": "2.0.0"
    }


# ============================================================
# ASK
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:
        return {
            "error": "Question cannot be empty."
        }

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    results = final_onnx_retrieve(
        question,
        LEGAL_CHUNKS,
        LEGAL_EMBEDDINGS,
        encode_query,
        top_k=5
    )

    # --------------------------------------------------------
    # Build RAG context
    # --------------------------------------------------------

    context = build_rag_context(results)

    # --------------------------------------------------------
    # Generate structured response
    # --------------------------------------------------------

    response = generate_legal_answer(
        question,
        results
    )

    response["query"] = question
    response["retrieved_results"] = len(results)
    response["retrieval_types"] = [
        result.get("type", "unknown")
        for result in results
    ]

    # Keep response lightweight
    response["context_preview"] = context[:1500]

    return response
