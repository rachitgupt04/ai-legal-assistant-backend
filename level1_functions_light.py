
import re
import numpy as np


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

def understand_query(query):
    q = normalize_text(query)

    language = "English"

    hindi_chars = len(re.findall(r"[\u0900-\u097F]", query))

    hinglish_markers = [
        "kya", "kaise", "kyu", "kyon",
        "hai", "h", "batao", "btao",
        "matlab", "mera", "mere",
        "mujhe", "rights", "kanun",
        "kanoon", "adhikar"
    ]

    if hindi_chars > 0:
        language = "Hindi"
    elif any(word in q.split() for word in hinglish_markers):
        language = "Hinglish"

    article = None

    # Article 14
    patterns = [
        r"\barticle\s*[-:]?\s*(\d+[a-z]?)\b",
        r"\bart\.?\s*[-:]?\s*(\d+[a-z]?)\b",
        r"\banuchhed\s*[-:]?\s*(\d+[a-z]?)\b",
        r"\banuchhed\s*(\d+[a-z]?)\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)

        if match:
            article = match.group(1)
            break

    # Hindi numeral queries such as:
    # "अनुच्छेद 21 क्या है?"
    if article is None:
        match = re.search(
            r"अनुच्छेद\s*(\d+[a-z]?)",
            query,
            re.IGNORECASE
        )

        if match:
            article = match.group(1)

    if article:
        query_type = "legal_identifier"
    else:
        query_type = "legal_concept"

    return {
        "query": query,
        "query_type": query_type,
        "article": article,
        "language": language
    }


# ============================================================
# EXACT ARTICLE SEARCH
# ============================================================

def exact_article_search(query, legal_chunks):
    info = understand_query(query)

    article = info.get("article")

    if not article:
        return []

    article = str(article).upper()

    results = []

    for chunk in legal_chunks:
        chunk_article = str(
            chunk.get("article", "")
        ).strip().upper()

        if chunk_article == article:
            results.append(chunk)

    return results


# ============================================================
# FUNDAMENTAL RIGHTS ROUTING
# ============================================================

def is_fundamental_rights_query(query):
    q = normalize_text(query)

    keywords = [
        "fundamental right",
        "fundamental rights",
        "mool adhikar",
        "मौलिक अधिकार",
        "basic rights",
        "constitutional rights",
        "freedom of speech",
        "right to equality",
        "right to life",
        "right to freedom",
        "right against exploitation"
    ]

    return any(keyword in q for keyword in keywords)


def get_fundamental_rights_context(legal_chunks):
    results = []

    for chunk in legal_chunks:
        article = str(
            chunk.get("article", "")
        ).strip()

        try:
            number = int(
                re.match(r"\d+", article).group()
            )
        except Exception:
            continue

        # Fundamental Rights are primarily Articles 12-35
        if 12 <= number <= 35:
            results.append(chunk)

    return results


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(query, text):
    query_words = set(
        normalize_text(query).split()
    )

    text_words = set(
        normalize_text(text).split()
    )

    if not query_words:
        return 0.0

    overlap = len(
        query_words.intersection(text_words)
    )

    return overlap / len(query_words)


# ============================================================
# LEGAL RERANKING
# ============================================================

def rerank_results(
    query,
    candidates,
    semantic_scores=None
):
    if not candidates:
        return []

    if semantic_scores is None:
        semantic_scores = {}

    query_info = understand_query(query)
    article = query_info.get("article")

    ranked = []

    for index, chunk in enumerate(candidates):

        chunk_article = str(
            chunk.get("article", "")
        ).strip().upper()

        title = chunk.get("title", "")
        text = chunk.get("text", "")

        semantic_score = float(
            semantic_scores.get(index, 0.0)
        )

        keyword_score = keyword_overlap(
            query,
            f"{title} {text}"
        )

        exact_bonus = 0.0

        if article and chunk_article == str(article).upper():
            exact_bonus = 1.0

        score = (
            0.55 * semantic_score
            + 0.20 * keyword_score
            + 0.25 * exact_bonus
        )

        ranked.append({
            "score": score,
            "chunk": chunk
        })

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked


# ============================================================
# LIGHTWEIGHT SEMANTIC RETRIEVAL
# ============================================================

def lightweight_semantic_search(
    query,
    legal_chunks,
    embeddings,
    model,
    top_k=8
):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    candidates = []
    semantic_scores = {}

    for position, idx in enumerate(top_indices):

        idx = int(idx)

        candidates.append(
            legal_chunks[idx]
        )

        semantic_scores[position] = float(
            scores[idx]
        )

    ranked = rerank_results(
        query,
        candidates,
        semantic_scores
    )

    return ranked


# ============================================================
# FINAL LEGAL RETRIEVAL
# ============================================================

def final_lightweight_retrieve(
    query,
    legal_chunks,
    embeddings,
    model,
    top_k=5
):
    query_info = understand_query(query)

    # --------------------------------------------------------
    # 1. Exact article routing
    # --------------------------------------------------------

    if query_info.get("article"):

        exact_results = exact_article_search(
            query,
            legal_chunks
        )

        if exact_results:

            results = []

            for chunk in exact_results[:top_k]:
                results.append({
                    "score": 1.0,
                    "chunk": chunk,
                    "type": "exact"
                })

            return results

    # --------------------------------------------------------
    # 2. Fundamental Rights structural routing
    # --------------------------------------------------------

    if is_fundamental_rights_query(query):

        rights_chunks = get_fundamental_rights_context(
            legal_chunks
        )

        # Semantic ranking only inside Articles 12-35
        if rights_chunks:

            rights_indices = []

            for i, chunk in enumerate(legal_chunks):

                if chunk in rights_chunks:
                    rights_indices.append(i)

            query_embedding = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0]

            scores = np.dot(
                embeddings[rights_indices],
                query_embedding
            )

            order = np.argsort(
                scores
            )[::-1][:top_k]

            results = []

            for pos in order:

                original_idx = rights_indices[int(pos)]

                results.append({
                    "score": float(
                        scores[int(pos)]
                    ),
                    "chunk": legal_chunks[
                        original_idx
                    ],
                    "type": "fundamental_right"
                })

            return results

    # --------------------------------------------------------
    # 3. General lightweight semantic search
    # --------------------------------------------------------

    ranked = lightweight_semantic_search(
        query,
        legal_chunks,
        embeddings,
        model,
        top_k=top_k
    )

    results = []

    for item in ranked:

        results.append({
            "score": item["score"],
            "chunk": item["chunk"],
            "type": "semantic"
        })

    return results


# ============================================================
# RAG CONTEXT
# ============================================================

def build_rag_context(results):
    if not results:
        return ""

    context_parts = []

    for i, result in enumerate(results, 1):

        chunk = result["chunk"]

        article = chunk.get(
            "article",
            "N/A"
        )

        title = chunk.get(
            "title",
            ""
        )

        page_start = chunk.get(
            "page_start",
            "N/A"
        )

        page_end = chunk.get(
            "page_end",
            page_start
        )

        text = chunk.get(
            "text",
            ""
        )

        context_parts.append(
            f"[Source {i}]\n"
            f"Article: {article}\n"
            f"Title: {title}\n"
            f"Page: {page_start}-{page_end}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# SIMPLE LEGAL ANSWER
# ============================================================

def generate_legal_answer(
    query,
    results
):
    if not results:
        return {
            "answer": (
                "I could not find sufficiently relevant "
                "constitutional information for this query."
            ),
            "sources": [],
            "next_steps": [],
            "disclaimer": (
                "This is general legal information, "
                "not legal advice. For a specific legal "
                "matter, consult a qualified legal professional."
            )
        }

    best = results[0]["chunk"]

    article = best.get(
        "article",
        "N/A"
    )

    title = best.get(
        "title",
        ""
    )

    text = best.get(
        "text",
        ""
    ).strip()

    page_start = best.get(
        "page_start",
        "N/A"
    )

    page_end = best.get(
        "page_end",
        page_start
    )

    if page_start == page_end:
        page = page_start
    else:
        page = f"{page_start}-{page_end}"

    answer = (
        f"According to Article {article} "
        f"of the Constitution of India, "
        f"{title}\n\n"
        f"{text}"
    )

    sources = [
        {
            "document": "Constitution of India",
            "article": str(article),
            "page": page
        }
    ]

    return {
        "answer": answer,
        "sources": sources,
        "next_steps": [
            f"You can read Article {article} "
            "for the exact constitutional provision."
        ],
        "disclaimer": (
            "This is general legal information, "
            "not legal advice. For a specific legal "
            "matter, consult a qualified legal professional."
        )
    }
