
import re
import numpy as np


def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def understand_query(query):
    q = normalize_text(query)

    hindi_chars = len(re.findall(r"[\u0900-\u097F]", query))

    hinglish_markers = [
        "kya", "kaise", "kyu", "kyon", "hai", "h",
        "batao", "btao", "matlab", "mera", "mere",
        "mujhe", "adhikar", "kanun", "kanoon"
    ]

    if hindi_chars > 0:
        language = "Hindi"
    elif any(x in q.split() for x in hinglish_markers):
        language = "Hinglish"
    else:
        language = "English"

    article = None

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

    if article is None:
        match = re.search(r"अनुच्छेद\s*(\d+[a-z]?)", query)
        if match:
            article = match.group(1)

    return {
        "query": query,
        "query_type": "legal_identifier" if article else "legal_concept",
        "article": article,
        "language": language
    }


def exact_article_search(query, legal_chunks):
    article = understand_query(query).get("article")

    if not article:
        return []

    article = str(article).upper()

    return [
        chunk
        for chunk in legal_chunks
        if str(chunk.get("article", "")).strip().upper() == article
    ]


def is_fundamental_rights_query(query):
    q = normalize_text(query)

    keywords = [
        "fundamental right",
        "fundamental rights",
        "mool adhikar",
        "मौलिक अधिकार",
        "basic rights",
        "constitutional rights",
        "right to equality",
        "right to life",
        "right to freedom",
        "right against exploitation"
    ]

    return any(k in q for k in keywords)


def is_freedom_of_speech_query(query):
    q = normalize_text(query)

    keywords = [
        "freedom of speech",
        "freedom of speech and expression",
        "right to freedom of speech",
        "speech and expression",
        "bolne ki azadi",
        "abhivyakti ki swatantrata",
        "अभिव्यक्ति की स्वतंत्रता",
        "वाक् स्वतंत्रता"
    ]

    return any(k in q for k in keywords)


def is_right_to_life_query(query):
    q = normalize_text(query)

    keywords = [
        "right to life",
        "right to life and personal liberty",
        "jeene ka adhikar",
        "जीवन का अधिकार",
        "personal liberty"
    ]

    return any(k in q for k in keywords)


def get_article_chunks(article, legal_chunks):
    article = str(article).upper()

    return [
        chunk
        for chunk in legal_chunks
        if str(chunk.get("article", "")).strip().upper() == article
    ]


def get_fundamental_rights_context(legal_chunks):
    results = []

    for chunk in legal_chunks:
        article = str(chunk.get("article", "")).strip()

        match = re.match(r"\d+", article)

        if not match:
            continue

        number = int(match.group())

        if 12 <= number <= 35:
            results.append(chunk)

    return results


def lightweight_semantic_search(
    query,
    legal_chunks,
    embeddings,
    encode_query,
    top_k=5
):
    query_embedding = encode_query(query)

    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "chunk": legal_chunks[int(idx)],
            "type": "semantic"
        })

    return results


def final_onnx_retrieve(
    query,
    legal_chunks,
    embeddings,
    encode_query,
    top_k=5
):
    info = understand_query(query)
    article = info.get("article")

    # --------------------------------
    # 1. Exact Article Routing
    # --------------------------------
    if article:
        exact = exact_article_search(
            query,
            legal_chunks
        )

        if exact:
            return [
                {
                    "score": 1.0,
                    "chunk": chunk,
                    "type": "exact"
                }
                for chunk in exact[:top_k]
            ]

    # --------------------------------
    # 2. Freedom of Speech → Article 19
    # --------------------------------
    if is_freedom_of_speech_query(query):
        chunks = get_article_chunks(
            "19",
            legal_chunks
        )

        if chunks:
            return [
                {
                    "score": 1.0,
                    "chunk": chunk,
                    "type": "concept_route"
                }
                for chunk in chunks[:top_k]
            ]

    # --------------------------------
    # 3. Right to Life → Article 21
    # --------------------------------
    if is_right_to_life_query(query):
        chunks = get_article_chunks(
            "21",
            legal_chunks
        )

        if chunks:
            return [
                {
                    "score": 1.0,
                    "chunk": chunk,
                    "type": "concept_route"
                }
                for chunk in chunks[:top_k]
            ]

    # --------------------------------
    # 4. Fundamental Rights
    # --------------------------------
    if is_fundamental_rights_query(query):
        rights = get_fundamental_rights_context(
            legal_chunks
        )

        if rights:
            # Keep structural routing deterministic.
            return [
                {
                    "score": 1.0,
                    "chunk": chunk,
                    "type": "fundamental_right"
                }
                for chunk in rights[:top_k]
            ]

    # --------------------------------
    # 5. Semantic fallback
    # --------------------------------
    return lightweight_semantic_search(
        query,
        legal_chunks,
        embeddings,
        encode_query,
        top_k
    )


def build_rag_context(results):
    if not results:
        return ""

    parts = []

    for i, result in enumerate(results, 1):
        chunk = result["chunk"]

        article = chunk.get("article", "N/A")
        title = chunk.get("title", "")
        page_start = chunk.get("page_start", "N/A")
        page_end = chunk.get("page_end", page_start)
        text = chunk.get("text", "")

        parts.append(
            f"[Source {i}]\n"
            f"Article: {article}\n"
            f"Title: {title}\n"
            f"Page: {page_start}-{page_end}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(parts)


def generate_legal_answer(query, results):

    disclaimer = (
        "This is general legal information, not legal advice. "
        "For a specific legal matter, consult a qualified legal professional."
    )

    if not results:
        return {
            "answer": (
                "I could not find sufficiently relevant "
                "constitutional information for this query."
            ),
            "sources": [],
            "next_steps": [],
            "disclaimer": disclaimer
        }

    best = results[0]["chunk"]

    article = best.get("article", "N/A")
    title = best.get("title", "")
    text = best.get("text", "").strip()

    page_start = best.get("page_start", "N/A")
    page_end = best.get("page_end", page_start)

    page = (
        page_start
        if page_start == page_end
        else f"{page_start}-{page_end}"
    )

    answer = (
        f"According to Article {article} of the Constitution of India"
    )

    if title:
        answer += f", {title}"

    answer += f"\n\n{text}"

    return {
        "answer": answer,
        "sources": [
            {
                "document": "Constitution of India",
                "article": str(article),
                "page": page
            }
        ],
        "next_steps": [
            f"You can read Article {article} "
            "for the exact constitutional provision."
        ],
        "disclaimer": disclaimer
    }
