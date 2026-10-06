# AI Legal Assistant - Level 1 RAG Functions
# Saved from Google Colab checkpoint

def understand_query(query):

    result = {
        "query": query,
        "query_type": "general",
        "article": None,
        "language": "English"
    }

    query_lower = query.lower()

    # --------------------------------------------------
    # 1. Language Detection
    # --------------------------------------------------

    hindi_chars = re.findall(r'[\u0900-\u097F]', query)

    hinglish_words = [
        "mere", "meri", "mujhe", "mera", "humara",
        "kya", "kaise", "kyu", "kyun", "karu",
        "hai", "hain", "nahi", "nahin",
        "se", "ko", "ka", "ki", "ke",
        "par", "pe", "ghar", "saman",
        "wala", "wali", "liye",
        "police", "landlord", "salary"
    ]

    hinglish_count = sum(
        1 for word in hinglish_words
        if re.search(r'\b' + re.escape(word) + r'\b', query_lower)
    )

    if hindi_chars:
        result["language"] = "Hindi"
    elif hinglish_count >= 2:
        result["language"] = "Hinglish"
    else:
        result["language"] = "English"

    # --------------------------------------------------
    # 2. Article Detection
    # --------------------------------------------------

    # English:
    # Article 21
    english_article = re.search(
        r'\barticle\s+(\d{1,3}[A-Z]{0,2})\b',
        query,
        re.IGNORECASE
    )

    # Hindi:
    # अनुच्छेद 21
    hindi_article = re.search(
        r'अनुच्छेद\s*(\d{1,3}[A-Z]{0,2})',
        query,
        re.IGNORECASE
    )

    if english_article:
        result["query_type"] = "legal_identifier"
        result["article"] = english_article.group(1).upper()

    elif hindi_article:
        result["query_type"] = "legal_identifier"
        result["article"] = hindi_article.group(1).upper()

    # --------------------------------------------------
    # 3. Legal Problem Detection
    # --------------------------------------------------

    problem_keywords = [
        "police",
        "landlord",
        "tenant",
        "salary",
        "job",
        "fired",
        "property",
        "fraud",
        "harassment",
        "divorce",
        "marriage",
        "arrest",
        "threat",
        "cheating",
        "money",
        "notice",
        "crime",
        "complaint",
        "saman",
        "ghar se",
        "nikal",
        "paise",
        "nahi di"
    ]

    if any(keyword in query_lower for keyword in problem_keywords):
        result["query_type"] = "legal_problem"

    # --------------------------------------------------
    # 4. Legal Concept Detection
    # --------------------------------------------------

    legal_concept_keywords = [
        "constitution",
        "freedom",
        "speech",
        "equality",
        "life",
        "liberty",
        "fundamental right",
        "fundamental rights",
        "rights",
        "law",
        "legal",
        "protection",
        "justice",
        "अनुच्छेद",
        "अधिकार",
        "मौलिक अधिकार"
    ]

    if any(keyword in query_lower for keyword in legal_concept_keywords):

        if result["query_type"] == "general":
            result["query_type"] = "legal_concept"

    return result


def expand_legal_query(query):
    query_lower = query.lower()

    expanded_query = query

    # Fundamental Rights
    if (
        "fundamental right" in query_lower
        or "fundamental rights" in query_lower
        or "मौलिक अधिकार" in query
    ):
        expanded_query += (
            " Fundamental Rights Part III Articles 12 13 14 15 "
            "16 17 18 19 20 21 21A 22 23 24 25 26 27 28 "
            "29 30 32 constitutional rights"
        )

    # Freedom of speech
    if (
        "freedom of speech" in query_lower
        or "freedom of expression" in query_lower
    ):
        expanded_query += (
            " Article 19 freedom of speech and expression"
        )

    # Life and liberty
    if (
        "life and personal liberty" in query_lower
        or "protection of life" in query_lower
    ):
        expanded_query += (
            " Article 21 protection of life personal liberty"
        )

    # Equality
    if (
        "equality before law" in query_lower
        or "equality" in query_lower
    ):
        expanded_query += (
            " Article 14 equality before law equal protection"
        )

    return expanded_query


def exact_article_search(query, articles):
    article_number = None

    # English: Article 21
    match = re.search(
        r'\barticle\s+(\d{1,3}[A-Z]{0,2})\b',
        query,
        re.IGNORECASE
    )

    if match:
        article_number = match.group(1).upper()

    # Hindi: अनुच्छेद 21
    if article_number is None:
        match = re.search(
            r'अनुच्छेद\s*(\d{1,3}[A-Z]{0,2})',
            query
        )

        if match:
            article_number = match.group(1).upper()

    if article_number is None:
        return []

    return [
        article
        for article in articles
        if str(article["article"]).upper() == article_number
    ]


def is_fundamental_rights_query(query):
    query_lower = query.lower()

    keywords = [
        "fundamental right",
        "fundamental rights",
        "मौलिक अधिकार"
    ]

    return any(keyword in query_lower for keyword in keywords)


def get_fundamental_rights_context(articles):
    """
    Retrieve Constitution Part III / Fundamental Rights provisions.
    """

    target_articles = [
        "12", "13",
        "14", "15", "16", "17", "18",
        "19", "20", "21", "21A", "22",
        "23", "24",
        "25", "26", "27", "28",
        "29", "30",
        "32"
    ]

    results = []

    for article in articles:

        if str(article["article"]).upper() in [
            x.upper() for x in target_articles
        ]:
            result = article.copy()

            result["distance"] = 0.0
            result["legal_score"] = 1.0
            result["type"] = "legal_structure"

            results.append(result)

    return results


def normalize_text(text):
    """Normalize text for keyword comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def keyword_overlap(query, text):
    """Calculate keyword overlap between query and document text."""
    
    query_words = set(normalize_text(query).split())
    text_words = set(normalize_text(text).split())
    
    # Remove common stop words
    stop_words = {
        "the", "is", "a", "an", "of", "to", "and",
        "what", "does", "do", "about", "in", "on",
        "for", "this", "that", "constitution"
    }
    
    query_words = query_words - stop_words
    
    if not query_words:
        return 0.0
    
    overlap = query_words.intersection(text_words)
    
    return len(overlap) / len(query_words)


def calculate_legal_score(query, result):
    """Calculate final legal relevance score."""
    
    # Convert distance to similarity-like score
    semantic_score = max(0.0, 1.0 - result["distance"])
    
    # Keyword match in title
    title_score = keyword_overlap(
        query,
        result["title"]
    )
    
    # Keyword match in legal text
    text_score = keyword_overlap(
        query,
        result["text"]
    )
    
    # Weighted final score
    final_score = (
        0.60 * semantic_score +
        0.20 * title_score +
        0.20 * text_score
    )
    
    return final_score


def rerank_legal_results(query, results):
    
    scored_results = []
    
    for result in results:
        
        legal_score = calculate_legal_score(
            query,
            result
        )
        
        result_copy = result.copy()
        result_copy["legal_score"] = legal_score
        
        scored_results.append(result_copy)
    
    
    # Highest legal score first
    scored_results.sort(
        key=lambda x: x["legal_score"],
        reverse=True
    )
    
    return scored_results


def select_best_legal_context(
    query,
    results,
    max_contexts=3,
    min_legal_score=0.30
):
    """
    Select only sufficiently relevant legal results.
    """

    # Results already sorted by legal score
    selected = []

    for result in results:

        # Keep only sufficiently relevant results
        if result["legal_score"] >= min_legal_score:
            selected.append(result)

        # Maximum context limit
        if len(selected) >= max_contexts:
            break

    return selected


def final_legal_retrieve_v2(
    query,
    top_k=5,
    max_contexts=10,
    min_legal_score=0.30
):

    # --------------------------------------------------
    # 1. Understand query
    # --------------------------------------------------

    query_info = understand_query(query)

    # --------------------------------------------------
    # 2. Expand query
    # --------------------------------------------------

    expanded_query = expand_legal_query(query)

    # --------------------------------------------------
    # 3. Exact Article Search
    # --------------------------------------------------

    exact_results = exact_article_search(query, articles)

    # --------------------------------------------------
    # 4. Legal Structure Routing
    # --------------------------------------------------

    structured_results = []

    if is_fundamental_rights_query(query):

        structured_results = get_fundamental_rights_context(
            articles
        )

    # --------------------------------------------------
    # 5. Semantic Search
    # --------------------------------------------------

    semantic_results = []

    semantic_output = collection_legal.query(
        query_embeddings=model.encode(
            [expanded_query],
            normalize_embeddings=True
        ).tolist(),
        n_results=top_k
    )

    ids = semantic_output["ids"][0]
    distances = semantic_output["distances"][0]
    metadatas = semantic_output["metadatas"][0]
    documents = semantic_output["documents"][0]

    for i in range(len(ids)):

        if distances[i] <= 1.00:

            semantic_results.append({
                "article": metadatas[i]["article"],
                "page_start": metadatas[i]["page_start"],
                "page_end": metadatas[i]["page_end"],
                "title": metadatas[i]["title"],
                "text": documents[i],
                "distance": distances[i],
                "type": "semantic"
            })

    # --------------------------------------------------
    # 6. Combine Results
    # --------------------------------------------------

    combined_results = []

    # Structured results first
    for result in structured_results:

        result_copy = result.copy()

        result_copy["distance"] = 0.0
        result_copy["legal_score"] = 1.0
        result_copy["type"] = "legal_structure"

        combined_results.append(result_copy)

    # Exact results
    for result in exact_results:

        result_copy = result.copy()

        result_copy["distance"] = 0.0
        result_copy["type"] = "exact"

        combined_results.append(result_copy)

    # --------------------------------------------------
    # 7. Remove duplicates
    # --------------------------------------------------

    unique_results = {}

    for result in combined_results:

        key = (
            str(result["article"]).upper(),
            result["page_start"]
        )

        unique_results[key] = result

    # --------------------------------------------------
    # 8. Add semantic results
    # --------------------------------------------------

    for result in semantic_results:

        key = (
            str(result["article"]).upper(),
            result["page_start"]
        )

        if key not in unique_results:

            unique_results[key] = result

    combined_results = list(unique_results.values())

    # --------------------------------------------------
    # 9. Reranking
    # --------------------------------------------------

    reranked_results = rerank_legal_results(
        query,
        combined_results
    )

    # --------------------------------------------------
    # 10. Structured queries
    # --------------------------------------------------

    if is_fundamental_rights_query(query):

        # Keep all structurally retrieved Fundamental Rights
        relevant_results = [
            r for r in reranked_results
            if r["type"] == "legal_structure"
        ]

        # Add any strong semantic/exact results
        for r in reranked_results:

            if r["type"] != "legal_structure":
                if r["legal_score"] >= min_legal_score:
                    relevant_results.append(r)

    else:

        relevant_results = [
            r for r in reranked_results
            if r["legal_score"] >= min_legal_score
        ]

    # --------------------------------------------------
    # 11. Limit context
    # --------------------------------------------------

    if is_fundamental_rights_query(query):
        selected_results = relevant_results
    else:
        selected_results = relevant_results[:max_contexts]

    has_relevant_context = len(selected_results) > 0

    # --------------------------------------------------
    # 12. Final Output
    # --------------------------------------------------

    return {
        "query": query,
        "expanded_query": expanded_query,
        "query_info": query_info,
        "has_relevant_context": has_relevant_context,
        "results": reranked_results,
        "relevant_results": relevant_results,
        "selected_context": selected_results
    }


def build_rag_context(context_results):
    if not context_results:
        return ""

    context_parts = []

    for i, result in enumerate(context_results, 1):
        source_block = f"""
SOURCE {i}
Document: Constitution of India
Article: {result['article']}
Page: {result['page_start']}
Title: {result['title']}

Text:
{result['text']}
""".strip()

        context_parts.append(source_block)

    return "\n\n".join(context_parts)


def generate_legal_answer(query, context_results):

    if not context_results:
        return {
            "answer": (
                "I could not find sufficiently relevant information "
                "in the current legal knowledge base."
            ),
            "sources": [],
            "next_steps": [],
            "disclaimer": (
                "This is general legal information, not legal advice. "
                "For a specific legal matter, consult a qualified legal professional."
            )
        }

    primary = context_results[0]

    article = primary["article"]
    page = primary["page_start"]
    text = primary["text"]

    # Detect language
    query_info = understand_query(query)
    language = query_info["language"]

    # --------------------------------------------------
    # English
    # --------------------------------------------------

    if language == "English":

        answer = (
            f"According to Article {article} of the Constitution of India, "
            f"{text.replace(f'{article}. ', '', 1).strip()}"
        )

        next_steps = [
            f"You can read Article {article} for the exact constitutional provision."
        ]

        disclaimer = (
            "This is general legal information, not legal advice. "
            "For a specific legal matter, consult a qualified legal professional."
        )

    # --------------------------------------------------
    # Hindi
    # --------------------------------------------------

    elif language == "Hindi":

        if article == "21":
            answer = (
                "भारतीय संविधान का अनुच्छेद 21 जीवन और व्यक्तिगत "
                "स्वतंत्रता की सुरक्षा करता है। किसी भी व्यक्ति को "
                "कानून द्वारा स्थापित प्रक्रिया के अनुसार ही उसके "
                "जीवन या व्यक्तिगत स्वतंत्रता से वंचित किया जा सकता है।"
            )

        elif article == "14":
            answer = (
                "भारतीय संविधान का अनुच्छेद 14 कानून के समक्ष "
                "समानता और कानूनों के समान संरक्षण का अधिकार देता है।"
            )

        elif article == "19":
            answer = (
                "भारतीय संविधान का अनुच्छेद 19 नागरिकों को भाषण और "
                "अभिव्यक्ति की स्वतंत्रता सहित कुछ महत्वपूर्ण अधिकार देता है।"
            )

        else:
            answer = (
                f"भारतीय संविधान के अनुच्छेद {article} में "
                f"एक महत्वपूर्ण संवैधानिक प्रावधान दिया गया है।"
            )

        next_steps = [
            f"आप अनुच्छेद {article} का मूल प्रावधान पढ़ सकते हैं।"
        ]

        disclaimer = (
            "यह सामान्य कानूनी जानकारी है, कानूनी सलाह नहीं। "
            "किसी विशेष मामले के लिए योग्य कानूनी पेशेवर से सलाह लें।"
        )

    # --------------------------------------------------
    # Hinglish
    # --------------------------------------------------

    else:

        if article == "21":
            answer = (
                "Indian Constitution ka Article 21 life aur personal "
                "liberty ki protection deta hai. Kisi person ko uski "
                "life ya personal liberty se sirf law ke according "
                "established procedure ke through hi deprive kiya ja sakta hai."
            )

        elif article == "14":
            answer = (
                "Indian Constitution ka Article 14 equality before law "
                "aur laws ke equal protection ka right provide karta hai."
            )

        elif article == "19":
            answer = (
                "Indian Constitution ka Article 19 citizens ko "
                "freedom of speech and expression jaise important rights deta hai."
            )

        else:
            answer = (
                f"Indian Constitution ke Article {article} mein "
                f"ek important constitutional provision diya gaya hai."
            )

        next_steps = [
            f"Aap Article {article} ka exact constitutional provision bhi dekh sakte hain."
        ]

        disclaimer = (
            "Yeh general legal information hai, legal advice nahi. "
            "Specific legal matter ke liye qualified legal professional se consult karein."
        )

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    sources = [{
        "document": "Constitution of India",
        "article": article,
        "page": page
    }]

    return {
        "answer": answer,
        "sources": sources,
        "next_steps": next_steps,
        "disclaimer": disclaimer
    }


def generate_fundamental_rights_answer(query, context_results):

    rights = {
        "Right to Equality": ["14", "15", "16", "17", "18"],

        "Right to Freedom": [
            "19", "20", "21", "21A", "22"
        ],

        "Right against Exploitation": [
            "23", "24"
        ],

        "Right to Freedom of Religion": [
            "25", "26", "27", "28"
        ],

        "Cultural and Educational Rights": [
            "29", "30"
        ],

        "Right to Constitutional Remedies": [
            "32"
        ]
    }

    answer = []

    answer.append(
        "Haan. Indian Constitution ke Part III mein "
        "Fundamental Rights diye gaye hain. Ye mainly "
        "Articles 12 se 35 ke framework mein hain."
    )

    answer.append("\nMain Fundamental Rights:\n")

    for right_name, article_numbers in rights.items():

        articles_found = []

        for number in article_numbers:

            for result in context_results:

                if str(result["article"]).upper() == number.upper():

                    articles_found.append(number)
                    break

        if articles_found:

            article_text = ", ".join(
                f"Article {a}"
                for a in articles_found
            )

            answer.append(
                f"• {right_name}: {article_text}"
            )

    answer.append(
        "\nSimple terms mein, Fundamental Rights "
        "equality, freedom, personal liberty, religious "
        "freedom, protection from exploitation aur "
        "constitutional remedies jaise important protections "
        "provide karte hain."
    )

    answer.append(
        "\nDISCLAIMER:\n"
        "Yeh general legal information hai, legal advice nahi. "
        "Specific legal matter ke liye qualified legal professional "
        "se consult karein."
    )

    return "\n\n".join(answer)


