"""
Query router - classifies queries as document Q&A or web search.
"""

from loguru import logger

from app.models.schemas import QueryType
from app.services import vector_store


# Questions about the assistant itself — always General, never web search
IDENTITY_PATTERNS = [
    "who are you", "what are you", "who built you", "who made you",
    "who created you", "who developed you", "who is your creator",
    "what is your name", "your name", "tell me about yourself",
    "introduce yourself", "what can you do", "how do you work",
    "are you an ai", "are you a bot", "are you human",
    "who designed you", "who trained you", "what model are you",
    "who is islam", "islam kathat",
    "what does rag stand for",
    "what is rag",
    "what is a rag system",
]


async def classify_query(query: str, has_documents: bool = False) -> QueryType:
    """Classify whether to use document Q&A, web search, or general chat.

    Logic:
    1. Identity/self-referential questions → always General
    2. Casual greetings → always General
    3. No documents + factual question → web search
    4. Documents exist + doc keywords → document Q&A
    5. Web/knowledge keywords → web search
    6. Ambiguous with documents → document Q&A
    """

    query_lower = query.lower().strip()

    # --- 1. Identity questions: always answer from system prompt, never web search ---
    if any(pattern in query_lower for pattern in IDENTITY_PATTERNS):
        logger.info("Identity/self-referential query, routing to general chat")
        return QueryType.GENERAL

    # --- 2. Casual greetings ---
    casual_patterns = [
        "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
        "thanks", "thank you", "bye", "goodbye", "how are you",
        "help", "ok", "okay", "cool", "great", "awesome",
    ]
    is_casual = any(query_lower.startswith(p) or query_lower == p for p in casual_patterns)

    if is_casual:
        logger.info("Casual query, routing to general chat")
        return QueryType.GENERAL

    # --- 3. Document keywords (only relevant if docs exist) ---
    doc_keywords = [
        # Direct references
        "document", "file", "uploaded", "pdf", "csv", "doc",
        # Natural language references to "the" document
        "the document", "this document", "that document",
        "the file", "this file", "that file",
        "the pdf", "this pdf",
        # Action words people use with documents
        "according to", "in the", "from the", "based on",
        "what does the", "summarize", "extract from",
        "check the", "read the", "look at the", "analyze",
        "what is it about", "what's it about", "tell me about it",
        "what does it say", "what does it contain",
        "give me a summary", "give me an overview",
        "what are the key", "main points", "key points",
    ]
    if has_documents and any(kw in query_lower for kw in doc_keywords):
        logger.info("Query contains document keywords, routing to document Q&A")
        return QueryType.DOCUMENT_QA

    # --- 4. No documents → web search for factual questions ---
    if not has_documents:
        logger.info("No documents indexed, routing to web search")
        return QueryType.WEB_SEARCH

    # --- 5. Web/knowledge keywords (exclude "who is" — too broad, catches identity Qs) ---
    web_keywords = [
        "latest", "today", "current", "news", "how to",
        "tutorial", "search", "find online", "what is the price",
        "weather", "stock", "2024", "2025", "2026",
        "what is", "which is", "where is", "when did", "when was",
        "top 10", "top 5", "list of", "facts", "best",
        "tell me about", "explain", "give me information",
    ]
    if any(kw in query_lower for kw in web_keywords):
        logger.info("Query contains web/knowledge keywords, routing to web search")
        return QueryType.WEB_SEARCH

    # --- 6. Ambiguous with documents → try document Q&A ---
    logger.info("Ambiguous query with documents available, routing to document Q&A")
    return QueryType.DOCUMENT_QA