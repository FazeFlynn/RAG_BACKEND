"""
Streamlit UI for the RAG System.
Run: streamlit run app/ui/streamlit_app.py
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8011"

st.set_page_config(page_title="RAG System", page_icon="🔍", layout="wide")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📁 Document Management")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md", "csv", "xls", "xlsx", "docx", "html", "json"],
    )

    if uploaded_file and st.button("📤 Upload & Index"):
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data['message']}")
                else:
                    st.error(f"❌ {resp.json().get('detail', 'Upload failed')}")
            except requests.ConnectionError:
                st.error("❌ Cannot connect to API. Is the server running?")

    st.divider()

    # List indexed documents
    st.subheader("Indexed Documents")
    try:
        resp = requests.get(f"{API_URL}/documents/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data["documents"]:
                for doc in data["documents"]:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"📄 {doc}")
                    if col2.button("🗑️", key=f"del_{doc}"):
                        requests.delete(f"{API_URL}/documents/{doc}", timeout=10)
                        st.rerun()
                st.caption(f"Total: {data['total']} documents")
            else:
                st.caption("No documents uploaded yet")
    except requests.ConnectionError:
        st.caption("⚠️ API not available")

    st.divider()

    # Query mode selector
    st.subheader("⚙️ Settings")
    query_mode = st.selectbox(
        "Query Mode",
        ["Auto-detect", "Document Q&A", "Web Search", "Hybrid"],
        index=0,
    )

    mode_map = {
        "Auto-detect": None,
        "Document Q&A": "document_qa",
        "Web Search": "web_search",
        "Hybrid": "hybrid",
    }

    # Health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            st.divider()
            status_color = "🟢" if health["status"] == "healthy" else "🟡"
            st.caption(f"{status_color} API: {health['status']}")
            st.caption(f"{'🟢' if health['ollama_connected'] else '🔴'} Ollama: {'connected' if health['ollama_connected'] else 'disconnected'}")
            st.caption(f"📊 Indexed chunks: {health['documents_indexed']}")
    except requests.ConnectionError:
        st.caption("🔴 API not running")


# --- Main Chat Interface ---
st.title("🔍 RAG System")
st.caption("Ask questions about your documents or search the web")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources if available
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**" + (f" (Page {src['page']})" if src.get('page') else ""))
                    st.caption(src["content"][:200] + "...")

        if msg.get("web_sources"):
            with st.expander("🌐 Web Sources"):
                for ws in msg["web_sources"]:
                    st.markdown(f"[{ws['title']}]({ws['url']})")
                    st.caption(ws["snippet"][:200])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "query": prompt,
                    "conversation_id": st.session_state.conversation_id,
                }
                if mode_map[query_mode]:
                    payload["query_type"] = mode_map[query_mode]

                resp = requests.post(f"{API_URL}/chat/", json=payload, timeout=180)

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.conversation_id = data.get("conversation_id")

                    st.markdown(data["answer"])

                    msg_data = {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "web_sources": data.get("web_sources", []),
                    }
                    st.session_state.messages.append(msg_data)

                    # Show sources
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for src in data["sources"]:
                                st.markdown(f"**{src['source']}**" + (f" (Page {src['page']})" if src.get('page') else ""))
                                st.caption(src["content"][:200] + "...")

                    if data.get("web_sources"):
                        with st.expander("🌐 Web Sources"):
                            for ws in data["web_sources"]:
                                st.markdown(f"[{ws['title']}]({ws['url']})")
                                st.caption(ws["snippet"][:200])

                    # Show query type badge
                    qt = data.get("query_type", "unknown")
                    badge = {"document_qa": "📄 Document Q&A", "web_search": "🌐 Web Search", "hybrid": "🔀 Hybrid"}
                    st.caption(f"Mode: {badge.get(qt, qt)}")

                else:
                    error = resp.json().get("detail", "Unknown error")
                    st.error(f"Error: {error}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error}"})

            except requests.ConnectionError:
                st.error("Cannot connect to the API server. Make sure it's running on http://localhost:8011")
                st.session_state.messages.append({"role": "assistant", "content": "Error: API server not available"})
