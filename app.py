from __future__ import annotations

import logging
import os

import streamlit as st

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_RETRIEVAL_K, MAX_FILE_SIZE_MB, THEME_OPTIONS
from database.db_manager import DatabaseManager
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.document_processor import DocumentProcessingError, process_upload
from modules.embeddings import create_embeddings
from modules.question_generator import generate_suggested_questions
from modules.rag_pipeline import RagPipeline, create_chat_model
from modules.summarizer import summarize_document
from modules.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_database() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource
def get_embeddings(model_name: str) -> Embeddings:
    return create_embeddings(model_name)


@st.cache_resource
def get_vector_store(model_name: str) -> VectorStore:
    return VectorStore(get_embeddings(model_name))


def get_api_key() -> str:
    return st.session_state.get("gemini_api_key", os.environ.get("GOOGLE_API_KEY", "")).strip()


def get_chat_model() -> ChatGoogleGenerativeAI:
    api_key = get_api_key()
    if not api_key:
        raise ValueError("Add a Gemini API key in Settings or configure GOOGLE_API_KEY in your environment.")
    return create_chat_model(api_key)


def apply_styles(theme: str) -> None:
    dark = theme == "Dark"
    background = "#101827" if dark else "#f5f7fb"
    surface = "#182235" if dark else "#ffffff"
    text = "#e7edf7" if dark else "#1c2940"
    muted = "#a8b4c7" if dark else "#65748b"
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {"dark" if dark else "light"}; }}
        .stApp {{ background: {background}; color: {text}; }}
        [data-testid="stSidebar"] {{ background: {surface}; border-right: 1px solid {"#29364b" if dark else "#e5eaf2"}; }}
        [data-testid="stMetric"] {{ background: {surface}; border: 1px solid {"#29364b" if dark else "#e8edf5"}; padding: 18px; border-radius: 16px; }}
        [data-testid="stChatMessage"] {{ border-radius: 14px; }}
        .hero-card {{ padding: 28px; border-radius: 20px; color: white; background: linear-gradient(120deg, #2563eb, #7047d7 70%, #13a8c7); box-shadow: 0 12px 30px rgba(37, 99, 235, .18); }}
        .hero-card h1 {{ color: white; margin: 0 0 8px; }}
        .hero-card p {{ color: #e8f0ff; margin: 0; }}
        .muted {{ color: {muted}; }}
        div.stButton > button {{ border-radius: 10px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(db: DatabaseManager) -> str:
    with st.sidebar:
        if "navigation" in st.session_state:
            st.session_state["navigation_selection"] = st.session_state.pop("navigation")
        st.markdown("## 🧠 DocuMind AI")
        st.caption("Your Documents. Your Knowledge. One Intelligent Assistant.")
        page = st.radio(
            "Navigation",
            ["Dashboard", "Chat with Documents", "My Documents", "Document Summary", "Settings", "About Project"],
            label_visibility="collapsed",
            key="navigation_selection",
        )
        st.divider()
        documents = db.list_documents()
        st.metric("Indexed documents", len(documents))
        st.caption(f"{sum(doc['chunk_count'] for doc in documents)} searchable chunks")
        if st.button("＋ New conversation", use_container_width=True):
            conversation_id = db.create_conversation()
            st.session_state["conversation_id"] = conversation_id
            st.session_state["navigation"] = "Chat with Documents"
            st.session_state.pop("chat_history", None)
            st.rerun()
        conversations = db.list_conversations()
        if conversations:
            st.markdown("**Recent conversations**")
            for conversation in conversations[:6]:
                title = conversation["title"]
                if st.button(title, key=f"conversation-{conversation['id']}", use_container_width=True):
                    st.session_state["conversation_id"] = conversation["id"]
                    st.session_state["navigation"] = "Chat with Documents"
                    st.session_state.pop("chat_history", None)
                    st.rerun()
    return page


def get_conversation_id(db: DatabaseManager) -> int:
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id is None or db.get_conversation(conversation_id) is None:
        conversation_id = db.create_conversation()
        st.session_state["conversation_id"] = conversation_id
    return conversation_id


def render_dashboard(db: DatabaseManager) -> None:
    st.markdown(
        '<div class="hero-card"><h1>Welcome to DocuMind AI</h1>'
        "<p>Upload your documents, explore their ideas, and get answers grounded in your sources.</p></div>",
        unsafe_allow_html=True,
    )
    documents = db.list_documents()
    total_pages = sum(doc["page_count"] or 0 for doc in documents)
    total_chunks = sum(doc["chunk_count"] for doc in documents)
    total_questions = db.count_messages(role="user")
    cols = st.columns(4)
    cols[0].metric("Documents", len(documents))
    cols[1].metric("PDF pages", total_pages)
    cols[2].metric("Text chunks", total_chunks)
    cols[3].metric("Questions asked", total_questions)
    st.subheader("Get started")
    left, right = st.columns(2)
    with left:
        st.markdown("### 📄 Build your library")
        st.write("Upload PDF or TXT files to index them for private, source-grounded search.")
        if st.button("Open document library", type="primary"):
            st.session_state["navigation"] = "My Documents"
            st.rerun()
    with right:
        st.markdown("### 💬 Ask your documents")
        st.write("Ask questions and inspect the source passages behind each response.")
        if st.button("Start asking questions"):
            st.session_state["navigation"] = "Chat with Documents"
            st.rerun()
    if documents:
        st.subheader("Recently added")
        for doc in documents[:5]:
            st.write(f"📄 **{doc['filename']}** · {doc['chunk_count']} chunks · {doc['uploaded_at'][:10]}")


def render_upload(db: DatabaseManager, store: VectorStore) -> None:
    st.subheader("Upload documents")
    st.caption(f"PDF and TXT files · up to {MAX_FILE_SIZE_MB} MB each · duplicate files are detected automatically.")
    uploads = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="document_upload",
    )
    if uploads:
        for uploaded in uploads:
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            st.caption(f"{uploaded.name} · {size_mb:.2f} MB")
    chunk_size = int(st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
    if uploads and st.button("Process selected files", type="primary"):
        progress = st.progress(0, text="Preparing documents…")
        added = 0
        latest_document_id = None
        for index, uploaded in enumerate(uploads):
            try:
                processed = process_upload(uploaded.name, uploaded.getvalue(), chunk_size, chunk_overlap)
                existing = db.get_document_by_hash(processed.content_hash)
                if existing:
                    st.info(f"{processed.filename} is already in your library.")
                else:
                    store.add_documents(processed.chunks)
                    try:
                        db.add_document(
                            document_id=processed.document_id,
                            filename=processed.filename,
                            content_hash=processed.content_hash,
                            file_type=processed.file_type,
                            page_count=processed.page_count,
                            chunk_count=len(processed.chunks),
                        )
                    except Exception:
                        store.delete_document(processed.document_id)
                        raise
                    added += 1
                    latest_document_id = processed.document_id
                    st.success(f"Indexed {processed.filename}: {len(processed.chunks)} chunks.")
            except DocumentProcessingError as exc:
                st.error(f"Could not process {uploaded.name}: {exc}")
            except Exception as exc:
                logger.error("Document indexing failed (%s).", type(exc).__name__)
                st.error(f"Could not index {uploaded.name}. Check the application log and try again.")
            progress.progress((index + 1) / len(uploads), text=f"Processed {index + 1} of {len(uploads)}")
        if added:
            if get_api_key():
                try:
                    st.session_state["suggested_questions"] = generate_suggested_questions(
                        get_chat_model(), store, latest_document_id
                    )
                    st.session_state["suggested_questions_doc_id"] = latest_document_id
                except Exception as exc:
                    logger.error("Automatic question generation failed (%s).", type(exc).__name__)
                    st.warning("The documents were indexed, but suggested questions could not be generated.")
            else:
                st.session_state.pop("suggested_questions", None)
                st.session_state.pop("suggested_questions_doc_id", None)
            st.toast(f"Added {added} document{'s' if added != 1 else ''} to your library.")


def render_chat(db: DatabaseManager, store: VectorStore) -> None:
    st.title("Chat with your documents")
    docs = db.list_documents()
    if not docs:
        st.info("Upload a PDF or TXT document in **My Documents** to start asking questions.")
        return
    options = [("All documents", None)] + [
        (f"{doc['filename']} · {doc['document_id'][:8]}", doc["document_id"]) for doc in docs
    ]
    preferred = st.session_state.pop("chat_document_id", None)
    preferred_index = next(
        (index for index, (_label, document_id) in enumerate(options[1:], start=1) if document_id == preferred),
        0,
    )
    selected_name = st.selectbox("Search in", [label for label, _document_id in options], index=preferred_index)
    selected_document_id = dict(options)[selected_name]
    conversation_id = get_conversation_id(db)
    conversation = db.get_conversation(conversation_id)
    if conversation:
        with st.popover("Conversation"):
            title = st.text_input("Conversation title", value=conversation["title"], key=f"title-{conversation_id}")
            if st.button("Save title", key=f"rename-{conversation_id}"):
                db.rename_conversation(conversation_id, title)
                st.toast("Conversation renamed.")
            if st.button("Delete conversation", key=f"delete-conversation-{conversation_id}"):
                db.delete_conversation(conversation_id)
                st.session_state.pop("conversation_id", None)
                st.rerun()
    if st.button("Clear this conversation"):
        db.clear_conversation(conversation_id)
        st.rerun()

    messages = db.get_messages(conversation_id)
    history = [{"role": message["role"], "content": message["content"]} for message in messages]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                sources = db.get_message_sources(message["id"])
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for source in sources:
                            page = f" · page {source['page']}" if source["page"] else ""
                            st.markdown(f"**{source['filename']}**{page}")
                            st.caption(source["snippet"])

    suggestions = st.session_state.get("suggested_questions", [])
    suggestions_document_id = st.session_state.get("suggested_questions_doc_id")
    show_suggestions = bool(suggestions) and (
        selected_document_id is None or suggestions_document_id in (None, selected_document_id)
    )
    if show_suggestions:
        st.markdown("**Suggested questions**")
        for i, question in enumerate(suggestions):
            if st.button(question, key=f"suggest-{i}", use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()
    if st.button("Generate suggested questions"):
        try:
            model = get_chat_model()
            st.session_state["suggested_questions"] = generate_suggested_questions(
                model, store, selected_document_id
            )
            st.session_state["suggested_questions_doc_id"] = selected_document_id
            st.rerun()
        except Exception as exc:
            logger.error("Question generation failed (%s).", type(exc).__name__)
            st.error("Could not generate suggested questions. Check your Gemini API key and try again.")

    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about your documents…")
    if pending:
        question = pending
    if question:
        db.add_message(conversation_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
        try:
            pipeline = RagPipeline(store, get_chat_model())
            with st.chat_message("assistant"):
                with st.spinner("Searching your documents…"):
                    response = pipeline.answer(
                        question,
                        history,
                        document_id=selected_document_id,
                        top_k=int(st.session_state.get("retrieval_k", DEFAULT_RETRIEVAL_K)),
                        relevance_threshold=float(st.session_state.get("relevance_threshold", 0.2)),
                    )
                st.markdown(response.answer)
                if response.sources:
                    with st.expander(f"Sources ({len(response.sources)})", expanded=True):
                        for source in response.sources:
                            page = f" · page {source['page']}" if source.get("page") else ""
                            st.markdown(f"**{source['filename']}**{page}")
                            st.caption(source["snippet"])
            assistant_id = db.add_message(conversation_id, "assistant", response.answer)
            for source in response.sources:
                db.add_message_source(
                    assistant_id,
                    source["filename"],
                    source.get("page"),
                    source["snippet"],
                )
            st.rerun()
        except ValueError as exc:
            logger.info("Chat request configuration issue: %s", exc)
            st.error(str(exc))
        except Exception as exc:
            logger.error("Chat request failed (%s).", type(exc).__name__)
            st.error("The answer could not be generated. Check your Gemini API key and connection, then try again.")


def render_library(db: DatabaseManager, store: VectorStore) -> None:
    st.title("My Documents")
    render_upload(db, store)
    st.divider()
    st.subheader("Document library")
    documents = db.list_documents()
    query = st.text_input("Search documents by name", placeholder="Search your library…").casefold()
    documents = [doc for doc in documents if query in doc["filename"].casefold()]
    if not documents:
        st.info("No matching documents yet.")
        return
    for doc in documents:
        with st.container(border=True):
            info, actions = st.columns([4, 2])
            with info:
                st.markdown(f"**{doc['filename']}**")
                pages = f"{doc['page_count']} pages" if doc["page_count"] is not None else "Text document"
                st.caption(
                    f"{doc['file_type'].upper()} · {pages} · {doc['chunk_count']} chunks · "
                    f"Added {doc['uploaded_at'][:10]} · Indexed"
                )
            with actions:
                detail_key = f"details-{doc['document_id']}"
                if st.button("Details", key=detail_key):
                    st.session_state["details_document"] = doc["document_id"]
                if st.button("Chat", key=f"chat-document-{doc['document_id']}"):
                    st.session_state["chat_document_id"] = doc["document_id"]
                    st.session_state["navigation"] = "Chat with Documents"
                    st.rerun()
                if st.button("Summarize", key=f"summary-document-{doc['document_id']}"):
                    st.session_state["summary_document_id"] = doc["document_id"]
                    st.session_state["navigation"] = "Document Summary"
                    st.rerun()
                if st.button("Delete", key=f"delete-document-{doc['document_id']}", type="secondary"):
                    try:
                        store.delete_document(doc["document_id"])
                        db.delete_document(doc["document_id"])
                        st.success(f"Deleted {doc['filename']} and its indexed chunks.")
                        st.rerun()
                    except Exception as exc:
                        logger.error("Document deletion failed (%s).", type(exc).__name__)
                        st.error("Could not delete this document. Please try again.")
            if st.session_state.get("details_document") == doc["document_id"]:
                st.caption(f"Document ID: {doc['document_id']} · SHA-256 duplicate check enabled")


def render_summary(db: DatabaseManager, store: VectorStore) -> None:
    st.title("Document Summary")
    docs = db.list_documents()
    if not docs:
        st.info("Upload a document before generating a summary.")
        return
    preferred = st.session_state.pop("summary_document_id", None)
    options = [
        (f"{doc['filename']} · {doc['document_id'][:8]}", doc) for doc in docs
    ]
    default_index = next((i for i, (_label, doc) in enumerate(options) if doc["document_id"] == preferred), 0)
    selected = st.selectbox("Choose a document", [label for label, _doc in options], index=default_index)
    doc = dict(options)[selected]
    detail = st.radio("Summary style", ["Concise", "Detailed"], horizontal=True)
    scope = st.radio("Summary scope", ["Entire document", "Individual section"], horizontal=True)
    section_index = None
    if scope == "Individual section":
        sections = store.get_document_chunks(doc["document_id"])
        if not sections:
            st.warning("No indexed sections are available.")
            return
        section_index = st.selectbox(
            "Section",
            range(len(sections)),
            format_func=lambda index: (
                f"Section {index + 1} · page {sections[index].metadata['page']}"
                if sections[index].metadata.get("page")
                else f"Section {index + 1}"
            ),
        )
    if st.button("Generate summary", type="primary"):
        try:
            model = get_chat_model()
            with st.spinner("Summarizing document content…"):
                result = summarize_document(
                    model,
                    store,
                    doc["document_id"],
                    detail.lower(),
                    chunk_index=section_index,
                )
            st.markdown(result)
        except Exception as exc:
            logger.error("Document summarization failed (%s).", type(exc).__name__)
            st.error("Could not summarize this document. Check your Gemini API key and try again.")


def render_settings() -> None:
    st.title("Settings")
    st.subheader("Gemini API")
    st.caption("Your key is used only for this application session. Set GOOGLE_API_KEY to persist it between runs.")
    env_key = os.environ.get("GOOGLE_API_KEY", "")
    key = st.text_input("Gemini API key", value=st.session_state.get("gemini_api_key", env_key), type="password")
    if st.button("Use this key"):
        if key.strip():
            st.session_state["gemini_api_key"] = key.strip()
            st.success("API key set for this session.")
        else:
            st.session_state.pop("gemini_api_key", None)
            st.warning("No API key was entered.")

    st.subheader("Retrieval and chunking")
    st.number_input("Chunk size (characters)", min_value=200, max_value=4000, step=100, value=int(st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE)), key="chunk_size")
    st.number_input("Chunk overlap (characters)", min_value=0, max_value=1000, step=50, value=int(st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)), key="chunk_overlap")
    st.number_input("Retrieval Top K", min_value=1, max_value=12, step=1, value=int(st.session_state.get("retrieval_k", DEFAULT_RETRIEVAL_K)), key="retrieval_k")
    st.slider("Minimum relevance score", min_value=0.0, max_value=1.0, step=0.05, value=float(st.session_state.get("relevance_threshold", 0.2)), key="relevance_threshold")
    st.caption("Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    st.selectbox("Theme", THEME_OPTIONS, key="theme")

    db = get_database()
    st.subheader("Data management")
    if st.button("Clear chat history"):
        db.clear_all_conversations()
        st.success("All saved conversations were deleted.")
    with st.popover("Clear vector database"):
        st.warning("This permanently removes all indexed documents and their vectors.")
        if st.button("Confirm clear all documents", type="primary"):
            store = get_vector_store("sentence-transformers/all-MiniLM-L6-v2")
            store.clear()
            db.clear_documents()
            st.success("The document library and vector database were cleared.")
            st.rerun()


def render_about() -> None:
    st.title("About DocuMind AI")
    st.write(
        "DocuMind AI is a local-first document assistant. It extracts PDF/TXT text, creates "
        "overlapping semantic chunks, indexes them in persistent ChromaDB, and uses Gemini "
        "to answer questions from retrieved passages."
    )
    st.markdown(
        """
        **RAG flow:** Upload → PyPDF/TXT extraction → chunking with source metadata → HuggingFace
        embeddings → persistent ChromaDB retrieval → Gemini response with citations.

        **Privacy:** Uploaded originals are processed in memory and are not saved to disk. The
        local ChromaDB index and SQLite conversation history remain on this machine. Gemini API
        requests send the question and retrieved document passages to Google.
        """
    )


def main() -> None:
    db = get_database()
    if "theme" not in st.session_state:
        st.session_state["theme"] = THEME_OPTIONS[0]
    page = render_sidebar(db)
    apply_styles(st.session_state.get("theme", "Light"))
    store = None
    if page in {"Chat with Documents", "My Documents", "Document Summary"}:
        try:
            store = get_vector_store("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:
            logger.error("Embedding model initialization failed (%s).", type(exc).__name__)
            st.error("The embedding model could not be loaded. Check your internet connection and installed dependencies.")
            return

    if page == "Dashboard":
        render_dashboard(db)
    elif page == "Chat with Documents":
        assert store is not None
        render_chat(db, store)
    elif page == "My Documents":
        assert store is not None
        render_library(db, store)
    elif page == "Document Summary":
        assert store is not None
        render_summary(db, store)
    elif page == "Settings":
        render_settings()
    else:
        render_about()


if __name__ == "__main__":
    main()
