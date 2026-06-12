import streamlit as st
import uuid
import requests
import json

from api_client import BackendClient, BackendAPIError, BASE_URL as BACKEND_URL

st.set_page_config(page_title="GenAI Doc Assistant", layout="wide")

client = BackendClient()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalize_observability(obs):
    if isinstance(obs, str):
        try:
            return json.loads(obs)
        except json.JSONDecodeError:
            return {"raw": obs}

    if obs is None:
        return {}

    if not isinstance(obs, dict):
        return {"raw": str(obs)}

    return obs


def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id


def reset_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


@st.cache_data(ttl=60)
def cached_upload_limits():
    """Fetch max size and allowed types from backend (single source of truth)."""
    try:
        return client.upload_limits()
    except BackendAPIError:
        return {
            "max_file_size_mb": 10,
            "allowed_file_types": ["pdf", "txt", "csv", "xlsx", "json", "yaml", "yml"],
        }


def render_observability(obs: dict):
    obs = _normalize_observability(obs)
    with st.expander("📊 Observability Metadata", expanded=False):
        if not obs:
            st.info("No observability metadata returned.")
            return
        st.markdown("### Raw Observability Payload")
        st.json(obs)


# ---------------------------------------------------------
# Cache documents list
# ---------------------------------------------------------
@st.cache_data(ttl=5)
def cached_list_documents():
    return client.list_documents()


# ---------------------------------------------------------
# Init global observability history
# ---------------------------------------------------------
if "observability_history" not in st.session_state:
    st.session_state.observability_history = []


# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("📄 GenAI Document Assistant")
st.caption("Upload → Ingest → Ask Questions → HITL Document Selection")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Dynamic uploader key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# HITL state initialization
for key, default in {
    "hitl_active": False,
    "hitl_candidates": None,
    "hitl_question": None,
    "hitl_thread_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------
# Sidebar: Upload + List Documents + Observability
# ---------------------------------------------------------
with st.sidebar:
    st.header("📤 Upload Document")

    upload_limits = cached_upload_limits()
    max_mb = upload_limits.get("max_file_size_mb", 10)
    allowed_types = upload_limits.get("allowed_file_types") or [
        "pdf", "txt", "csv", "xlsx", "json", "yaml", "yml"
    ]
    types_label = ", ".join(f".{ext}" for ext in allowed_types)
    st.caption(f"Max {max_mb} MB per file · Allowed: {types_label}")

    uploaded_file = st.file_uploader(
        f"Upload a file (max {max_mb} MB)",
        type=allowed_types,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file:
        try:
            with st.spinner("Uploading and ingesting..."):
                result = client.upload_document(uploaded_file)
        except BackendAPIError as exc:
            st.error(str(exc))
            if exc.correlation_id:
                st.caption(f"Correlation ID: `{exc.correlation_id}`")
            st.stop()

        if isinstance(result, dict) and result.get("duplicate"):
            st.info(result.get("message", "Duplicate file skipped."))
            st.json(result)
        elif isinstance(result, dict) and "error" not in result:
            st.success("Document ingested successfully!")
            st.json(result)

            # Push observability to global history
            obs = result.get("observability")
            if obs:
                st.session_state.observability_history.append({
                    "label": f"Upload: {uploaded_file.name}",
                    "endpoint": "/upload-and-ingest",
                    "correlation_id": result.get("correlation_id"),
                    "model": result.get("model"),
                    "title": result.get("title"),
                    "summary": result.get("summary"),
                    "num_chunks": result.get("num_chunks"),
                    "observability": obs,
                })

        # Reset uploader widget completely
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.header("📚 Documents in Vector DB")

    with st.expander("Backend connection", expanded=False):
        st.code(BACKEND_URL, language=None)
        if st.button("Test backend /health"):
            try:
                import requests
                r = requests.get(f"{BACKEND_URL}/health", timeout=120)
                st.write(f"HTTP {r.status_code}: {r.text[:200]}")
            except Exception as exc:
                st.error(str(exc))

    try:
        docs = cached_list_documents()
    except BackendAPIError as exc:
        st.error(str(exc))
        if exc.status_code:
            st.caption(f"HTTP status: {exc.status_code}")
        if exc.correlation_id:
            st.caption(f"Correlation ID: `{exc.correlation_id}`")
        st.info(
            "If the backend is still starting, wait a moment and refresh. "
            "Check `GET /health` on the API (port 8000) if the problem persists."
        )
        docs = {}

    if isinstance(docs, dict) and "documents" in docs:
        if not docs.get("documents"):
            st.caption("No documents yet — upload a file above.")
        for d in docs.get("documents", []):
            st.write(f"**{d['title']}** ({d.get('filename','unknown')}) — {d['summary']}")
    elif isinstance(docs, dict) and docs.get("error"):
        st.error(docs["error"])

    st.divider()
    st.header("🩺 Observability")

    if st.button("Run Observability Test"):
        with st.spinner("Running observability pipeline..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/observability")
                data = resp.json()
            except Exception as e:
                st.error(f"Observability request failed: {e}")
                data = None

        if data:
            if data.get("status") == "ok" or data.get("message", "").lower().startswith("observability test ok"):
                st.success("Observability test completed successfully")

                st.markdown("### 🔍 Core Results")
                st.json({
                    "correlation_id": data.get("correlation_id"),
                    "model_used": data.get("model"),
                    "answer": data.get("answer")
                })

                obs = data.get("observability")
                render_observability(obs)

                # Push test trace to global history
                if obs:
                    st.session_state.observability_history.append({
                        "label": "Observability Test",
                        "endpoint": "/observability",
                        "correlation_id": data.get("correlation_id"),
                        "model": data.get("model"),
                        "answer": data.get("answer"),
                        "observability": obs,
                    })
            else:
                st.error("Observability test failed")
                st.json(data)


# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.subheader("💬 Chat with your documents")

thread_id = get_thread_id()

for msg in st.session_state.messages:
    role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Assistant"
    st.chat_message(msg["role"]).write(f"**{role}:** {msg['content']}")


# ---------------------------------------------------------
# HITL UI
# ---------------------------------------------------------
if st.session_state.hitl_active:
    st.warning("Multiple documents match your question. Please choose one.")

    candidates = st.session_state.hitl_candidates
    options = {
        f"{d['title']} ({d.get('filename','unknown')}) — {d['summary']}": d["doc_id"]
        for d in candidates
    }

    choice = st.radio("Select the most relevant document:", list(options.keys()))

    if st.button("Submit Choice"):
        selected_doc_id = options[choice]
        hitl_question = st.session_state.hitl_question or ""
        hitl_thread_id = st.session_state.hitl_thread_id

        try:
            with st.spinner("Resuming pipeline..."):
                final = client.choose_document(
                    thread_id=hitl_thread_id,
                    question=hitl_question,
                    selected_doc_id=selected_doc_id,
                )
        except BackendAPIError as exc:
            st.error(str(exc))
            if exc.correlation_id:
                st.caption(f"Correlation ID: `{exc.correlation_id}`")
            st.stop()

        st.session_state.hitl_active = False
        st.session_state.hitl_candidates = None
        st.session_state.hitl_question = None
        st.session_state.hitl_thread_id = None

        answer = final["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        obs = final.get("observability")
        render_observability(obs)

        if obs:
            label_question = final.get("question") or hitl_question or "document selection"
            st.session_state.observability_history.append({
                "label": f"HITL: {label_question[:40]}...",
                "endpoint": "/choose-document",
                "correlation_id": final.get("correlation_id"),
                "model": final.get("model"),
                "answer": final.get("answer"),
                "observability": obs,
            })

        st.rerun()


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        with st.spinner("Thinking..."):
            response = client.ask_question(user_input, thread_id)
    except BackendAPIError as exc:
        st.error(str(exc))
        if exc.correlation_id:
            st.caption(f"Correlation ID: `{exc.correlation_id}`")
        st.stop()

    if response.get("cache_hit"):
        st.caption("Answer served from cache.")

    if response.get("needs_user_choice"):
        st.session_state.hitl_active = True
        st.session_state.hitl_candidates = response["candidate_documents"]
        st.session_state.hitl_question = user_input
        st.session_state.hitl_thread_id = thread_id
        st.rerun()
    else:
        answer = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        obs = response.get("observability")
        render_observability(obs)

        if obs:
            st.session_state.observability_history.append({
                "label": f"Query: {user_input[:40]}...",
                "endpoint": "/ask-question",
                "correlation_id": response.get("correlation_id"),
                "model": response.get("model"),
                "answer": answer,
                "observability": obs,
            })


# ---------------------------------------------------------
# Reset conversation
# ---------------------------------------------------------
st.divider()
if st.button("🔄 Start New Conversation"):
    reset_thread()
    st.rerun()
