import streamlit as st
import pandas as pd
from rag_engine import MedicalRAGEngine

# Page Configuration
st.set_page_config(
    page_title="NIH Health Info Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4B44;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5A54;
        margin-bottom: 1.5rem;
    }
    .guardrail-card {
        background-color: #FBEDE9;
        border-left: 5px solid #D98A6E;
        padding: 14px 16px;
        border-radius: 4px;
        margin: 10px 0;
        color: #7A2818;
    }
    .topic-pill {
        display: inline-block;
        background-color: #E6F3F1;
        color: #1F4B44;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .confidence-pill {
        display: inline-block;
        background-color: #F6F4EB;
        color: #8C621E;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .citation-box {
        margin-top: 10px;
        padding: 10px 14px;
        background-color: #F8F9FA;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading medical index and dataset...")
def get_rag_engine():
    return MedicalRAGEngine()


engine = get_rag_engine()

# Sidebar: Assistant Information & Dataset Stats
with st.sidebar:
    st.title("🩺 NIH Medical RAG")
    st.markdown(
        "A grounded, retrieval-based medical question-answering assistant trained on **MedQuAD** (NIH data)."
    )

    st.divider()

    st.subheader("📊 Knowledge Base Stats")
    total_qa = len(engine.df)
    st.metric(label="Total Q&A Pairs", value=f"{total_qa:,}")

    with st.expander("Explore Sources Breakdown"):
        source_counts = engine.df["source"].value_counts().reset_index()
        source_counts.columns = ["NIH Source", "Records"]
        st.dataframe(source_counts, hide_index=True)

    st.divider()

    st.subheader("💡 Example Questions")
    sample_queries = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes Parkinson's disease?",
        "What is the treatment for asthma in children?",
        "What are the risk factors for stroke?",
    ]

    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}"):
            st.session_state["pending_query"] = sq

    st.divider()
    st.caption(
        "⚠️ **Disclaimer**: For informational purposes only. In case of a medical emergency, please call 999 / 911 or visit your nearest emergency department."
    )

# Main Content Area
st.markdown('<div class="main-header">NIH Health Information Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Accurate, citation-backed answers directly grounded in official NIH clinical Q&A documents.</div>',
    unsafe_allow_html=True,
)

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "answer",
            "content": (
                "Hello! I can answer questions about diseases, conditions, tests, and treatments using verified NIH sources. "
                "How can I help you today?"
            ),
            "sources": [],
            "topic": None,
            "confidence": None,
        }
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "guardrail_redirect":
            st.markdown(
                f'<div class="guardrail-card"><strong>⚠️ Medical Safety Notice:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif msg.get("type") == "no_answer":
            st.info(msg["content"])
        else:
            # Display metadata tags if available
            topic = msg.get("topic")
            confidence = msg.get("confidence")
            tags_html = ""
            if topic:
                tags_html += f'<span class="topic-pill">Topic: {topic}</span>'
            if confidence is not None:
                tags_html += f'<span class="confidence-pill">Confidence: {confidence * 100:.1f}%</span>'
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

            st.write(msg["content"])

            # Display citations
            sources = msg.get("sources", [])
            if sources:
                citations_md = "**Sources & Citations:**\n"
                for s in sources:
                    source_name = s.get("source", "NIH")
                    url = s.get("url", "")
                    if url:
                        citations_md += f"- [{source_name}]({url})\n"
                    else:
                        citations_md += f"- {source_name}\n"
                st.markdown(citations_md)

# Handle Query Input (from input box or clicked sample question)
user_query = st.chat_input("Ask a medical or health question...")

if "pending_query" in st.session_state and st.session_state["pending_query"]:
    user_query = st.session_state.pop("pending_query")

if user_query:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Process via RAG Engine
    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge base..."):
            response = engine.answer(user_query)

        resp_type = response.get("type", "answer")
        msg_text = response.get("message", "")
        topic = response.get("topic")
        confidence = response.get("confidence")
        sources = response.get("sources", [])

        if resp_type == "guardrail_redirect":
            st.markdown(
                f'<div class="guardrail-card"><strong>⚠️ Medical Safety Notice:</strong><br>{msg_text}</div>',
                unsafe_allow_html=True,
            )
        elif resp_type == "no_answer":
            st.info(msg_text)
        else:
            tags_html = ""
            if topic:
                tags_html += f'<span class="topic-pill">Topic: {topic}</span>'
            if confidence is not None:
                tags_html += f'<span class="confidence-pill">Confidence: {confidence * 100:.1f}%</span>'
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

            st.write(msg_text)

            if sources:
                citations_md = "**Sources & Citations:**\n"
                for s in sources:
                    source_name = s.get("source", "NIH")
                    url = s.get("url", "")
                    if url:
                        citations_md += f"- [{source_name}]({url})\n"
                    else:
                        citations_md += f"- {source_name}\n"
                st.markdown(citations_md)

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "type": resp_type,
            "content": msg_text,
            "sources": sources,
            "topic": topic,
            "confidence": confidence,
        })
