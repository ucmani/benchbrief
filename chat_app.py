"""
Chat front end for BenchBrief.   Run:  streamlit run chat_app.py
Needs the API running:           uvicorn api.main:app

Two modes (sidebar):
  Analyze a case   -> the message is a case; POST /analyze -> brief + reasoning paths
  Ask a question   -> POST /chat -> answer from retrieved precedent, with citations
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("BENCHBRIEF_API", "http://127.0.0.1:8000")
ANALYZE, ASK = "Analyze a case", "Ask a question"

st.set_page_config(page_title="BenchBrief", page_icon="⚖️")
st.title("BenchBrief")
st.caption("Decision support for judges: summary, precedent and reasoning paths. "
           "It does not give a verdict.")


@st.cache_data(ttl=60)
def get_info() -> dict:
    try:
        return requests.get(f"{API_URL}/info", timeout=5).json()
    except requests.RequestException:
        return {}


with st.sidebar:
    mode = st.radio("Mode", [ASK, ANALYZE],
                    help="Analyze: paste a case to get a brief. Ask: questions about precedent.")
    top_k = st.slider("Precedent passages (top_k)", 1, 12, 8)
    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()
    info = get_info()
    if info:
        st.caption("Model: " + " → ".join(info["generation"]))
        st.caption(f"Reranker: {'on' if info['rerank'] else 'off'}")
    else:
        st.caption("API not reachable")

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role", "content", "sources"?}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"Precedent used ({len(m['sources'])})"):
                for i, s in enumerate(m["sources"], start=1):
                    label = s.get("case_name") or f"Opinion {s.get('opinion_id')}"
                    st.markdown(f"**{i}. {label}** ({s.get('section_label', '')})")
                    st.caption(s.get("text", ""))


def call_analyze(text: str) -> tuple[str, list[dict]]:
    r = requests.post(f"{API_URL}/analyze", timeout=300,
                      json={"current_case_text": text, "top_k": top_k})
    r.raise_for_status()
    body = r.json()
    answer = (f"### Case brief\n{body['case_brief']}\n\n"
              f"### Reasoning paths for your consideration\n{body['recommendation']}\n\n"
              "_Switch to **Ask a question** to dig into this precedent._")
    return answer, body["precedent_chunks"]


def call_chat(history: list[dict]) -> tuple[str, list[dict]]:
    payload = [{"role": m["role"], "content": m["content"]} for m in history]
    r = requests.post(f"{API_URL}/chat", json={"messages": payload, "top_k": top_k}, timeout=300)
    r.raise_for_status()
    body = r.json()
    return body["answer"], body["sources"]


placeholder = "Paste the case to analyze" if mode == ANALYZE else "Ask about precedent"
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt if len(prompt) < 3000 else prompt[:3000] + "\n\n*...(case text truncated in display)*")
    with st.chat_message("assistant"):
        with st.spinner("Retrieving precedent and drafting..."):
            try:
                if mode == ANALYZE:
                    answer, sources = call_analyze(prompt)
                else:
                    answer, sources = call_chat(st.session_state.messages)
            except requests.RequestException as e:
                answer, sources = f"Request failed: {e}\n\nIs `uvicorn api.main:app` running?", []
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()