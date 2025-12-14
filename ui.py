import streamlit as st
import requests

# ---- CONFIG ----
BACKEND_URL = "http://127.0.0.1:8000/query"  # FastAPI backend endpoint

st.set_page_config(page_title="Joarmanaj Agency RAG Assistant", page_icon="🤖")

st.title("Joarmanaj Agency Digital Marketing Assistant")
st.write("Ask questions about our services. Answers come from retrieved context documents.")

# ---- USER INPUT ----
question = st.text_input("Your Question:", "")

if st.button("Ask") and question.strip():
    with st.spinner("Fetching answer..."):
        try:
            response = requests.post(
                BACKEND_URL,
                json={"question": question},
                timeout=120
            )
            response.raise_for_status()
            answer = response.json().get("answer", "I don't know.")
            st.success(answer)
        except requests.exceptions.RequestException as e:
            st.error(f"Error contacting the backend: {e}")
