import streamlit as st
from rag import ask_question

st.set_page_config(
    page_title="RAG-Based AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 RAG-Based AI Teaching Assistant")

st.write(
    "Ask any question related to the Sigma Web Development course."
)

question = st.text_input(
    "Ask your question"
)

if st.button("Ask AI"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching through lecture videos..."):

            answer, chunks = ask_question(question)

        st.success("Answer")

        st.write(answer)

        st.divider()

        st.subheader("Relevant Video Chunks")

        st.dataframe(
            chunks[
                [
                    "title",
                    "number",
                    "start",
                    "end"
                ]
            ],
            use_container_width=True
        )