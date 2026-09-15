"""Single-file Streamlit demo for the Hybrid RAG pipeline.

Run with:
    streamlit run streamlit_app.py

Builds the pipeline directly via src/pipeline.py (no need to run the FastAPI
service separately) and caches it with st.cache_resource so it is built once
per process, not once per interaction.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import build_pipeline

st.set_page_config(page_title="Hybrid RAG Demo", page_icon="🔎")


@st.cache_resource(show_spinner="Building index and loading models (first run only)...")
def get_pipeline():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    data_dir = os.getenv("DATA_DIR", "./data/")
    return build_pipeline(openai_api_key=api_key, data_dir=data_dir)


st.title("Hybrid RAG Demo")
st.caption(
    "Dense + BM25 retrieval, RRF fusion, cross-encoder reranking, "
    "citation-verified generation."
)

query = st.text_input("Ask a question about the indexed documents:")
submitted = st.button("Submit", type="primary")

if submitted:
    if not query or not query.strip():
        st.warning("Please enter a question first.")
    else:
        pipeline = get_pipeline()
        with st.spinner("Retrieving and generating an answer..."):
            result = pipeline.generate_robust_answer(query)

        status = result.get("status", "Unknown")

        if status == "Success":
            st.subheader("Answer")
            st.write(result.get("answer", ""))

            confidence = result.get("confidence_metrics", {})
            if confidence:
                st.subheader("Confidence")
                cols = st.columns(4)
                cols[0].metric("Composite", f"{confidence.get('composite_score', 0):.2f}")
                cols[1].metric("Retrieval", f"{confidence.get('retrieval_confidence', 0):.2f}")
                cols[2].metric("Citation coverage", f"{confidence.get('citation_coverage', 0):.2f}")
                cols[3].metric("Completeness", f"{confidence.get('answer_completeness', 0):.2f}")

            flagged = result.get("flagged_citations", [])
            st.subheader("Citations")
            if flagged:
                st.warning(f"{len(flagged)} citation(s) flagged as unsupported by the source text:")
                for c in flagged:
                    st.markdown(f"- **Claim:** {c.get('claim')}  \n  **Reasoning:** {c.get('reasoning')}")
            else:
                st.success("All citations were verified as supported by the retrieved context.")

            retrieved_chunks = result.get("retrieved_chunks", [])
            if retrieved_chunks:
                with st.expander(f"Retrieved context ({len(retrieved_chunks)} chunks)"):
                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        source = chunk.metadata.get("filepath", "Unknown source")
                        st.markdown(f"**[{i}] {source}**")
                        st.text(chunk.page_content)
        else:
            st.subheader("Insufficient Information")
            st.write(result.get("reason", ""))
            st.write(result.get("found_context", ""))
            suggested = result.get("suggested_documents")
            if suggested:
                st.markdown("**Suggested documents to review:**")
                for doc in suggested:
                    st.markdown(f"- {doc}")
