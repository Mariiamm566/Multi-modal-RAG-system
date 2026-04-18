from pathlib import Path
import streamlit as st

from src.pipeline import RAGPipeline
from src.config import settings

st.set_page_config(page_title="Lightweight Multi-Modal RAG", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "built" not in st.session_state:
    st.session_state.built = False

st.title("Lightweight Multi-Modal Document Intelligence")
st.caption("Text + table retrieval with image-aware CLIP retrieval in a shared text-image embedding space")

with st.sidebar:
    st.subheader("Runtime")
    st.write(f"Text model: `{settings.text_embedding_model}`")
    st.write(f"Image model: `{settings.image_embedding_model}`")
    st.write(f"Top-k text: {settings.top_k_text}")
    st.write(f"Top-k image: {settings.top_k_image}")
    st.write(f"Final evidence count: {settings.final_top_k}")
    st.write(f"Minimum images: {settings.min_images}")
    st.write(f"Minimum score: {settings.min_score}")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Build Index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF file.")
        else:
            settings.ensure()
            paths = []
            for f in uploaded_files:
                save_path = settings.work_dir / f.name
                save_path.write_bytes(f.read())
                paths.append(Path(save_path))
            with st.spinner("Building text, table, and image indexes..."):
                stats = st.session_state.pipeline.build(paths)
            st.session_state.built = True
            st.success("Index built successfully.")
            st.json(stats)

with col2:
    query = st.text_input("Ask a question about the uploaded documents")
    if st.button("Get Answer", use_container_width=True):
        if not st.session_state.built:
            try:
                st.session_state.pipeline.load()
                st.session_state.built = True
            except Exception:
                st.error("Build the index first.")

        if st.session_state.built:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving evidence and generating the answer..."):
                    answer, evidence = st.session_state.pipeline.ask(query)

                st.subheader("Answer")
                st.write(answer)

                if evidence:
                    st.subheader("Retrieved Evidence")
                    for item in evidence:
                        title = (
                            f"{item['modality'].upper()} | {item['file_name']} | page {item['page']} | "
                            f"score {float(item['score']):.3f} | channel {item.get('retrieval_channel', 'n/a')}"
                        )
                        with st.expander(title):
                            if item["modality"] == "table":
                                st.code(item["text"], language="text")
                            else:
                                st.write(item["text"])
                            if item["modality"] == "image" and "path" in item.get("meta", {}):
                                img_path = item["meta"]["path"]
                                if Path(img_path).exists():
                                    st.image(img_path, use_container_width=True)
                                if item["meta"].get("nearby_text"):
                                    st.caption("Nearby text: " + item["meta"]["nearby_text"])
                else:
                    st.info("No sufficiently relevant evidence was found.")