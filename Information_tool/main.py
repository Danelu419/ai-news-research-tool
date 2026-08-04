import os
import tempfile
import streamlit as st
import time
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool")
st.sidebar.title("Information Sources")

if "url_count" not in st.session_state:
    st.session_state.url_count = 3
if "show_pdf_uploader" not in st.session_state:
    st.session_state.show_pdf_uploader = False

# --- URLs ---
st.sidebar.subheader("News URLs (optional)")
urls = []
for i in range(st.session_state.url_count):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Add URL"):
        st.session_state.url_count += 1
        st.rerun()
with col2:
    if st.button("Add PDF"):
        st.session_state.show_pdf_uploader = True
        st.rerun()

if st.session_state.url_count > 3:
    if st.sidebar.button("Remove URL"):
        st.session_state.url_count -= 1
        st.rerun()

# --- PDFs: show Browse files right after Add PDF is clicked ---
uploaded_pdfs = []
if st.session_state.show_pdf_uploader:
    st.sidebar.markdown("**Upload PDF**")
    uploaded_pdfs = (
        st.sidebar.file_uploader(
            "Browse files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            help="Select one or more PDF files from your computer",
        )
        or []
    )
    if uploaded_pdfs:
        st.sidebar.success(f"{len(uploaded_pdfs)} PDF(s) selected")

process_clicked = st.sidebar.button("Process Sources", type="primary")

main_placeholder = st.empty()
index_path = "faiss_store"

if process_clicked:
    urls = [u.strip() for u in urls if u.strip()]
    if not urls and not uploaded_pdfs:
        st.error("Please enter at least one URL or upload a PDF.")
        st.stop()

    data = []
    main_placeholder.write("Loading data...")

    # Load from URLs
    if urls:
        url_docs = UnstructuredURLLoader(urls=urls).load()
        for doc in url_docs:
            if "source" not in doc.metadata or not doc.metadata["source"]:
                doc.metadata["source"] = doc.metadata.get("url", "unknown")
        data.extend(url_docs)

    # Load from uploaded PDFs
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                tmp_path = tmp.name
            try:
                pdf_docs = PyPDFLoader(tmp_path).load()
                for doc in pdf_docs:
                    # Show PDF file name (+ page) as the source
                    page = doc.metadata.get("page")
                    label = pdf.name
                    if page is not None:
                        label = f"{pdf.name} (page {page + 1})"
                    doc.metadata["source"] = label
                data.extend(pdf_docs)
            finally:
                os.remove(tmp_path)

    if not data:
        st.error("No content could be loaded from the given sources.")
        st.stop()

    loaded_sources = sorted(
        {doc.metadata.get("source", "") for doc in data if doc.metadata.get("source")}
    )
    main_placeholder.write(
        f"Loaded {len(data)} document(s) from {len(loaded_sources)} source(s)."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
    )
    main_placeholder.write("Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.write("Embedding vectorstore building index...")
    time.sleep(2)

    vectorstore.save_local(index_path)
    main_placeholder.write("Index saved. You can ask questions now.")

query = main_placeholder.text_input("Enter your Question")

if query:
    if os.path.exists(index_path):
        embeddings = FastEmbedEmbeddings()
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.getenv("Groq_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
        )

        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query, k=8
        )
        score_threshold = 0.45

        relevant_docs = []
        best_score_by_source = {}
        for doc, score in docs_with_scores:
            if score < score_threshold:
                continue
            relevant_docs.append(doc)
            src = doc.metadata.get("source") or doc.metadata.get("url")
            if not src:
                continue
            if src not in best_score_by_source or score > best_score_by_source[src]:
                best_score_by_source[src] = score

        if not relevant_docs and docs_with_scores:
            best_doc, best_score = docs_with_scores[0]
            relevant_docs = [best_doc]
            src = best_doc.metadata.get("source") or best_doc.metadata.get("url")
            if src:
                best_score_by_source[src] = best_score

        qa_chain = load_qa_chain(llm, chain_type="stuff")
        answer = qa_chain.run(input_documents=relevant_docs, question=query)

        st.header("Answer")
        st.write(answer)

        source_urls = [
            src
            for src, _ in sorted(
                best_score_by_source.items(), key=lambda x: x[1], reverse=True
            )
        ]

        if source_urls:
            st.subheader("Sources")
            for source in source_urls:
                st.write(source)
        else:
            st.info("No relevant source found for this question.")
    else:
        st.error("Please process sources first to build the knowledge base.")
