import os
import streamlit as st
import time
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool")
st.sidebar.title("Information News Urls")

if "url_count" not in st.session_state:
    st.session_state.url_count = 3

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
    if st.session_state.url_count > 3 and st.button("Remove"):
        st.session_state.url_count -= 1
        st.rerun()

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
index_path = "faiss_store"

if process_url_clicked:
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        st.error("Please enter at least one valid URL (http/https).")
        st.stop()

    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.write("Loading data...")
    data = loader.load()
    if not data:
        st.error("No content could be loaded from the given URLs.")
        st.stop()

    # Keep source URL on every chunk so all URLs can be shown later
    for doc in data:
        if "source" not in doc.metadata or not doc.metadata["source"]:
            doc.metadata["source"] = doc.metadata.get("url", "unknown")

    loaded_sources = sorted(
        {doc.metadata.get("source", "") for doc in data if doc.metadata.get("source")}
    )
    main_placeholder.write(
        f"Loaded {len(data)} page(s) from {len(loaded_sources)} URL(s): "
        + ", ".join(loaded_sources)
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
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
        )

        # Score chunks against the question; keep only strong matches
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query, k=8
        )
        score_threshold = 0.45  # higher = stricter (only more relevant sources)

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

        # If nothing passed the threshold, fall back to the single best chunk
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

        # Sources sorted by relevance (best match first)
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
        st.error("Please process URLs first to build the knowledge base.")
