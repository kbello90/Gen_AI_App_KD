# ---------------------------------------------------------
# Quest Analytics RAG Assistant - Streamlit App
# ---------------------------------------------------------

import os
import tempfile
import warnings

import streamlit as st
from dotenv import load_dotenv

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings

# LangChain RAG (New API)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

warnings.filterwarnings("ignore")
load_dotenv()  # Load .env for Streamlit Cloud


def get_watsonx_config():
    """
    Read watsonx config from environment variables or Streamlit secrets.
    """
    api_key = (
        os.environ.get("WATSONX_APIKEY")
        or st.secrets.get("WATSONX_APIKEY", None)
    )
    url = (
        os.environ.get("WATSONX_URL")
        or st.secrets.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    )
    project_id = (
        os.environ.get("WATSONX_PROJECT_ID")
        or st.secrets.get("WATSONX_PROJECT_ID", None)
    )

    if api_key is None or project_id is None:
        st.error(
            "❌ Watsonx credentials missing.\n\n"
            "Please set WATSONX_APIKEY and WATSONX_PROJECT_ID "
            "in Streamlit Secrets or your .env file."
        )
        st.stop()

    return url, project_id


# ---------------------------------------------------------
# Task 1: Load Document
# ---------------------------------------------------------

def document_loader(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


# ---------------------------------------------------------
# Task 2: Text Splitter
# ---------------------------------------------------------

def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_documents(data)


# ---------------------------------------------------------
# Task 3: Embeddings
# ---------------------------------------------------------

def watsonx_embedding(url: str, project_id: str):
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url=url,
        project_id=project_id,
        params=embed_params,
    )


# ---------------------------------------------------------
# Task 4: Vector Database
# ---------------------------------------------------------

def vector_database(chunks, url: str, project_id: str):
    embedding_model = watsonx_embedding(url, project_id)
    return Chroma.from_documents(chunks, embedding_model)


# ---------------------------------------------------------
# Task 5: Build Retriever
# ---------------------------------------------------------

def build_retriever(file_path: str, url: str, project_id: str):
    docs = document_loader(file_path)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks, url, project_id)
    return vectordb.as_retriever()


# ---------------------------------------------------------
# LLM Setup (Granite 3.3 8B)
# ---------------------------------------------------------

def get_llm(url: str, project_id: str):
    params = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(
        model_id="ibm/granite-3-3-8b-instruct",
        url=url,
        project_id=project_id,
        params=params,
    )


# ---------------------------------------------------------
# Task 6: New RAG Pipeline (NO MORE RetrievalQA)
# ---------------------------------------------------------

def retriever_qa(file_path: str, query: str, url: str, project_id: str):
    llm = get_llm(url, project_id)
    retriever = build_retriever(file_path, url, project_id)

    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    result = rag_chain.invoke({"input": query})
    return result["answer"]


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Quest Analytics RAG Assistant",
        layout="wide",
    )

    st.title("🧪 Quest Analytics RAG Assistant - by Karen Delea")
    st.write(
        "Upload a scientific paper (PDF), ask questions, and the system will "
        "answer using **RAG with IBM watsonx.ai**."
    )

    # Load Watsonx credentials
    url, project_id = get_watsonx_config()

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    query = st.text_input(
        "Ask a question about the PDF",
        placeholder="Example: What is the main contribution of this study?"
    )

    if st.button("Ask"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
            st.stop()

        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        # Save PDF for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Processing document, creating embeddings, and querying the LLM..."):
            try:
                answer = retriever_qa(pdf_path, query, url, project_id)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                return

        st.subheader("🧠 Answer")
        st.write(answer)


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

