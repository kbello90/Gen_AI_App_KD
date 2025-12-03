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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval import RetrievalQA  


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

warnings.filterwarnings("ignore")

# Load .env so WATSONX_* vars are available in os.environ
load_dotenv()


def get_watsonx_config():
    """
    Read watsonx config from environment variables or Streamlit secrets.
    You must define:
      WATSONX_APIKEY
      WATSONX_URL
      WATSONX_PROJECT_ID
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
            "Watsonx credentials not configured.\n\n"
            "Please set WATSONX_APIKEY and WATSONX_PROJECT_ID "
            "in your .env file or Streamlit secrets."
        )
        st.stop()

    return url, project_id


# ---------------------------------------------------------
# Task 1: Load Document
# ---------------------------------------------------------

def document_loader(file_path: str):
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document


# ---------------------------------------------------------
# Task 2: Text Splitter
# ---------------------------------------------------------

def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks


# ---------------------------------------------------------
# Task 3: Embedding Model
# ---------------------------------------------------------

def watsonx_embedding(url: str, project_id: str):
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url=url,
        project_id=project_id,
        params=embed_params,
    )
    return embedding_model


# ---------------------------------------------------------
# Task 4: Vector Database
# ---------------------------------------------------------

def vector_database(chunks, url: str, project_id: str):
    embedding_model = watsonx_embedding(url, project_id)
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


# ---------------------------------------------------------
# Task 5: Retriever
# ---------------------------------------------------------

def build_retriever(file_path: str, url: str, project_id: str):
    splits = document_loader(file_path)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks, url, project_id)
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


# ---------------------------------------------------------
# LLM Setup (Granite 3.3 8B — same as your lab app)
# ---------------------------------------------------------

def get_llm(url: str, project_id: str):
    model_id = "ibm/granite-3-3-8b-instruct"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm


# ---------------------------------------------------------
# Task 6: QA Bot Logic
# ---------------------------------------------------------

def retriever_qa(file_path: str, query: str, url: str, project_id: str) -> str:
    llm = get_llm(url, project_id)
    retriever_obj = build_retriever(file_path, url, project_id)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )

    response = qa.invoke(query)

    # In newer LangChain versions, invoke may return a dict with "result"
    if isinstance(response, dict) and "result" in response:
        return response["result"]
    return str(response)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Quest Analytics RAG Assistant",
        layout="wide",
    )

    st.title("🧪 Quest Analytics RAG Assistant by Karen Delea")
    st.write(
        "Upload a scientific paper in PDF format and ask questions about it.\n\n"
        "This assistant uses **IBM watsonx LLMs**, **LangChain**, and **Chroma** "
        "for Retrieval-Augmented Generation (RAG)."
    )

    url, project_id = get_watsonx_config()

    st.sidebar.header("🔧 Model Configuration")
    st.sidebar.write("**LLM:** ibm/granite-3-3-8b-instruct")
    st.sidebar.write("**Embeddings:** ibm/slate-125m-english-rtrvr-v2")

    uploaded_file = st.file_uploader(
        "Upload a scientific paper (PDF)",
        type=["pdf"],
    )

    query = st.text_area(
        "Ask a question about the document",
        placeholder="e.g., What is the main contribution of this paper?",
    )

    if st.button("Ask"):

        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
            st.stop()

        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        # Save uploaded PDF to a temporary file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner(
            "Reading document, building vector store, and querying the model..."
        ):
            try:
                answer = retriever_qa(tmp_path, query, url, project_id)
            except Exception as e:
                st.error(f"Error while processing the document: {e}")
                return

        st.subheader("🧠 Answer")
        st.write(answer)


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
