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
from langchain.chains import RetrievalQA   # ✅ corrected import

warnings.filterwarnings("ignore")
load_dotenv()

# ---------------------------------------------------------
# Watsonx Config
# ---------------------------------------------------------
def get_watsonx_config():
    api_key = os.environ.get("WATSONX_APIKEY") or st.secrets.get("WATSONX_APIKEY", None)
    url = os.environ.get("WATSONX_URL") or st.secrets.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.environ.get("WATSONX_PROJECT_ID") or st.secrets.get("WATSONX_PROJECT_ID", None)

    if api_key is None or project_id is None:
        st.error("Watsonx credentials not configured. Please set WATSONX_APIKEY and WATSONX_PROJECT_ID.")
        st.stop()

    return api_key, url, project_id

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
        length_function=len,
    )
    return splitter.split_documents(data)

# ---------------------------------------------------------
# Task 3: Embedding Model
# ---------------------------------------------------------
def watsonx_embedding(api_key: str, url: str, project_id: str):
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url=url,
        project_id=project_id,
        params=embed_params,
        api_key=api_key   # ✅ added
    )

# ---------------------------------------------------------
# Task 4: Vector Database
# ---------------------------------------------------------
def vector_database(chunks, api_key: str, url: str, project_id: str):
    embedding_model = watsonx_embedding(api_key, url, project_id)
    return Chroma.from_documents(chunks, embedding_model)

# ---------------------------------------------------------
# Task 5: Retriever
# ---------------------------------------------------------
def build_retriever(file_path: str, api_key: str, url: str, project_id: str):
    splits = document_loader(file_path)
    chunks = text_splitter(splits)
    st.sidebar.write(f"📄 Chunks created: {len(chunks)}")
    vectordb = vector_database(chunks, api_key, url, project_id)
    return vectordb.as_retriever()

# ---------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------
def get_llm(api_key: str, url: str, project_id: str):
    model_id = "ibm/granite-3-3-8b-instruct"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters,
        api_key=api_key   # ✅ added
    )

# ---------------------------------------------------------
# Task 6: QA Bot Logic
# ---------------------------------------------------------
def retriever_qa(file_path: str, query: str, api_key: str, url: str, project_id: str) -> str:
    try:
        llm = get_llm(api_key, url, project_id)
        retriever_obj = build_retriever(file_path, api_key, url, project_id)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False
        )
        response = qa.invoke(query)
        if isinstance(response, dict) and "result" in response:
            return response["result"]
        return str(response)
    except Exception as e:
        return f"⚠️ Failed to generate response: {e}"

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Quest Analytics RAG Assistant", layout="wide")
    st.title("🧪 Quest Analytics RAG Assistant by Karen Delea")
    st.write("Upload a scientific paper and ask questions. Powered by IBM watsonx LLMs, LangChain, and Chroma.")

    api_key, url, project_id = get_watsonx_config()

    st.sidebar.header("🔧 Model Configuration")
    st.sidebar.write("**LLM:** ibm/granite-3-3-8b-instruct")
    st.sidebar.write("**Embeddings:** ibm/slate-125m-english-rtrvr-v2")

    uploaded_file = st.file_uploader("Upload a scientific paper (PDF)", type=["pdf"])
    query = st.text_area("Ask a question about the document", placeholder="e.g., What is the main contribution of this paper?")

    with st.expander("💡 Example Queries"):
        st.markdown("- What is the main contribution of this paper?")
        st.markdown("- How does LoRA compare to full fine-tuning?")
        st.markdown("- What datasets were used in the experiments?")

    if st.button("Ask"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
            st.stop()
        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("Reading document, building vector store, and querying the model..."):
            answer = retriever_qa(tmp_path, query, api_key, url, project_id)

        st.subheader("🧠 Answer")
        st.write(answer)

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
