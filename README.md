🧪 Quest Analytics RAG Assistant
Retrieval-Augmented Generation (RAG) System Powered by IBM Watsonx + LangChain + Streamlit

Built by Karen Delea

<div align="center">
🚀 Ask scientific papers anything.

<img width="926" height="389" alt="image" src="https://github.com/user-attachments/assets/e35f5134-15cd-40d6-a5c1-1046767c8d57" />


Upload a PDF → System reads, embeds, stores, retrieves → Granite 3.3 8B answers grounded in the document.

</div>
🌟 Live Demo

👉 App URL:
🔗 https://genaiappkd.streamlit.app

👉 Demo Video:
🎥 [Add your MP4 demo link here](https://www.youtube.com/watch?v=BIZeFo50uPU)



🧠 Overview

**Quest Analytics RAG Assistant is a Retrieval-Augmented Generation system designed to help researchers, analysts, and students extract insights from scientific papers.**

This application implements a full RAG workflow using:

IBM watsonx Granite 3.3 8B Instruct (LLM)

IBM Slate-125M Retriever V2 (Embeddings)

LangChain Core + Community (LCEL) for the pipeline

ChromaDB for vector indexing

Users can upload a PDF, ask a question, and receive an accurate, document-grounded answer.


✨ **Features**

✔ Upload any scientific paper (PDF)
✔ Automatic text extraction + chunking
✔ Watsonx Embeddings with Slate-125M RTRVR V2
✔ Chroma vector database
✔ Watsonx Granite 3.3B Instruct LLM for grounded answers
✔ RAG pipeline implemented using LCEL (no deprecated LangChain modules)
✔ Production-ready Streamlit UI
✔ Secrets managed securely with .env or Streamlit Secrets Manager

🛠️ **Tech Stack**
Component	Technology
LLM	ibm/granite-3-3-8b-instruct
Embeddings	ibm/slate-125m-english-rtrvr-v2
Vector Database	ChromaDB
Framework	LangChain Core + Community (LCEL)
App UI	Streamlit
Environment	Python 3.10+
Deployment	Streamlit Cloud

<img width="358" height="565" alt="image" src="https://github.com/user-attachments/assets/8d813b50-b90a-4b2b-965e-00746e999810" />

🔑 Environment Variables

Create a .env file:

WATSONX_APIKEY=your_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id_here


**Or add them in:**

Streamlit Cloud → Settings → Secrets

⚙️ Local Installation
1. Clone the repository
git clone https://github.com/kbello90/Gen_AI_App_KD.git
cd Gen_AI_App_KD

2. Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate      # Mac/Linux

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py

🌐 **Deployment (Streamlit Cloud)**

Push project to GitHub

Go to https://streamlit.io/cloud

Create a new app → Select this repository

**Set:**

Main file: app.py

Python version: 3.10

Add secrets in:

Settings → Secrets

Done — the app deploys automatically.


💼 Author

Karen Delea
AI Engineer • Data Scientist • Supply Chain Analytics Specialist
🔗 Portfolio: [Add link here](https://sites.google.com/view/karendelea-portfolio/home)
🔗 LinkedIn: [Add link here](https://www.linkedin.com/in/karen-bello-delea/)
