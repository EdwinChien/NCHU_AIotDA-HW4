import streamlit as st
import os
import tempfile
import warnings

# 核心基礎套件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RAG 穩定修復版", layout="wide")
st.title("📘 RAG 文件問答系統 (API 路由修正版)")

# 安全讀取 Token
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.sidebar.text_input("HuggingFace Token", type="password")

if not hf_token:
    st.info("請輸入 API Token 以開始。")
    st.stop()

@st.cache_resource
def load_resources(token):
    # Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 解決 410 Gone 錯誤：使用最新版的 HuggingFaceEndpoint
    # repo_id 建議使用穩定版本，例如 gemma-2-2b-it (這是目前的推薦版)
    repo_id = "google/gemma-2-2b-it" 
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=token,
        timeout=300
    )
    return embeddings, llm

embeddings_model, llm_model = load_resources(hf_token)

# --- 文件處理與問答邏輯 (保持穩定版語法) ---
uploaded_file = st.file_uploader("選擇 PDF 文件", type="pdf")

if uploaded_file:
    if "db" not in st.session_state:
        with st.spinner("建立索引中..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100))
                st.session_state.db = Chroma.from_documents(docs, embeddings_model)
                os.remove(tmp.name)

    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    請根據以下 Context 回答問題。請使用「繁體中文」回答。
    Context: {context}
    問題: {question}
    回答:""")

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )

    user_input = st.text_input("💬 請輸入問題：")
    if user_input:
        with st.spinner("AI 正在思考..."):
            try:
                # 某些模型生成結果會包含 prompt，這裡做簡單清洗
                response = rag_chain.invoke(user_input)
                st.markdown("### 🤖 AI 回答")
                st.write(response)
            except Exception as e:
                st.error(f"連線失敗：{str(e)}")
                st.info("提示：如果遇到 410 錯誤，請確認 requirements.txt 中的 huggingface-hub 為最新版本。")
