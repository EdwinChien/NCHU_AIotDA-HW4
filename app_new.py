import streamlit as st
import os
import tempfile
import warnings
from huggingface_hub import InferenceClient

# 核心基礎套件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RAG 終極修復版", layout="wide")
st.title("📘 RAG 文件問答系統 (API 穩定版)")

# 安全讀取 Token
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.sidebar.text_input("HuggingFace Token", type="password")

if not hf_token:
    st.info("👋 請輸入您的 HuggingFace API Token 以開始。")
    st.stop()

# --- 初始化模型與向量庫 ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_inference_client(token):
    # 更換為更穩定的模型節點
    return InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=token)

# --- PDF 處理 ---
def process_pdf(file, _embeddings):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.getvalue())
        loader = PyPDFLoader(tmp.name)
        docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100))
        vector_db = Chroma.from_documents(docs, _embeddings)
        os.remove(tmp.name)
    return vector_db

# --- 介面邏輯 ---
embeddings = get_embedding_model()
client = get_inference_client(hf_token)

uploaded_file = st.file_uploader("上傳 PDF 文件", type="pdf")

if uploaded_file:
    if "db" not in st.session_state:
        with st.spinner("📄 正在建立索引..."):
            st.session_state.db = process_pdf(uploaded_file, embeddings)
            st.success("✅ 索引建立完成！")

    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})
    user_input = st.text_input("💬 請針對文件提問：")

    if user_input:
        with st.spinner("🤖 思考中..."):
            try:
                search_results = retriever.invoke(user_input)
                context_text = "\n\n".join([doc.page_content for doc in search_results])

                # 構建 Prompt
                prompt = f"<|system|>\n請根據以下內容回答問題，並一律使用繁體中文回答。</s>\n<|user|>\n內容：{context_text}\n問題：{user_input}</s>\n<|assistant|>\n"

                # 使用官方 Client
                response = client.text_generation(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.2,
                )
                
                st.markdown("### 🤖 AI 回答")
                st.write(response)
                
                with st.expander("📄 查看參考來源"):
                    for i, doc in enumerate(search_results):
                        st.caption(f"來源 {i+1}: {doc.page_content}")

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
else:
    st.info("👈 請先上傳 PDF 文件開始對話。")
