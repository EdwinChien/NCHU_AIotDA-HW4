import streamlit as st
import os
import tempfile
import warnings
from huggingface_hub import InferenceClient

# 核心基礎套件 (使用 2025 最新路徑)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    # 使用 HuggingFace 託管的 Embedding 模型
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_inference_client(token):
    # 使用 InferenceClient 避開所有 LangChain 封裝 Bug
    return InferenceClient(model="google/gemma-2-2b-it", token=token)

# --- PDF 處理 ---
def process_pdf(file, _embeddings):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.getvalue())
        loader = PyPDFLoader(tmp.name)
        # 增加切分容錯
        docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100))
        # 使用記憶體形式的 Chroma
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

    # 建立檢索器
    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})

    user_input = st.text_input("💬 請針對文件提問：")

    if user_input:
        with st.spinner("🤖 思考中..."):
            try:
                # 【關鍵修正】: 舊版 get_relevant_documents 已棄用，改用 invoke
                search_results = retriever.invoke(user_input)
                
                context_text = "\n\n".join([doc.page_content for doc in search_results])

                # 構建 Prompt
                prompt = f"請根據以下內容回答問題：\n\n內容：{context_text}\n\n問題：{user_input}\n\n回答（請用繁體中文）："

                # 使用官方 Client 進行推論
                response = client.text_generation(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.1,
                    stop_sequences=["問題："]
                )
                
                st.markdown("### 🤖 AI 回答")
                st.write(response)
                
                with st.expander("📄 查看參考來源"):
                    for i, doc in enumerate(search_results):
                        st.caption(f"來源 {i+1}: {doc.page_content}")

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
                st.info("提示：如果遇到版本問題，請嘗試點擊側邊欄的清除快取。")
else:
    st.info("👈 請先上傳 PDF 文件開始對話。")
