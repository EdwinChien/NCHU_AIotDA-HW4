import streamlit as st
import os
import tempfile
import warnings

# 1. 導入核心基礎套件 (避開容易消失的 .chains)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 忽略無關警告
warnings.filterwarnings("ignore")

# --- 頁面設定 ---
st.set_page_config(page_title="RAG 穩定版", layout="wide")
st.title("📘 RAG 文件問答系統 (2025 穩定修復版)")

# --- 安全讀取 Token ---
# 優先從 Streamlit Secrets 讀取，否則從側邊欄輸入
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.sidebar.text_input("請輸入 HuggingFace Token", type="password")

if not hf_token:
    st.info("👋 請在側邊欄輸入您的 HuggingFace API Token 以開始。")
    st.stop()

# --- 資源載入 (快取) ---
@st.cache_resource
def load_llm_and_embeddings(token):
    # 使用 all-MiniLM-L6-v2 作為 Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 改用 HuggingFaceHub 避開 StopIteration Bug
    llm = HuggingFaceHub(
        repo_id="google/gemma-1.1-2b-it",
        huggingfacehub_api_token=token,
        model_kwargs={"max_new_tokens": 512, "temperature": 0.1}
    )
    return embeddings, llm

# --- PDF 處理功能 ---
def process_uploaded_file(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter)
    
    # 建立向量庫
    embeddings, _ = load_llm_and_embeddings(hf_token)
    vector_db = Chroma.from_documents(docs, embeddings)
    
    os.remove(tmp_path) # 清理暫存檔
    return vector_db

# --- 主程式邏輯 ---
embeddings_model, llm_model = load_llm_and_embeddings(hf_token)

uploaded_file = st.file_uploader("選擇 PDF 文件", type="pdf")

if uploaded_file:
    if "db" not in st.session_state:
        with st.spinner("正在建立文件索引..."):
            st.session_state.db = process_uploaded_file(uploaded_file)
            st.success("✅ 文件索引建立成功！")

    # 設定檢索器
    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})

    # 設定 Prompt (LCEL 語法)
    prompt = ChatPromptTemplate.from_template("""
    你是一位專業的助手。請根據以下提供的 Context (內容) 回答問題。
    如果 Context 中沒有相關資訊，請誠實回答「我不知道」，不要編造。
    請一律使用「繁體中文」回答。

    Context:
    {context}

    問題:
    {question}

    回答:""")

    # 建立 RAG 鏈 (完全避開 langchain.chains)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )

    # UI 問答區
    st.divider()
    user_input = st.text_input("💬 請輸入您的問題：")

    if user_input:
        with st.spinner("正在檢索並生成答案..."):
            try:
                response = rag_chain.invoke(user_input)
                st.markdown("### 🤖 AI 回答")
                st.write(response)
                
                with st.expander("查看參考來源片段"):
                    source_docs = retriever.get_relevant_documents(user_input)
                    for i, doc in enumerate(source_docs):
                        st.info(f"來源 {i+1}:\n{doc.page_content}")
            except Exception as e:
                st.error(f"執行時出錯：{str(e)}")
else:
    st.warning("👈 請先上傳 PDF 文件。")
