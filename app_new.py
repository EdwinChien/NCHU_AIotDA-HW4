import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="PDF RAG (LCEL Version)", layout="wide")
st.title("📘 穩定版 RAG 文件問答 (LCEL)")

# 安全讀取 Token
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.sidebar.text_input("HF Token", type="password")
if not hf_token:
    st.info("請輸入 Token 或在 Secrets 設定中配置。")
    st.stop()

@st.cache_resource
def load_resources():
    # 載入 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 載入 LLM
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-1.1-2b-it",
        task="text-generation",
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token
    )
    return embeddings, llm

embeddings_model, llm_model = load_resources()

# 文件上傳
uploaded_file = st.file_uploader("上傳 PDF", type="pdf")

if uploaded_file:
    if "vector_db" not in st.session_state:
        with st.spinner("建立索引中..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50))
                st.session_state.vector_db = Chroma.from_documents(docs, embeddings_model)
    
    # --- 核心邏輯：使用 LCEL (避免使用 langchain.chains) ---
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
    請根據以下內容回答問題：
    {context}
    
    問題：{question}
    回答（請用繁體中文）：""")

    # 建立 LCEL 鏈 (這是一個像管道一樣的流，完全避開舊的 chains 模組)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )

    # UI 互動
    user_query = st.text_input("輸入您的問題：")
    if user_query:
        with st.spinner("思考中..."):
            result = rag_chain.invoke(user_query)
            st.markdown("### AI 回答")
            st.write(result)
