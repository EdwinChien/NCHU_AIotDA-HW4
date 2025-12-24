import streamlit as st
import os
import tempfile
import warnings
from langchain_community.document_loaders import PyPDFLoader
# ✅ 新版路徑 (對應 langchain-text-splitters 套件)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# =========================
# 頁面配置
# =========================
st.set_page_config(page_title="RAG PDF 助手", layout="wide")
st.title("🤖 RAG 智慧文件對話系統")

# =========================
# Token 管理 (優先讀取 Secrets)
# =========================
# 在 Streamlit Cloud 的設定頁面中添加 Secrets: HUGGINGFACEHUB_API_TOKEN = "your_token"
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.sidebar.text_input("輸入 HF Token", type="password")

if not hf_token:
    st.warning("⚠️ 請在側邊欄輸入 HuggingFace Token 或在 Secrets 中設定。")
    st.stop()

# =========================
# 核心功能組件 (快取處理)
# =========================
@st.cache_resource
def load_llm(token):
    return HuggingFaceEndpoint(
        repo_id="google/gemma-1.1-2b-it",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=token,
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100))
    vectorstore = Chroma.from_documents(documents=docs, embedding=get_embeddings())
    os.remove(tmp_path) # 清理臨時文件
    return vectorstore

# =========================
# 側邊欄與文件處理
# =========================
with st.sidebar:
    st.header("📄 文件上傳")
    uploaded_file = st.file_uploader("上傳 PDF 開始問答", type="pdf")
    if st.button("清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

if uploaded_file:
    if "vectordb" not in st.session_state:
        with st.spinner("正在建立知識庫..."):
            st.session_state.vectordb = process_pdf(uploaded_file)
            st.success("文件處理完成！")

    # 設定 RAG 鏈
    llm = load_llm(hf_token)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個助理，請根據 context 回答問題。若不知道請說『我不知道』。請用繁體中文回答。\n\nContext: {context}"),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.vectordb.as_retriever(), combine_docs_chain)

    # =========================
    # 對話界面 (Chat UI)
    # =========================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示歷史訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 使用者輸入
    if prompt_input := st.chat_input("對 PDF 內容提問..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = rag_chain.invoke({"input": prompt_input})
                full_response = response["answer"]
                st.markdown(full_response)
                
                # 顯示來源
                with st.expander("查看參考來源"):
                    for doc in response["context"]:
                        st.caption(f"内容：{doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("請先上傳 PDF 文件以啟用問答功能。")
