import os
import warnings
import streamlit as st

# 1. 隱藏所有版本相容性警告 (針對 urllib3, LibreSSL, Pydantic)
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =========================
# Streamlit 基本設定
# =========================
st.set_page_config(page_title="RAG 穩定修復版", layout="centered")
st.title("📘 修正版 RAG 文件問答系統")

#Token 安全讀取 (請確保已設定環境變數，或手動填入新 Token)
# 建議在終端機執行: export HUGGINGFACEHUB_API_TOKEN='您的新Token'
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("❌ 找不到 HUGGINGFACEHUB_API_TOKEN 環境變數，請確認設定。")
    st.stop()

# =========================
# 1. 載入 LLM (解決 410 Gone 與 StopIteration)
# =========================
@st.cache_resource
def load_llm():
    # 使用 gemma-1.1 版本，這在目前免費 API 上連線最穩定
    repo_id = "google/gemma-1.1-2b-it"
    
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
        timeout=300 # 針對 Mac SSL 握手較慢增加超時時間
    )

llm = load_llm()

# =========================
# 2. 建立 Vector DB (向量資料庫)
# =========================
@st.cache_resource
def build_vector_db():
    pdf_path = "data/documents.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"找不到檔案：{pdf_path}，請確認目錄結構。")
        st.stop()
        
    # 載入 PDF
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter)

    # 嵌入模型 (使用 HuggingFace 託管模型)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 建立記憶體向量庫
    return Chroma.from_documents(documents=docs, embedding=embeddings)

vectordb = build_vector_db()

# =========================
# 3. 配置 RAG 檢索鏈 (避開 Pydantic 錯誤)
# =========================


# 定義 Prompt 範本
system_prompt = (
    "你是一個專業的助理。請根據提供的 Context（上下文）回答問題。"
    "如果答案不在 Context 內，請回答『我不知道』，不要編造答案。"
    "\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 建立文件組合鏈與檢索鏈
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectordb.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)

# =========================
# 4. UI 互動介面
# =========================
question = st.text_input("請輸入您的問題：", placeholder="例如：這份文件的主要內容是什麼？")

if question:
    with st.spinner("🔍 正在檢索資料並生成回答..."):
        try:
            # 呼叫檢索鏈 (新版參數為 input)
            response = rag_chain.invoke({"input": question})
            
            st.subheader("🤖 AI 回答")
            st.success(response["answer"])

            with st.expander("📄 查看參考來源片段"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**來源 {i+1} (頁碼: {doc.metadata.get('page', 'N/A')})**")
                    st.info(doc.page_content)
        except Exception as e:
            st.error(f"發生執行錯誤：{str(e)}")
            st.info("提示：如果遇到 StopIteration，請嘗試執行 'pip install -U huggingface-hub'")