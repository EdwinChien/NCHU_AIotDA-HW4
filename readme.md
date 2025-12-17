這是一份為你的 **RAG 知識庫問答系統** 設計的 `README.md`。它參考了影片中的實作流程，並結合了 Streamlit 的部署說明。

---

```markdown
# 🤖 TAICA 生成式 AI：RAG 知識庫問答系統

這是一個基於 **RAG (Retrieval-Augmented Generation)** 技術開發的 AI 應用程式。本專案參考 1141 【TAICA。生成式 AI】課程實作，整合了 LangChain、Hugging Face Embedding 以及 Groq 高速推理引擎。

## 🌟 功能特色
* **減少幻覺**：AI 僅根據你提供的 PDF、Docx 或 TXT 文件內容進行回答。
* **高效檢索**：使用 FAISS 向量資料庫進行語義比對，精準定位相關片段。
* **高速生成**：串接 Groq API，利用 Llama 3 模型提供近乎即時的對話體驗。
* **互動界面**：使用 Streamlit 打造簡潔直觀的網頁操作介面。

---

## 🛠️ 快速開始

### 1. 環境準備
確保你的電腦已安裝 Python 3.9+，並複製此專案到本地：

```bash
git clone <your-repo-url>
cd <your-repo-name>

```

### 2. 安裝套件

使用 pip 安裝 `requirements.txt` 中列出的必要套件：

```bash
pip install -r requirements.txt

```

### 3. 準備向量資料庫 (Vector DB)

根據課程影片實作，你需要先完成以下步驟：

1. 執行課程提供的 **「程式 A」** (Colab)，上傳你的文件並生成向量庫。
2. 下載產生的 `index.zip` 並解壓縮。
3. 將解壓後的資料夾命名為 `faiss_index`，並放在本專案的根目錄下。

### 4. 取得 API Key

本系統需要以下金鑰：

* **Groq API Key**: 前往 [Groq Cloud](https://console.groq.com/) 申請（用於 LLM 生成）。
* **Hugging Face Token**: 用於下載 Embedding 模型（如影片中的 `Gemma` 相關模型）。

### 5. 執行應用程式

在終端機輸入以下指令啟動 Streamlit：

```bash
streamlit run app.py

```

---

## 📂 專案架構

* `app.py`: Streamlit 網頁主程式。
* `faiss_index/`: 儲存向量化的文件特徵（需自行放入）。
* `requirements.txt`: 專案依賴清單。
* `README.md`: 專案說明文件。

---

## 🧠 RAG 運作流程

1. **文件切塊 (Chunking)**：將大型文件拆解為易於檢索的小片段。
2. **嵌入 (Embedding)**：將文字轉為數學向量。
3. **檢索 (Retrieval)**：當使用者提問時，找出最相似的 N 個文件塊。
4. **增強生成 (Generation)**：將「問題 + 文件塊」組成 Prompt 丟給 LLM 產生答案。

---

## ⚠️ 注意事項

* 請確保 `faiss_index` 資料夾內的模型與 `app.py` 中指定的 `HuggingFaceEmbeddings` 模型一致，否則向量比對會失敗。
* 建議使用 `allow_dangerous_deserialization=True` 參數載入本地 FAISS 時，僅載入來源可信的資料庫檔案。

```

---
