## ⚙️ How It Works

1. PDF documents are stored in the `static/uploads` folder.
2. The application loads the PDF documents using PDFPlumber.
3. Documents are divided into smaller text chunks.
4. HuggingFace embeddings convert the text into vector representations.
5. FAISS stores and searches the document vectors.
6. When a user asks a question, relevant document chunks are retrieved.
7. The retrieved context is sent to the Groq LLM.
8. The chatbot generates an answer based only on the retrieved documents.
9. The source PDF and page number are displayed.

---

## 🧠 RAG Architecture

```text
User Question
      │
      ▼
HuggingFace Embeddings
      │
      ▼
FAISS Vector Search
      │
      ▼
Relevant PDF Chunks
      │
      ▼
Groq LLM
      │
      ▼
Answer + Source Page
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mohit01112/Sustainability-Chatbot.git
```

### 2. Navigate to the Project Folder

```bash
cd Sustainability-Chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root directory.

```text
GROQ_API_KEY=your_groq_api_key_here
```

⚠️ Never upload your `.env` file to GitHub.

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 💬 Example Questions

### Packaging

- What is the maximum concentration of heavy metals allowed in packaging?
- What requirements must reusable packaging meet?
- What measures are taken to prevent packaging waste?
- What is the difference between reusable and single-use packaging?

### Climate Change

- What are the main causes of climate change?
- What are the effects of global warming?
- How can greenhouse gas emissions be reduced?

### Renewable Energy

- What role does renewable energy play in reducing greenhouse gas emissions?
- How does renewable energy contribute to sustainability?

---

## 📚 Document Sources

The chatbot uses PDF documents related to:

- Climate Change
- Renewable Energy
- Sustainable Development Goals
- EU Packaging and Packaging Waste Regulations
- Packaging Waste Prevention
- Reusable Packaging

---

## ⚠️ Limitations

- Answer quality depends on the content available in the uploaded PDFs.
- The chatbot may return incorrect results if irrelevant document chunks are retrieved.
- Large PDF collections may increase the initial indexing time.
- The FAISS index must be recreated when documents are significantly changed.

---

## 🔮 Future Improvements

- Upload PDFs directly through the web interface
- Support chat history
- Improve document retrieval accuracy
- Add multiple source citations
- Add conversation memory
- Deploy the application online
- Add hybrid search
- Improve the user interface

---

## 👨‍💻 Author

**Mohit *

GitHub: https://github.com/Mohit01112

---

⭐ If you found this project useful, consider giving it a star!