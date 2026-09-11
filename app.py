import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PDF_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

INDEX_DIR = os.path.join(
    BASE_DIR,
    "faiss_index"
)


# ==========================================
# LOAD ALL PDF DOCUMENTS
# ==========================================

def load_all_documents(pdf_folder):

    all_docs = []

    if not os.path.exists(pdf_folder):

        print(
            f"PDF folder not found: {pdf_folder}"
        )

        return all_docs

    pdf_files = [

        file

        for file in os.listdir(pdf_folder)

        if file.lower().endswith(".pdf")

    ]

    if not pdf_files:

        print(
            "No PDF files found in uploads folder."
        )

        return all_docs


    for filename in pdf_files:

        pdf_path = os.path.join(
            pdf_folder,
            filename
        )

        print(
            f"Loading: {filename}"
        )

        try:

            loader = PDFPlumberLoader(
                pdf_path
            )

            documents = loader.load()

            all_docs.extend(
                documents
            )

            print(
                f"✓ Loaded {len(documents)} pages"
            )

        except Exception as e:

            print(
                f"✗ Error loading {filename}: {e}"
            )


    return all_docs


# ==========================================
# CREATE OR LOAD FAISS VECTOR STORE
# ==========================================

def get_vector_store(
    documents,
    embedder
):

    if os.path.exists(INDEX_DIR):

        print(
            "\nLoading existing FAISS index..."
        )

        try:

            vector_store = FAISS.load_local(

                INDEX_DIR,

                embedder,

                allow_dangerous_deserialization=True

            )

            print(
                "✓ FAISS index loaded successfully."
            )

            return vector_store

        except Exception as e:

            print(
                f"FAISS index could not be loaded: {e}"
            )

            print(
                "Creating a new FAISS index..."
            )


    print(
        "\nCreating new FAISS index..."
    )


    # Split documents into chunks

    text_splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=150

        )
    )


    chunks = text_splitter.split_documents(
        documents
    )


    print(
        f"Created {len(chunks)} document chunks."
    )


    # Create vector database

    vector_store = FAISS.from_documents(

        chunks,

        embedder

    )


    # Save vector database

    vector_store.save_local(
        INDEX_DIR
    )


    print(
        "✓ FAISS index created successfully."
    )


    return vector_store


# ==========================================
# BUILD LLM CHAIN
# ==========================================

def build_llm_chain(llm):

    prompt = """

You are an AI Sustainability Chatbot.

Answer the user's question using ONLY the information
provided in the context.

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not make up information.
4. If the answer is not available in the context, say exactly:

"I don't know based on the available documents."

5. Give a clear and accurate answer.
6. Keep the answer concise.
7. Use approximately 3-5 sentences maximum.

Context:

{context}

Question:

{question}

Answer:

"""


    prompt_template = (
        PromptTemplate.from_template(
            prompt
        )
    )


    llm_chain = LLMChain(

        llm=llm,

        prompt=prompt_template,

        verbose=False

    )


    return llm_chain


# ==========================================
# BUILD DOCUMENT CHAIN
# ==========================================

def build_document_chain(
    llm_chain
):

    document_prompt = PromptTemplate(

        input_variables=[

            "page_content",

            "source"

        ],

        template="""

Document:

{page_content}

Source:

{source}

"""

    )


    document_chain = (
        StuffDocumentsChain(

            llm_chain=llm_chain,

            document_variable_name="context",

            document_prompt=document_prompt

        )
    )


    return document_chain


# ==========================================
# BUILD RETRIEVAL QA CHAIN
# ==========================================

def build_qa_chain(
    retriever,
    document_chain
):

    qa_chain = RetrievalQA(

        combine_documents_chain=
        document_chain,

        retriever=retriever,

        return_source_documents=True,

        verbose=False

    )


    return qa_chain


# ==========================================
# INITIALIZE CHATBOT
# ==========================================

def initialize_chatbot():

    print()

    print("=" * 55)

    print(
        "Initializing Sustainability Chatbot..."
    )

    print("=" * 55)

    print()


    # --------------------------------------
    # LOAD PDF DOCUMENTS
    # --------------------------------------

    print(
        "Loading PDF documents..."
    )


    documents = load_all_documents(
        PDF_FOLDER
    )


    if not documents:

        print()

        print(
            "ERROR: No PDF documents could be loaded."
        )

        print(
            f"Check folder: {PDF_FOLDER}"
        )

        return None


    print()

    print(
        f"Total pages loaded: {len(documents)}"
    )


    # --------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------

    print()

    print(
        "Loading embedding model..."
    )


    try:

        embeddings = HuggingFaceEmbeddings(

            model_name=
            "sentence-transformers/all-MiniLM-L6-v2"

        )

        print(
            "✓ Embedding model loaded."
        )


    except Exception as e:

        print()

        print(
            f"ERROR loading embedding model: {e}"
        )

        return None


    # --------------------------------------
    # LOAD OR CREATE VECTOR STORE
    # --------------------------------------

    try:

        vector_store = get_vector_store(

            documents,

            embeddings

        )

    except Exception as e:

        print()

        print(
            f"ERROR creating FAISS index: {e}"
        )

        return None


    # --------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------

    retriever = vector_store.as_retriever(

        search_type="similarity",

        search_kwargs={

            "k": 3

        }

    )


    # --------------------------------------
    # GET GROQ API KEY
    # --------------------------------------

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )


    if not groq_api_key:

        print()

        print(
            "ERROR: GROQ_API_KEY not found."
        )

        print(
            "Add GROQ_API_KEY to your .env file."
        )

        return None


    # --------------------------------------
    # INITIALIZE GROQ MODEL
    # --------------------------------------

    try:

        llm = ChatGroq(

            groq_api_key=groq_api_key,

            model_name="openai/gpt-oss-20b",

            temperature=0,

            max_retries=2

        )

        print(
            "✓ Groq model initialized."
        )


    except Exception as e:

        print()

        print(
            f"ERROR initializing Groq: {e}"
        )

        return None


    # --------------------------------------
    # BUILD CHAINS
    # --------------------------------------

    llm_chain = build_llm_chain(
        llm
    )


    document_chain = build_document_chain(
        llm_chain
    )


    qa_chain = build_qa_chain(

        retriever,

        document_chain

    )


    return qa_chain


# ==========================================
# GET CHATBOT RESPONSE
# ==========================================

def get_response(
    question
):

    global qa_chain


    if qa_chain is None:

        return (

            "The chatbot could not initialize. "
            "Please check the terminal for errors.",

            None

        )


    try:

        response = qa_chain.invoke({

            "query": question

        })


        answer = response.get(

            "result",

            "I don't know based on the available documents."

        )


        source_documents = response.get(

            "source_documents",

            []

        )


        pdf_url = None


        # ----------------------------------
        # GET SOURCE PDF
        # ----------------------------------

        if source_documents:

            source_document = (
                source_documents[0]
            )


            metadata = (
                source_document.metadata
            )


            source_path = metadata.get(

                "source",

                ""

            )


            page_number = metadata.get(

                "page",

                0

            )


            # Get filename only

            filename = os.path.basename(
                source_path
            )


            # Create Flask static URL

            pdf_url = (

                f"/static/uploads/{filename}"

                f"#page={page_number + 1}"

            )


        return (

            answer,

            pdf_url

        )


    except Exception as e:

        print()

        print(
            "ERROR generating response:"
        )

        print(e)


        return (

            "Sorry, an error occurred while "
            "processing your question.",

            None

        )


# ==========================================
# INITIALIZE CHATBOT
# ==========================================

qa_chain = initialize_chatbot()


print()

print("=" * 55)

if qa_chain:

    print(
        "✓ Sustainability Chatbot is ready!"
    )

else:

    print(
        "✗ Chatbot initialization failed."
    )

print("=" * 55)

print()


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# CHAT PAGE
# ==========================================

@app.route("/chat")
def chat():

    return render_template(
        "chat.html"
    )


# ==========================================
# ASK QUESTION API
# ==========================================

@app.route(
    "/ask",
    methods=["POST"]
)

def ask():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "answer":
                "Invalid request.",

                "source":
                None

            }), 400


        question = data.get(

            "question",

            ""

        ).strip()


        if not question:

            return jsonify({

                "answer":
                "Please enter a question.",

                "source":
                None

            }), 400


        answer, source = get_response(
            question
        )


        return jsonify({

            "answer": answer,

            "source": source

        })


    except Exception as e:

        print()

        print(
            "API ERROR:"
        )

        print(e)


        return jsonify({

            "answer":
            "Sorry, something went wrong.",

            "source":
            None

        }), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True,

        use_reloader=False

    )