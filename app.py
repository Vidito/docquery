import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os


import nest_asyncio

nest_asyncio.apply()


# --- Configuration ---
def configure_gemini():
    """Sets up the Google Gemini API key from Streamlit secrets or sidebar input."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = st.sidebar.text_input(
            "Enter Google API Key", type="password", key="api_key_input"
        )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        return True
    else:
        st.sidebar.warning("Please enter your Google Gemini API Key to proceed.")
        return False


# --- Data Processing Functions ---
def get_file_text(uploaded_files):
    """Extracts text from uploaded PDF and TXT files."""
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append(
                        {"text": text, "source": file.name, "page": page_num}
                    )
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            if text:
                documents.append({"text": text, "source": file.name, "page": None})
    return documents


def get_text_chunks(documents):
    """Splits extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for split_text in split_texts:
            chunks.append(
                {"text": split_text, "source": doc["source"], "page": doc["page"]}
            )
    return chunks


def get_vector_store(chunks):
    """Creates a FAISS vector store from text chunks and their metadata."""
    if not chunks:
        return None
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "page": str(chunk["page"])} for chunk in chunks
    ]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    return vector_store


# --- Conversational Chain ---
def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain with memory."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    custom_prompt_template = """
    Create a final answer to the given questions using the provided document excerpts (given in no particular order) as sources. ALWAYS include a "SOURCES" section in your answer citing only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not have enough information to answer the question and leave the SOURCES section empty. Use only the provided documents and do not attempt to fabricate an answer.

---------

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
SOURCES: 1-32
Content: While we're at it, let's make sure every American can get the health care they need. \n\nWe've already made historic investments in health care. \n\nWe've made it easier for Americans to get the care they need, when they need it. \n\nWe've made it easier for Americans to get the treatments they need, when they need them. \n\nWe've made it easier for Americans to get the medications they need, when they need them.
SOURCES: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat's why I'm calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
SOURCES: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
SOURCES: 1-32

---------

QUESTION: {question}
=========
{context}
=========
FINAL ANSWER:

    """

    CUSTOM_PROMPT = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
    )
    return conversation_chain


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with Your Docs", page_icon="📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    # --- Sidebar for Configuration and File Upload ---
    with st.sidebar:
        st.title("Configuration")
        st.markdown("---")

        if not configure_gemini():
            st.stop()

        st.header("Upload Your Files")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", accept_multiple_files=True, type=["pdf", "txt"]
        )

        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
                    documents = get_file_text(uploaded_files)
                    if documents:
                        chunks = get_text_chunks(documents)
                        vector_store = get_vector_store(chunks)
                        if vector_store:
                            st.session_state.conversation = get_conversation_chain(
                                vector_store
                            )
                            st.session_state.processed = True
                            st.session_state.messages = (
                                []
                            )  # Clear chat on new processing
                            st.success("✅ Documents processed successfully!")
                        else:
                            st.error("❌ Failed to create vector store.")
                    else:
                        st.error("❌ No text could be extracted from the files.")
            else:
                st.warning("⚠️ Please upload at least one file.")

        st.markdown("---")
        st.info(
            "Your documents are processed for this session only and are not stored permanently."
        )

    # --- Main Chat Interface ---
    st.title("Chat with Your Documents 💬")
    st.markdown(
        "Upload your documents in the sidebar, and then ask questions about their content."
    )

    if not st.session_state.processed:
        st.info(
            "Get started by uploading your files and clicking 'Process Documents' in the sidebar."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.processed:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.invoke(
                        {"question": prompt}
                    )
                    answer = response["answer"]
                    sources = response["source_documents"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(
                                    f"**Source {i}:** `{source.metadata['source']}` (Page: {source.metadata.get('page', 'N/A')})"
                                )
                                st.info(f"Content: *'{source.page_content[:250]}...'*")
        else:
            st.warning(
                "Please upload and process your documents before asking questions."
            )


if __name__ == "__main__":
    main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os


import nest_asyncio

nest_asyncio.apply()


# --- Configuration ---
def configure_gemini():
    """Sets up the Google Gemini API key from Streamlit secrets or sidebar input."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = st.sidebar.text_input(
            "Enter Google API Key", type="password", key="api_key_input"
        )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        return True
    else:
        st.sidebar.warning("Please enter your Google Gemini API Key to proceed.")
        return False


# --- Data Processing Functions ---
def get_file_text(uploaded_files):
    """Extracts text from uploaded PDF and TXT files."""
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append(
                        {"text": text, "source": file.name, "page": page_num}
                    )
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            if text:
                documents.append({"text": text, "source": file.name, "page": None})
    return documents


def get_text_chunks(documents):
    """Splits extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for split_text in split_texts:
            chunks.append(
                {"text": split_text, "source": doc["source"], "page": doc["page"]}
            )
    return chunks


def get_vector_store(chunks):
    """Creates a FAISS vector store from text chunks and their metadata."""
    if not chunks:
        return None
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "page": str(chunk["page"])} for chunk in chunks
    ]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    return vector_store


# --- Conversational Chain ---
def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain with memory."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    custom_prompt_template = """
    Create a final answer to the given questions using the provided document excerpts (given in no particular order) as sources. ALWAYS include a "SOURCES" section in your answer citing only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not have enough information to answer the question and leave the SOURCES section empty. Use only the provided documents and do not attempt to fabricate an answer.

---------

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
SOURCES: 1-32
Content: While we're at it, let's make sure every American can get the health care they need. \n\nWe've already made historic investments in health care. \n\nWe've made it easier for Americans to get the care they need, when they need it. \n\nWe've made it easier for Americans to get the treatments they need, when they need them. \n\nWe've made it easier for Americans to get the medications they need, when they need them.
SOURCES: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat's why I'm calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
SOURCES: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
SOURCES: 1-32

---------

QUESTION: {question}
=========
{context}
=========
FINAL ANSWER:

    """

    CUSTOM_PROMPT = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
    )
    return conversation_chain


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with Your Docs", page_icon="📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    # --- Sidebar for Configuration and File Upload ---
    with st.sidebar:
        st.title("Configuration")
        st.markdown("---")

        if not configure_gemini():
            st.stop()

        st.header("Upload Your Files")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", accept_multiple_files=True, type=["pdf", "txt"]
        )

        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
                    documents = get_file_text(uploaded_files)
                    if documents:
                        chunks = get_text_chunks(documents)
                        vector_store = get_vector_store(chunks)
                        if vector_store:
                            st.session_state.conversation = get_conversation_chain(
                                vector_store
                            )
                            st.session_state.processed = True
                            st.session_state.messages = (
                                []
                            )  # Clear chat on new processing
                            st.success("✅ Documents processed successfully!")
                        else:
                            st.error("❌ Failed to create vector store.")
                    else:
                        st.error("❌ No text could be extracted from the files.")
            else:
                st.warning("⚠️ Please upload at least one file.")

        st.markdown("---")
        st.info(
            "Your documents are processed for this session only and are not stored permanently."
        )

    # --- Main Chat Interface ---
    st.title("Chat with Your Documents 💬")
    st.markdown(
        "Upload your documents in the sidebar, and then ask questions about their content."
    )

    if not st.session_state.processed:
        st.info(
            "Get started by uploading your files and clicking 'Process Documents' in the sidebar."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.processed:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.invoke(
                        {"question": prompt}
                    )
                    answer = response["answer"]
                    sources = response["source_documents"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(
                                    f"**Source {i}:** `{source.metadata['source']}` (Page: {source.metadata.get('page', 'N/A')})"
                                )
                                st.info(f"Content: *'{source.page_content[:250]}...'*")
        else:
            st.warning(
                "Please upload and process your documents before asking questions."
            )


if __name__ == "__main__":
    main()
