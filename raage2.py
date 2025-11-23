import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# langchain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# load environment
load_dotenv()

# page config
st.set_page_config(page_title=" 📝 Rag Q&A", layout="wide")

st.title("📝 Rag Q&A with Multiple PDFs + Chat History")

# Sidebar
with st.sidebar:
    st.header("⚙️ Config")
    api_key = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs -> Ask questions -> Get Answers")

if not api_key:
    st.warning(" 🔑  Please enter your Groq API Key in the sidebar")
# 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

# LLM + embeddings
llm = ChatGroq(api_key=api_key, model="openai/gpt-oss-120b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# upload PDFs
uploaded_files = st.file_uploader(" 📚 Upload PDF files", type="pdf", accept_multiple_files=True)

all_docs = []

if uploaded_files:
    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = pdf.name
            all_docs.extend(docs)
    st.success(" ✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")
else:
    st.info(" Please upload one or more PDFs to begin")
    st.stop()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(all_docs)

# cache vectorstore
@st.cache_resource
def get_vectorstore(_splits):
    return Chroma.from_documents(_splits, embeddings,persist_directory="./chroma_index")

vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# debug: show how many chunks wwere embedded
st.sidebar.write(f" 🔍 Indexed {len(splits)} chunks into vectorstore")

# history aware retriever
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system","Rephrase user questions using chat history."),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# QA chain with context injection
qa_prompt = ChatPromptTemplate.from_messages([
    ("system","You are an assistant. Use the retrieved context below:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# SESSION STATE

if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id:str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# chat UI
session_id = st.text_input(" 🆔 Session ID", value="default_session")
user_q = st.chat_input(" 💬 Ask Question...")


if user_q:
    history = get_history(session_id)
    result = conversational_rag.invoke(
        {"input": user_q}, config={"configurable":{"session_id":session_id}}
    )

    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(result["answer"])

    # debug to show received docs
    if "context" in result:
        with st.expander(" 📑 Retrieved Chunks"):
            for doc in result["context"]:
                st.write(f"{doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')}):")
                st.write(doc.page_content[:300]+ "...")
    
    with st.expander(" 📚 Chat History"):
        for msg in history.messages:
            role = getattr(msg, "role", msg.type).title()
            st.write(f"**{role}:** {msg.content}")
                