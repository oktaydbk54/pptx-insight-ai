import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.vectorstores import Chroma
import chromadb
from dotenv import load_dotenv
import tempfile
import ssl

load_dotenv()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def create_chroma_client():
    try:
        client = chromadb.Client()
        return client
    except Exception as e:
        st.error(f"Error creating Chroma client: {e}")
        return None

def create_chroma_collection(client, collection_name="consent_collection"):
    try:
        if collection_name not in [col['name'] for col in client.list_collections()]:
            client.create_collection(collection_name)
        return client.get_collection(collection_name)
    except Exception as e:
        st.error(f"Error creating Chroma collection: {e}")
        return None

def load_chunk_persist_pptx(pptx_folder_path):
    documents = []
    for file in os.listdir(pptx_folder_path):
        if file.endswith('.pptx'):
            pptx_path = os.path.join(pptx_folder_path, file)
            loader = UnstructuredPowerPointLoader(pptx_path)
            documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    
    client = chromadb.Client()
    if not client.list_collections():
        client.create_collection("consent_collection")
    
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    return vectordb

# Function to create the QA chain
def create_agent_chain(model_choice):
    if model_choice == "GPT-4o":
        llm = ChatOpenAI(model_name="gpt-4o")
    elif model_choice == 'GPT-4o-mini':
        llm = ChatOpenAI(model_name='gpt-4o-mini')
    elif model_choice == "Claude 3 Opus":
        llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=os.getenv('ANTHROPIC_API_KEY'))
    elif model_choice == "Claude 3 Sonnet":
        llm = ChatAnthropic(model="claude-3-sonnet-20240229", api_key=os.getenv("ANTHROPIC_API_KEY"))
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# Function to get response from the LLM
def get_llm_response(query, vectordb, chain):
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "ChatAI"])

model_choice = st.sidebar.selectbox("Choose a model", ["GPT-4o","GPT-4o-mini" , "Claude 3 Opus", "Claude 3 Sonnet"])

if page == "Main":
    st.title("Document Upload and Processing")

    # File uploader for PPTX files
    uploaded_files = st.file_uploader("Upload PPTX files", type=["pptx"], accept_multiple_files=True)

    if uploaded_files:
        pptx_folder_path = tempfile.mkdtemp()
        os.makedirs(pptx_folder_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            pptx_path = os.path.join(pptx_folder_path, uploaded_file.name)
            with open(pptx_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success("PPTX files uploaded successfully!")

        if st.button("Process PPTX and Create Embeddings"):
            vectordb = load_chunk_persist_pptx(pptx_folder_path)
            if vectordb:
                chain = create_agent_chain(model_choice)
                st.session_state.vectordb = vectordb
                st.session_state.chain = chain
                st.success("PPTX files processed and vector database created!")
            else:
                st.error("Failed to create vector database. Please check the logs for more details.")

elif page == "ChatAI":
    st.title("Chat with your Documents")

    if "vectordb" in st.session_state and "chain" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask a question:")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = get_llm_response(prompt, st.session_state.vectordb, st.session_state.chain)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.warning("Please upload and process your documents on the 'Main' page first.")
