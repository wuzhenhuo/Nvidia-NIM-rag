# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a simple standalone implementation showing rag pipeline using Nvidia AI Foundational models.
# It uses a simple Streamlit UI and multiple file implementation of a minimalistic RAG pipeline, managing each session seperately.
import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid

# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
llm = ChatNVIDIA(model="mixtral_8x7b")
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")

# Initialize SessionState to manage session-specific data
def initialize_session_state():
    if "session_data" not in st.session_state:
        st.session_state.session_data = {}

# Load vector store
def load_vectorstore(raw_documents):
    if "vector_file" in st.session_state.session_data:
        vector_store_path = st.session_state.session_data["vector_file"]
        with st.spinner("Splitting new documents into chunks..."):
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            documents = text_splitter.split_documents(raw_documents)
        with st.spinner("Appending document chunks to vector database..."):
            vectorstore = FAISS.from_documents(documents, document_embedder)
        with st.spinner("Saving vector store"):
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)
        st.session_state.session_data["vectorstore"] = vectorstore
        st.sidebar.success("Vector store updated with new file.")
    else:
        with st.sidebar:
            if raw_documents:
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    documents = text_splitter.split_documents(raw_documents)

                with st.spinner("Adding document chunks to vector database..."):
                    vector_store_path = f"vectorstore_{str(uuid.uuid4())}.pkl"  
                    vectorstore = FAISS.from_documents(documents, document_embedder)
                    st.session_state.session_data["vector_file"] = vector_store_path
                    st.session_state.session_data["vectorstore"] = vectorstore

                with st.spinner("Saving vector store"):
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                st.sidebar.success("Vector store created and saved.")
            else:
                st.sidebar.warning("No documents available to process!", icon="⚠️")

# Document Loader
def document_loader():
    with st.sidebar:
        st.subheader("Add to the Knowledge Base")
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
            submitted = st.form_submit_button("Upload!")

        if uploaded_files and submitted:
            if "uploaded_dir" not in st.session_state.session_data:
                doc_dir = os.path.abspath(f"./uploaded_docs_{str(uuid.uuid4())}")
                if not os.path.exists(doc_dir):
                    os.makedirs(doc_dir)
                st.session_state.session_data["uploaded_dir"] = doc_dir
            else:
                doc_dir = st.session_state.session_data["uploaded_dir"]
                
            for uploaded_file in uploaded_files:
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                with open(os.path.join(doc_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.read())
            session_uploaded_dir = st.session_state.session_data["uploaded_dir"]
            
            raw_documents = DirectoryLoader(session_uploaded_dir).load()
            load_vectorstore(raw_documents)

# Response Generation and Chat
def response_generation_and_chat():
    st.subheader("Chat with your AI Assistant, Envie!")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
    )
    user_input = st.chat_input("Can you tell me what NVIDIA is known for?")

    chain = prompt_template | llm | StrOutputParser()

    if user_input:
        if "vectorstore" in st.session_state.session_data:
            st.session_state.messages.append({"role": "user", "content": user_input})
            retriever = st.session_state.session_data["vectorstore"].as_retriever()
            docs = retriever.get_relevant_documents(user_input)
            with st.chat_message("user"):
                st.markdown(user_input)

            context = ""
            for doc in docs:
                context += doc.page_content + "\n\n"

            augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for response in chain.stream({"input": augmented_user_input}):
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("No documents provided as context", icon="⚠️")

def main():
    st.set_page_config(layout="wide")
    initialize_session_state()
    document_loader()
    response_generation_and_chat()

if __name__ == "__main__":
    main()
