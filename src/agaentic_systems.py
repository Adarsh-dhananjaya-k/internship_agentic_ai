import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def main():
    st.set_page_config(page_title="Simple RAG App", layout="wide")
    st.title("📄 Simple RAG Application")

    # Sidebar for configuration
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar.")
        return

    if uploaded_file:
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Process document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load and split text
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(texts, embeddings)

            # Setup QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4o-mini"),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            # User interaction
            query = st.text_input("Ask a question about the document:")
            if query:
                with st.spinner("Generating answer..."):
                    response = qa_chain.invoke(query)
                    st.markdown("### Answer:")
                    st.write(response["result"])

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    main()
