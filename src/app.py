import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Document Analyzer & Research Assistant", layout="wide")
st.title("📚 Document Analyzer & Research Assistant")

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'doc_metadata' not in st.session_state:
    st.session_state.doc_metadata = None
if 'question_submitted' not in st.session_state:
    st.session_state.question_submitted = False

# Add a sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = os.environ["OPENAI_API_KEY"]
    # if api_key:
    #     os.environ["OPENAI_API_KEY"] = api_key

    model_option = st.selectbox(
        "Select LLM Model:",
        ["gpt-3.5-turbo", "gpt-4"]
    )

    chunk_size = st.slider("Chunk Size:", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=500, value=200, step=50)

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")


# Main function to process the PDF
def process_pdf(file):
    # Create a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, file.name)

    # Save the uploaded file
    with open(temp_filepath, "wb") as f:
        f.write(file.getbuffer())

    # Load the PDF
    loader = PyPDFLoader(temp_filepath)
    documents = loader.load()

    # Extract document metadata
    doc_metadata = {
        "filename": file.name,
        "pages": len(documents),
        "total_chars": sum(len(doc.page_content) for doc in documents)
    }

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create a memory object
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create a conversational chain
    llm = ChatOpenAI(temperature=0, model_name=model_option)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )

    return conversation, vectorstore, doc_metadata


# Define callback for question submission
def submit_question():
    st.session_state.question_submitted = True


# Process the file when uploaded
if uploaded_file and st.session_state.processed_file != uploaded_file.name:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Processing your PDF... This may take a minute."):
            try:
                st.session_state.conversation, st.session_state.vectorstore, st.session_state.doc_metadata = process_pdf(
                    uploaded_file)
                st.session_state.processed_file = uploaded_file.name
                st.session_state.chat_history = []
                st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Display document information and summary
if st.session_state.conversation and st.session_state.doc_metadata:
    st.header("Document Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filename", st.session_state.doc_metadata["filename"])
    with col2:
        st.metric("Pages", st.session_state.doc_metadata["pages"])
    with col3:
        st.metric("Characters", st.session_state.doc_metadata["total_chars"])

    # Generate document summary only once
    if len(st.session_state.chat_history) == 0:
        with st.spinner("Generating document summary..."):
            query = "Please provide a comprehensive summary of this document. Include the main topic, key points, and any important conclusions."
            result = st.session_state.conversation({"question": query})
            st.session_state.chat_history.append(("Summary Request", result["answer"]))

            # Display the summary
            st.header("Document Summary")
            st.write(result["answer"])

# Display chat interface
if st.session_state.conversation:
    st.header("Ask Questions About Your Document")

    # Create a form for question input
    with st.form(key="question_form"):
        query = st.text_input("Ask a question about your document:")
        submit_button = st.form_submit_button("Submit Question", on_click=submit_question)

    # Process the question only when submitted
    if st.session_state.question_submitted and query:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.conversation({"question": query})
                st.session_state.chat_history.append((query, result["answer"]))
                # Reset the submission flag after processing
                st.session_state.question_submitted = False
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.question_submitted = False

    # Display chat history (excluding the summary)
    if len(st.session_state.chat_history) > 0:
        st.header("Conversation History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            # Skip displaying the summary in the conversation
            if question == "Summary Request":
                continue
            with st.expander(f"Q: {question}", expanded=True):
                st.write(answer)

# Add a section for semantic search directly in the vector database
if st.session_state.vectorstore:
    st.header("Semantic Search")

    # Create a form for search
    with st.form(key="search_form"):
        search_query = st.text_input("Search for specific content in the document:")
        k_results = st.slider("Number of results:", min_value=1, max_value=10, value=3)
        search_button = st.form_submit_button("Search")

    if search_button and search_query:
        with st.spinner("Searching..."):
            search_results = st.session_state.vectorstore.similarity_search(search_query, k=k_results)

            st.subheader("Search Results")
            for i, doc in enumerate(search_results, 1):
                with st.expander(f"Result {i} - Page {doc.metadata.get('page', 'N/A')}"):
                    st.write(doc.page_content)
                    st.caption(f"Source: Page {doc.metadata.get('page', 'N/A')}")

# Add a footer
st.markdown("---")
st.caption("Document Analyzer & Research Assistant - Built with Streamlit, LangChain, and OpenAI")