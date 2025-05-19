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
# Build the retrieval chain using LCEL
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# from ResearchAssistantWrapper import ResearchAssistantWrapper

# Load environment variables
load_dotenv()

#Map the user's preference to a prompt modifier
length_prompts = {
    "Concise": "Provide a brief answer in 2-3 sentences.",
    "Balanced": "Provide a moderately detailed answer.",
    "Detailed": "Provide a thorough and detailed answer with examples where possible.",
    "Comprehensive": "Provide an extensive and comprehensive answer with multiple examples, explanations of nuances, and thorough context."
}

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

    # Your existing code...
    response_length = st.select_slider(
        "Response Length:",
        options=["Concise", "Balanced", "Detailed", "Comprehensive"],
        value="Balanced"
    )

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

    from langchain.prompts import ChatPromptTemplate

    research_assistant_prompt = ChatPromptTemplate.from_template("""
    You are an advanced Research Assistant specialized in making complex academic and scientific papers accessible and understandable.
    Your core strength is breaking down sophisticated concepts, methodologies, and findings into clear explanations without losing accuracy or nuance.

    ## YOUR APPROACH TO RESEARCH PAPERS:
    - First identify the paper's structure, key arguments, methodology, and conclusions before answering
    - Break down complex technical terminology into simpler language while preserving meaning
    - Use analogies, metaphors, and real-world examples to illustrate abstract concepts
    - Explain the significance and implications of research findings for broader context
    - Clarify statistical analyses and data interpretations in straightforward terms
    - Connect new concepts to foundational knowledge to build understanding
    - Visualize complex processes through clear description (as if creating a diagram)
    - Identify the "so what" factor - why the research matters and to whom

    ## RESPONSE STRUCTURE:
    - Begin with the simplest expression of the concept, then add layers of complexity as needed
    - Use a scaffolded approach: start with foundations, then build to more advanced elements
    - Separate core concepts from supplementary details
    - Create clear sections with intuitive headings for complex explanations
    - Use bullet points for multi-step processes or lists of related concepts
    - Provide "In other words..." simplifications after explaining technical concepts
    - When explaining methods, clearly distinguish between what was done, how it was done, and why it matters

    {length_preference}

    ## HANDLING LIMITATIONS:
    - If you encounter highly specialized concepts that require simplification, explicitly acknowledge this
    - When multiple interpretations are possible, present the most accessible one first, then note alternatives
    - If you cannot fully explain a concept based on the provided context, acknowledge the limitations and explain what you can confidently address
    - Never oversimplify to the point of inaccuracy - maintain scientific integrity while improving accessibility
    - NEVER fabricate explanations, citations, or content not supported by the document

    CONTEXT:
    {context}

    QUESTION: {question}

    Remember: Your greatest value is transforming what might seem impenetrable to a non-expert into something that builds genuine understanding. 
    Prioritize clarity and comprehension while maintaining accuracy.
    """)

    # Create a conversational chain
    llm = ChatOpenAI(temperature=0.3, model_name=model_option)

    # Create a vector store retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Chain to combine documents
    document_chain = create_stuff_documents_chain(llm, research_assistant_prompt)

    qa_chain = create_retrieval_chain(retriever, document_chain)

    length_prompts = {
        "Concise": """Response Style: Provide a brief, focused explanation in 2-3 sentences that captures the essential concept in the most accessible terms. Use the simplest possible language and focus only on the core idea, stripping away technical complexity while preserving accuracy. This should be understandable to a non-expert.""",

        "Balanced": """Response Style: Provide a moderately detailed explanation of 1-2 paragraphs that bridges simplicity and depth. Begin with an accessible overview anyone could understand, then add a layer of more specific details. Use one simple analogy or example to illustrate the concept. Define key terms that are essential to understanding.""",

        "Detailed": """Response Style: Provide a thorough explanation that fully addresses the complexity while ensuring accessibility. Structure your response with clear sections for different aspects of the concept. Include:
        1) A simple overview for beginners
        2) More nuanced details for those with some background
        3) Practical examples or analogies that illustrate the concept
        4) Definitions of technical terms in plain language
        5) Connections to related concepts within the paper""",

        "Comprehensive": """Response Style: Provide an extensive educational explanation that transforms complex research into a learning journey. Your response should:
        1) Start with a "simplest possible explanation" that anyone could understand
        2) Progressively build in complexity through clearly marked sections
        3) Use multiple complementary examples and analogies
        4) Create "mental hooks" that connect new concepts to familiar ideas
        5) Explain how experts think about this concept vs. how beginners might approach it
        6) Address potential misconceptions or confusion points
        7) Include a brief "key takeaways" summary at the end that reinforces core concepts

        Organize this longer response with descriptive subheadings and visual language. Your goal is to make the reader feel like they've gained genuine insight into something that initially seemed beyond their understanding."""
    }

    # Create a custom wrapper to handle the extra parameter
    class ResearchAssistantWrapper:
        def __init__(self, qa_chain, memory):
            self.qa_chain = qa_chain
            self.memory = memory

        def __call__(self, inputs):
            # Get chat history from memory
            chat_history = self.memory.chat_memory.messages

            # Add length preference to inputs if present
            if "length_preference" in inputs:
                # Run the chain with the length preference
                result = self.qa_chain.invoke({
                    "question": inputs["question"],
                    "length_preference": inputs["length_preference"],
                    "chat_history": chat_history
                })
            else:
                # Run the chain without length preference
                result = self.qa_chain.invoke({
                    "question": inputs["question"],
                    "length_preference": "Response Style: Provide a balanced and complete answer with appropriate detail.",
                    "chat_history": chat_history
                })

            # Update memory
            self.memory.chat_memory.add_user_message(inputs["question"])
            self.memory.chat_memory.add_ai_message(result["answer"])

            # Return result in the expected format
            return {"answer": result["answer"]}

    conversation = ResearchAssistantWrapper(qa_chain, memory)

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
            result = st.session_state.conversation({
                "question": query,
                "length_preference": length_prompts[response_length]
            })
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

                # Get the length preference
                length_pref = length_prompts[response_length]

                result = st.session_state.conversation({
                    "question": query,
                    "length_preference": length_prompts[response_length]
                })
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