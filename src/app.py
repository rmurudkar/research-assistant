import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
# Build the retrieval chain using LCEL
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from prompts.process_pdf_prompt import process_pdf_prompt
from parse_categories_and_concept_output import parse_category_string
from category_and_concept_analyzer import get_categories_and_concepts_and_edges

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
if 'categories' not in st.session_state:
    st.session_state.categories = ""
if 'relationships' not in st.session_state:
    st.session_state.relationships = ""
if 'edges' not in st.session_state:
    edges = ""


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }

    .concept-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }

    .relationship-card {
        background: #9c790b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }

    .metric-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #c3e6cb;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #g0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .selected-concept {
        background: #0d5678;
        border: 2px solid #2196f3;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def parse_relationships(text):
    """Parse the relationship text into structured data"""
    relationships = []
    sections = text.strip().split('\n\n')

    for section in sections:
        lines = section.strip().split('\n')
        if len(lines) >= 3:
            from_concept = lines[0].replace('FROM: ', '').strip()
            to_concept = lines[1].replace('TO: ', '').strip()
            relationship = lines[2].replace('RELATIONSHIP: ', '').strip()

            relationships.append({
                'from': from_concept,
                'to': to_concept,
                'relationship': relationship,
                'weight': len(relationship.split())
            })

    return relationships
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

    research_assistant_prompt = ChatPromptTemplate.from_template(process_pdf_prompt)

    # Create a conversational chain
    llm = ChatOpenAI(temperature=0.3, model_name=model_option)

    # Create a vector store retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Chain to combine documents
    document_chain = create_stuff_documents_chain(llm, research_assistant_prompt)

    qa_chain = create_retrieval_chain(retriever, document_chain)

    categories, edges = get_categories_and_concepts_and_edges(temp_filepath)

    print("CATEGORIES: ", categories)
    print("EDGES", edges)

    # relationships_stg = get_relationships(edges)
    relationships = []
    for edge in edges:
        relationships.append({
            'from': edge[0],
            'to': edge[1],
            'relationship': edge[2]
        })

    print("RELATIONSHIPS", relationships)



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
                    "input": inputs["question"],
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

    return conversation, vectorstore, doc_metadata, categories, relationships, edges


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
                st.session_state.conversation, st.session_state.vectorstore, st.session_state.doc_metadata, st.session_state.categories, st.session_state.relationships, st.session_state.edges = process_pdf(uploaded_file)
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
                "input": query,
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
            # if question == "Summary Request":
            #     continue
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

def categorize_concept(concept, categories):
    """Categorize a concept based on predefined categories"""
    for category, concepts in categories.items():
        if any(c.lower() in concept.lower() or concept.lower() in c.lower() for c in concepts):
            return category
    return 'Other'






def create_concept_explorer(relationships, categories):
    """Create an interactive concept explorer"""
    # Get all unique concepts
    all_concepts = set()
    for rel in relationships:
        all_concepts.add(rel['from'])
        all_concepts.add(rel['to'])

    # Categorize concepts
    categorized_concepts = {}
    for concept in all_concepts:
        category = categorize_concept(concept, categories)
        if category not in categorized_concepts:
            categorized_concepts[category] = []
        categorized_concepts[category].append(concept)

    # Sort concepts within categories
    for category in categorized_concepts:
        categorized_concepts[category].sort()

    return categorized_concepts


tab1, tab2, tab4 = st.tabs(["🎯 Concept Explorer", "🌊 Flow Diagram", "📊 Analytics"])
# categories, edges = get_categories_and_concepts_and_edges('uploaded_pdf.pdf')
# relationships = get_relationships(edges)
# relationships = parse_relationships(relationships)

def show_concept_details(concept, relationships):
    """Show detailed information about a selected concept"""
    # Find all relationships involving this concept
    related_relationships = []
    for rel in relationships:
        if rel['from'] == concept or rel['to'] == concept:
            related_relationships.append(rel)

    if not related_relationships:
        st.write("No relationships found for this concept.")
        return

    # Show concept information
    category = categorize_concept(concept, st.session_state.categories)

    st.markdown(f"""
    <div class="selected-concept">
        <h3>🎯 {concept}</h3>
        <p><strong>Category:</strong> {category}</p>
        <p><strong>Total Connections:</strong> {len(related_relationships)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Show incoming and outgoing relationships
    incoming = [rel for rel in related_relationships if rel['to'] == concept]
    outgoing = [rel for rel in related_relationships if rel['from'] == concept]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⬅️ What affects this concept:")
        if incoming:
            for rel in incoming:
                st.markdown(f"""
                <div class="relationship-card">
                    <strong>{rel['from']}</strong><br>
                    <small>{rel['relationship']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No incoming relationships found.")

    with col2:
        st.markdown("#### ➡️ What this concept affects:")
        if outgoing:
            for rel in outgoing:
                st.markdown(f"""
                <div class="relationship-card">
                    <strong>{rel['to']}</strong><br>
                    <small>{rel['relationship']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No outgoing relationships found.")



if st.session_state.vectorstore:

    with tab1:
        st.markdown("### 🎯 Explore Concepts by Category")
        st.markdown("Click on any concept below to see its connections and relationships.")

        categorized_concepts = create_concept_explorer(st.session_state.relationships, st.session_state.categories)

        # Create concept explorer with categories
        for category, concepts in categorized_concepts.items():
            with st.expander(f"📁 {category} ({len(concepts)} concepts)", expanded=True):
                cols = st.columns(3)
                for i, concept in enumerate(concepts):
                    with cols[i % 3]:
                        if st.button(concept, key=f"concept_{concept}", use_container_width=True):
                            st.session_state.selected_concept = concept

        # Show details for selected concept
        if hasattr(st.session_state, 'selected_concept'):
            st.markdown("---")
            show_concept_details(st.session_state.selected_concept, st.session_state.relationships)




# Add a footer
st.markdown("---")
st.caption("Document Analyzer & Research Assistant - Built with Streamlit, LangChain, and OpenAI")

