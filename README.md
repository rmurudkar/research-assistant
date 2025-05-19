# PDF Analyzer & Research Assistant

This application allows you to upload PDF documents, analyze their content using embeddings, and ask questions about the text. It combines document analysis with a research assistant to help you extract information and insights from your documents.

## Features

- **PDF Upload & Processing**: Upload any PDF document and process it for analysis.
- **Document Summarization**: Automatically generate a comprehensive summary of the uploaded document.
- **Question & Answer**: Ask specific questions about the document content and get accurate answers.
- **Semantic Search**: Search for specific information within the document using natural language queries.
- **Document Metadata**: View basic metadata about your document such as page count and word count.
- **Vector Store Management**: Efficiently store and retrieve document embeddings for fast retrieval.

## Technology Stack

- **Streamlit**: For the web interface
- **LangChain**: For document processing and conversational chains
- **OpenAI**: For embeddings and language model
- **FAISS**: For vector storage and similarity search
- **PyPDF**: For PDF document handling

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf-analyzer.git
   cd pdf-analyzer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Alternatively, you can enter your API key in the application's sidebar.

### Running the Application

Start the Streamlit application:
```
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

## Usage

1. Enter your OpenAI API key in the sidebar (if not set in the `.env` file).
2. Upload a PDF document using the file uploader.
3. Wait for the document to be processed and the summary to be generated.
4. Ask questions about the document in the question input field.
5. Use the semantic search feature to find specific information in the document.

## Customization

- Adjust chunk size and overlap in the sidebar to optimize for different types of documents.
- Choose between different OpenAI models for different performance and cost trade-offs.

## Limitations

- The application is designed for text-based PDFs. Scanned documents or PDFs with complex layouts may not work well without OCR preprocessing.
- Processing very large documents may take some time and consume more API usage.
- The quality of answers depends on the quality of the document and the language model being used.

## Future Enhancements

- Support for more document types (DOCX, TXT, etc.)
- Multi-document analysis and cross-referencing
- Document comparison features
- Advanced visualization of document topics and concepts
- Integration with web search for supplementary information

