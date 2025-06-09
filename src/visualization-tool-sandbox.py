import re

from openai import OpenAI
import os
import PyPDF2
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            # if len(text) > max_chars:
            #     break
        return text # [:max_chars]

# 🧠 Prompt for concept extraction
def get_concepts_from_text(text_chunk):
    prompt_template = """
    You are a helpful research assistant. You will be given the full text of a research paper. Your task is to extract directional relationships between key concepts **from the main body of the document only** — and **skip the abstract, introduction, and conclusion**.

    Focus on:
    1. Relationships and dependencies between detailed concepts, methods, technical challenges, and innovations presented in the **body** (e.g., methods, experiments, model architecture).
    2. Exclude high-level summaries or broad headline concepts that typically appear in the abstract or intro.

    Return your output as a **numbered list** in this format:

    Concept A -> Concept B: explanation of the relationship

    Only include concept-to-concept links. Do not include generic statements or summary-level insights.

    Here is the document text:
    \"\"\"
    {document_chunk}
    \"\"\"
    """

    formatted_prompt = prompt_template.format(document_chunk=text_chunk)

    client = OpenAI()

    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You're a helpful assistant that extracts knowledge graphs from academic text."},
            {"role": "user", "content": formatted_prompt}
        ],

    )

    return completion.choices[0].message.content

def get_categorical_concepts(paper_context, concepts_list):
    prompt = f"""You are a research analysis expert. Given the following list of concepts from a research paper, please:

    1. Create 4-6 meaningful categories that best organize these concepts
    2. Assign each concept to exactly one category
    3. Categories should be descriptive and reflect the research domain

    Research Context: {paper_context if paper_context else "General research paper"}

    Concepts to categorize:
    {concepts_list}

    Please respond in this exact format:

    CATEGORY: [Category Name 1]
    CONCEPTS: concept1, concept2, concept3

    CATEGORY: [Category Name 2] 
    CONCEPTS: concept4, concept5, concept6

    CATEGORY: [Category Name 3]
    CONCEPTS: concept7, concept8

    Make sure every concept is assigned to exactly one category. Categories should be:
    - Descriptive and meaningful
    - Relevant to the research domain
    - Between 4-6 total categories
    - Balanced in terms of number of concepts per category when possible"""



    client = OpenAI()

    completion = client.chat.completions.create(
        model='gpt-4o',
        messages = [
            {'role': 'system', 'content': "You are a research analysis expert specializing in categorizing scientific concepts." },
            {"role": "user", "content": prompt}

        ],
        max_tokens=1500,
        temperature=0.3
    )

    return completion.choices[0].message.content

# 🚀 Run the process
if __name__ == "__main__":
    print("📚 Extracting text from PDF...")
    text_chunk = extract_text_from_pdf("uploaded_pdf.pdf")
    # print("✅ Extracted text (truncated):\n", text_chunk, "...\n")

    print("🤖 Sending to OpenAI for concept extraction...\n")
    output = get_concepts_from_text(text_chunk)
    # print(output)

    print("🔎 Extracted Concept Relationships:\n")
    # print(output)


    # Parse into (source, target, relationship) tuples
    def parse_llm_edges(text):
        concepts = []
        edge_pattern = re.compile(r"\d+\.\s*(.*?)\s*->\s*(.*?):\s*(.*)")
        edges = []

        for match in edge_pattern.finditer(text):
            source, target, label = match.groups()
            edges.append((source.strip(), target.strip(), label.strip()))
            concepts.append(source)
            concepts.append(target)

        return edges, set(concepts)


    # Run parser
    edges, concepts = parse_llm_edges(output)

    # Print result
    for src, tgt, label in edges:
        print(f"FROM: {src}\nTO: {tgt}\nRELATIONSHIP: {label}\n")

    print(get_categorical_concepts(text_chunk, list(concepts)))


