import re

import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

from prompts.get_concepts_from_text import get_concepts_from_text_prompt
from parse_categories_and_concept_output import parse_category_string


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
        return text  # [:max_chars]


# 🧠 Prompt for concept extraction
def get_concepts_from_text(text_chunk):
    formatted_prompt = get_concepts_from_text_prompt.format(document_chunk=text_chunk)

    client = OpenAI()

    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system",
             "content": "You're a helpful assistant that extracts knowledge graphs from academic text."},
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
        messages=[
            {'role': 'system',
             'content': "You are a research analysis expert specializing in categorizing scientific concepts."},
            {"role": "user", "content": prompt}

        ],
        max_tokens=1500,
        temperature=0.3
    )

    return completion.choices[0].message.content


def get_categories_and_concepts_and_edges(file):
    text_chunk = extract_text_from_pdf(file)

    concepts = get_concepts_from_text(text_chunk)

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
    edges, concepts = parse_llm_edges(concepts)

    # Run parser
    categories = get_categorical_concepts(text_chunk, list(concepts))

    categories = parse_category_string(category_string=categories)


    return categories, edges