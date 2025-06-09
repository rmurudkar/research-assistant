get_concepts_from_text_prompt = """
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