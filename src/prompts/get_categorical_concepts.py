get_categorical_concepts_prompt = """You are a research analysis expert. Given the following list of concepts from a research paper, please:

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