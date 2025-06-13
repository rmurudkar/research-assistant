def parse_category_string(category_string):
    """
    Parse a category string format and return a dictionary with categories as keys
    and lists of concepts as values.

    Args:
        category_string (str): String in the format:
            CATEGORY: [Category Name 1]
            CONCEPTS: concept1, concept2, concept3

            CATEGORY: [Category Name 2]
            CONCEPTS: concept4, concept5, concept6

    Returns:
        dict: Dictionary with category names as keys and lists of concept strings as values
    """
    categories = {}
    current_category = None

    # Split by lines and process each line
    lines = category_string.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check for category line
        if line.startswith('CATEGORY:'):
            # Extract category name, removing brackets if present
            category_part = line.replace('CATEGORY:', '').strip()
            # Remove brackets if they exist
            if category_part.startswith('[') and category_part.endswith(']'):
                current_category = category_part[1:-1].strip()
            else:
                current_category = category_part

            # Initialize category in dictionary
            if current_category:
                categories[current_category] = []

        # Check for concepts line
        elif line.startswith('CONCEPTS:') and current_category:
            # Extract concepts part
            concepts_part = line.replace('CONCEPTS:', '').strip()

            # Split by comma and clean up each concept
            if concepts_part:
                concept_list = [concept.strip() for concept in concepts_part.split(',')]
                # Filter out empty strings
                concept_list = [concept for concept in concept_list if concept]
                categories[current_category] = concept_list

    return categories


# Test function with example data
def test_parser():
    """Test the parser function with sample data"""

    test_string = """
CATEGORY: [Category Name 1]
CONCEPTS: concept1, concept2, concept3

CATEGORY: [Category Name 2] 
CONCEPTS: concept4, concept5, concept6

CATEGORY: [Category Name 3]
CONCEPTS: concept7, concept8
"""

    result = parse_category_string(test_string)
    print("Parsed Categories:")
    for category, concepts in result.items():
        print(f"  {category}: {concepts}")

    return result


# Test with various edge cases
def test_edge_cases():
    """Test the parser with various edge cases"""

    print("\n=== Testing Edge Cases ===")

    # Test 1: No brackets around category names
    test1 = """
CATEGORY: Materials and Methods
CONCEPTS: graphene, nanoplatelets, coating

CATEGORY: Biological Processes
CONCEPTS: cell differentiation, neural development
"""

    print("\nTest 1 - No brackets:")
    result1 = parse_category_string(test1)
    for category, concepts in result1.items():
        print(f"  {category}: {concepts}")

    # Test 2: Extra whitespace and empty lines
    test2 = """

CATEGORY: [Research Methods]  
CONCEPTS:  microscopy,  imaging,   analysis  

CATEGORY: [Data Analysis]
CONCEPTS: statistics, visualization

"""

    print("\nTest 2 - Extra whitespace:")
    result2 = parse_category_string(test2)
    for category, concepts in result2.items():
        print(f"  {category}: {concepts}")

    # Test 3: Single concept per category
    test3 = """
CATEGORY: [Main Topic]
CONCEPTS: primary concept

CATEGORY: [Secondary Topic]
CONCEPTS: secondary concept
"""

    print("\nTest 3 - Single concepts:")
    result3 = parse_category_string(test3)
    for category, concepts in result3.items():
        print(f"  {category}: {concepts}")


# Advanced version with error handling and validation
def parse_category_string_robust(category_string, validate_concepts=None):
    """
    Robust version of the category parser with error handling and validation.

    Args:
        category_string (str): String to parse
        validate_concepts (list, optional): List of valid concepts to match against

    Returns:
        dict: Parsed categories with error information
    """
    result = {
        'categories': {},
        'errors': [],
        'unmatched_concepts': []
    }

    try:
        categories = {}
        current_category = None
        line_number = 0

        lines = category_string.strip().split('\n')

        for line in lines:
            line_number += 1
            line = line.strip()

            if not line:
                continue

            if line.startswith('CATEGORY:'):
                category_part = line.replace('CATEGORY:', '').strip()
                if category_part.startswith('[') and category_part.endswith(']'):
                    current_category = category_part[1:-1].strip()
                else:
                    current_category = category_part

                if not current_category:
                    result['errors'].append(f"Line {line_number}: Empty category name")
                elif current_category in categories:
                    result['errors'].append(f"Line {line_number}: Duplicate category '{current_category}'")
                else:
                    categories[current_category] = []

            elif line.startswith('CONCEPTS:'):
                if not current_category:
                    result['errors'].append(f"Line {line_number}: Concepts found without category")
                    continue

                concepts_part = line.replace('CONCEPTS:', '').strip()

                if concepts_part:
                    concept_list = [concept.strip() for concept in concepts_part.split(',')]
                    concept_list = [concept for concept in concept_list if concept]

                    # Validate concepts if validation list provided
                    if validate_concepts:
                        validated_concepts = []
                        for concept in concept_list:
                            if concept in validate_concepts:
                                validated_concepts.append(concept)
                            else:
                                # Try to find close matches
                                close_matches = [c for c in validate_concepts
                                                 if concept.lower() in c.lower() or c.lower() in concept.lower()]
                                if close_matches:
                                    validated_concepts.append(close_matches[0])
                                    result['errors'].append(f"Concept '{concept}' matched to '{close_matches[0]}'")
                                else:
                                    result['unmatched_concepts'].append(concept)

                        categories[current_category] = validated_concepts
                    else:
                        categories[current_category] = concept_list

        result['categories'] = categories

    except Exception as e:
        result['errors'].append(f"Parsing error: {str(e)}")

    return result


if __name__ == "__main__":
    # Run tests
    test_parser()
    test_edge_cases()

    print("\n=== Testing Robust Parser ===")
    test_string = """
CATEGORY: [Research Methods]
CONCEPTS: microscopy, imaging, invalid_concept

CATEGORY: [Data Analysis]
CONCEPTS: statistics, visualization
"""

    valid_concepts = ['microscopy', 'imaging', 'statistics', 'visualization', 'analysis']
    robust_result = parse_category_string_robust(test_string, valid_concepts)

    print(f"Categories: {robust_result['categories']}")
    print(f"Errors: {robust_result['errors']}")
    print(f"Unmatched: {robust_result['unmatched_concepts']}")