from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Research Paper Concept Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Title
st.markdown('<div class="main-header">🔬 Research Paper Concept Explorer</div>', unsafe_allow_html=True)

st.markdown("""
**Explore research papers through their key concepts and relationships.**
Start by selecting a concept below to see its connections, or browse the interactive visualizations.
""")

# Sample data based on the provided example
sample_data = """FROM: Graphene nanoplatelets (GNPs)
TO: Neuronal differentiation
RELATIONSHIP: GNPs enhance the expression of neuronal markers MAP2, Nestin, and Tuj1, promoting the differentiation of human bone marrow mesenchymal stem cells (hBMSCs) into neurons.

FROM: GNP concentration
TO: Cell viability
RELATIONSHIP: Lower concentrations of GNPs maintained cell viability, whereas higher concentrations were detrimental to hBMSCs.

FROM: 0.4 µg/ml GNP coating
TO: Morphological changes
RELATIONSHIP: This specific concentration leads to observable morphological changes and increased fluorescence in hBMSC cultures, indicating enhanced differentiation.

FROM: Calcium imaging with Fluo4-AM
TO: Neuronal activity assessment
RELATIONSHIP: This method is used to show increased neuronal activity, underscoring GNPs' role in neuronal maturation.

FROM: GNPs
TO: Axon guidance pathway activation
RELATIONSHIP: GNPs influence the axon guidance pathway, among others, driving mechanisms involved in neuronal differentiation.

FROM: GNPs + Gelatin coating
TO: Neuronal differentiation enhancement
RELATIONSHIP: The combination provides a synergistic environment that supports effective neuronal differentiation of hBMSCs.

FROM: MTT assay
TO: GNP cytotoxicity evaluation
RELATIONSHIP: The assay shows that higher concentrations of GNPs lead to reduced cell viability, reinforcing the importance of optimizing GNP dosage for cell culture applications.

FROM: Lentiviral transduction with LeGO-G2-PURO
TO: GFP tracking of differentiation
RELATIONSHIP: The use of GFP-expressing hBMSCs allows for long-term tracking of neuronal differentiation under different conditions.

FROM: GNP concentration
TO: MAP2 expression
RELATIONSHIP: Higher GNP concentrations result in increased MAP2 expression, indicating enhanced neuronal differentiation.

FROM: GNP treatment
TO: Tuj1 expression
RELATIONSHIP: GNPs significantly increase Tuj1 expression across different concentrations, supporting the continuation of neuronal differentiation.

FROM: Gelatin and GNP coatings
TO: Enhanced fluorescence intensity
RELATIONSHIP: Both coatings, especially at 0.4 µg/ml GNP, promote increased fluorescence intensity, indicating neuronal differentiation.

FROM: GNP coating
TO: Calcium signaling
RELATIONSHIP: GNP coatings enhance calcium responses in differentiated neurons, reflecting their functional maturation and excitability.

FROM: ATP stimulation
TO: Ca2+ signaling enhancement
RELATIONSHIP: ATP-induced Ca2+ transients are stronger with gelatin and GNP coatings, suggesting enhanced signaling responses in differentiated neurons."""


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


def get_concept_categories():
    """Define concept categories for better organization"""
    return {
        'Materials': ['GNPs', 'Graphene nanoplatelets (GNPs)', 'GNP concentration', '0.4 µg/ml GNP coating',
                      'GNPs + Gelatin coating', 'Gelatin and GNP coatings', 'GNP coating', 'GNP treatment'],
        'Biological Processes': ['Neuronal differentiation', 'Cell viability', 'Morphological changes',
                                 'Neuronal differentiation enhancement', 'MAP2 expression', 'Tuj1 expression',
                                 'Enhanced fluorescence intensity', 'Calcium signaling', 'Ca2+ signaling enhancement',
                                 'Axon guidance pathway activation'],
        'Research Methods': ['Calcium imaging with Fluo4-AM', 'MTT assay', 'Lentiviral transduction with LeGO-G2-PURO',
                             'ATP stimulation'],
        'Measurements': ['Neuronal activity assessment', 'GNP cytotoxicity evaluation',
                         'GFP tracking of differentiation']
    }



def categorize_concept(concept, categories):
    """Categorize a concept based on predefined categories"""
    for category, concepts in categories.items():
        if any(c.lower() in concept.lower() or concept.lower() in c.lower() for c in concepts):
            return category
    return 'Other'


def create_concept_explorer(relationships):
    """Create an interactive concept explorer"""
    categories = get_concept_categories()

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
    categories = get_concept_categories()
    category = categorize_concept(concept, categories)

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


# Sidebar
st.sidebar.header("📊 Data Input")

# Data input section
use_sample = st.sidebar.checkbox("Use Sample Data (GNP Research)", value=True)

if use_sample:
    input_text = sample_data
    st.sidebar.success("✅ Using sample research paper data")
else:
    input_text = st.sidebar.text_area(
        "Paste your relationship data here:",
        height=200,
        placeholder="""FROM: Concept A
TO: Concept B
RELATIONSHIP: Description of how A relates to B

FROM: Concept C
TO: Concept D
RELATIONSHIP: Description of how C relates to D"""
    )

# Main content
if input_text:
    relationships = parse_relationships(input_text)

    if relationships:
        # Create tabs for different views
        tab1, tab2, tab4 = st.tabs(["🎯 Concept Explorer", "🌊 Flow Diagram", "📊 Analytics"])

        with tab1:
            st.markdown("### 🎯 Explore Concepts by Category")
            st.markdown("Click on any concept below to see its connections and relationships.")

            categorized_concepts = create_concept_explorer(relationships)

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
                show_concept_details(st.session_state.selected_concept, relationships)


    else:
        st.warning("⚠️ No valid relationships found. Please check your input format.")

else:
    st.info("👆 Please enter your relationship data in the sidebar or use the sample data to get started.")

# Footer with instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use This Tool

1. **Start with Concept Explorer**: Browse concepts by category and click to see connections
2. **View Flow Diagram**: Understand how concepts flow and influence each other
3. **Explore Network**: See the full network with interactive nodes
4. **Analyze Data**: Get insights and search for specific relationships

**💡 Pro Tips:**
- Node sizes indicate how connected a concept is
- Colors represent different concept categories
- Use the search function to find specific topics
- Click on concepts in the explorer to see detailed relationships
""")