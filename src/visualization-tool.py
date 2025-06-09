import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import numpy as np
from collections import Counter, defaultdict
import io
import base64

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

def get_concept_categories(text):


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


def create_sankey_diagram(relationships):
    """Create a Sankey diagram for concept flow"""
    # Get unique concepts and create indices
    all_concepts = set()
    for rel in relationships:
        all_concepts.add(rel['from'])
        all_concepts.add(rel['to'])

    concept_list = list(all_concepts)
    concept_to_idx = {concept: idx for idx, concept in enumerate(concept_list)}

    # Create category colors
    categories = get_concept_categories()
    category_colors = {
        'Materials': '#ff7f0e',
        'Biological Processes': '#2ca02c',
        'Research Methods': '#d62728',
        'Measurements': '#9467bd',
        'Other': '#8c564b'
    }

    # Assign colors to concepts
    node_colors = []
    for concept in concept_list:
        category = categorize_concept(concept, categories)
        node_colors.append(category_colors.get(category, '#8c564b'))

    # Create source, target, and value lists
    source = []
    target = []
    value = []

    for rel in relationships:
        source.append(concept_to_idx[rel['from']])
        target.append(concept_to_idx[rel['to']])
        value.append(1)  # All relationships have equal weight in Sankey

    # Create labels with line breaks for long names
    labels = []
    for concept in concept_list:
        if len(concept) > 25:
            words = concept.split()
            if len(words) > 3:
                mid = len(words) // 2
                label = ' '.join(words[:mid]) + '<br>' + ' '.join(words[mid:])
            else:
                label = concept
        else:
            label = concept
        labels.append(label)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(128,128,128,0.3)'
        )
    )])

    fig.update_layout(
        title_text="Concept Flow Diagram",
        font_size=10,
        height=600
    )

    return fig


def create_hierarchical_tree(relationships):
    """Create a hierarchical tree visualization"""
    # Build a graph to find the most central concepts
    G = nx.DiGraph()
    for rel in relationships:
        G.add_edge(rel['from'], rel['to'])

    # Find concepts with highest in-degree (most targeted)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Create a hierarchical layout
    fig = go.Figure()

    # Use a circular layout for better visibility
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Calculate node sizes based on total degree
    node_degrees = {node: in_degrees.get(node, 0) + out_degrees.get(node, 0) for node in G.nodes()}
    max_degree = max(node_degrees.values()) if node_degrees else 1

    categories = get_concept_categories()
    category_colors = {
        'Materials': '#ff7f0e',
        'Biological Processes': '#2ca02c',
        'Research Methods': '#d62728',
        'Measurements': '#9467bd',
        'Other': '#8c564b'
    }

    # Add edges first (so they appear behind nodes)
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Add arrowhead
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='rgba(128,128,128,0.6)'
        )

    # Add nodes
    for node in G.nodes():
        x, y = pos[node]
        category = categorize_concept(node, categories)
        color = category_colors.get(category, '#8c564b')

        size = 20 + (node_degrees[node] / max_degree) * 30

        # Create hover text
        connections_in = [n for n in G.predecessors(node)]
        connections_out = [n for n in G.successors(node)]

        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Category: {category}<br>"
        hover_text += f"Incoming: {len(connections_in)}<br>"
        hover_text += f"Outgoing: {len(connections_out)}"

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node if len(node) < 20 else node[:17] + '...',
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False,
            name=node
        ))

    fig.update_layout(
        title="Interactive Concept Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )

    return fig


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

        with tab2:
            st.markdown("### 🌊 Concept Flow Visualization")
            st.markdown(
                "This diagram shows how concepts flow and connect to each other. Colors represent different categories.")

            sankey_fig = create_sankey_diagram(relationships)
            st.plotly_chart(sankey_fig, use_container_width=True)

            # Add legend
            st.markdown("""
            **Color Legend:**
            - 🟠 **Materials**: GNPs, coatings, concentrations
            - 🟢 **Biological Processes**: Differentiation, cellular changes
            - 🔴 **Research Methods**: Experimental techniques
            - 🟣 **Measurements**: Assessment and evaluation methods
            """)

        # with tab3:
        #     st.markdown("### 🕸️ Interactive Network View")
        #     st.markdown(
        #         "Explore the network of concepts. Node size indicates connectivity. Click and drag to interact!")
        #
        #     network_fig = create_hierarchical_tree(relationships)
        #     st.plotly_chart(network_fig, use_container_width=True)

        with tab4:
            st.markdown("### 📊 Research Analytics")

            # Generate insights
            total_relationships = len(relationships)
            all_concepts = set()
            for rel in relationships:
                all_concepts.add(rel['from'])
                all_concepts.add(rel['to'])

            unique_concepts = len(all_concepts)

            # Find most connected concepts
            concept_connections = defaultdict(int)
            for rel in relationships:
                concept_connections[rel['from']] += 1
                concept_connections[rel['to']] += 1

            most_connected = max(concept_connections.items(), key=lambda x: x[1]) if concept_connections else ("None",
                                                                                                               0)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Relationships", total_relationships)
            with col2:
                st.metric("Unique Concepts", unique_concepts)
            with col3:
                st.metric("Most Connected", most_connected[1])
            with col4:
                st.metric("Avg. Description Length", int(np.mean([len(rel['relationship']) for rel in relationships])))

            st.markdown(f"**Most Connected Concept:** {most_connected[0]}")

            # Create frequency charts
            col1, col2 = st.columns(2)

            with col1:
                # Concept frequency
                all_concepts_list = []
                for rel in relationships:
                    all_concepts_list.extend([rel['from'], rel['to']])

                concept_counts = Counter(all_concepts_list)
                df_freq = pd.DataFrame(list(concept_counts.items()), columns=['Concept', 'Frequency'])
                df_freq = df_freq.sort_values('Frequency', ascending=True).tail(10)

                fig_freq = px.bar(df_freq, x='Frequency', y='Concept', orientation='h',
                                  title='Most Frequently Mentioned Concepts',
                                  color='Frequency',
                                  color_continuous_scale='viridis')
                fig_freq.update_layout(height=400)
                st.plotly_chart(fig_freq, use_container_width=True)

            with col2:
                # Relationship strength
                df_rel = pd.DataFrame(relationships)
                df_rel['relationship_length'] = df_rel['relationship'].str.len()
                df_rel = df_rel.sort_values('relationship_length', ascending=False).head(8)
                df_rel['connection'] = df_rel['from'] + ' → ' + df_rel['to']

                fig_strength = px.bar(df_rel, x='relationship_length', y='connection',
                                      orientation='h',
                                      title='Strongest Relationships (by description)',
                                      color='relationship_length',
                                      color_continuous_scale='plasma')
                fig_strength.update_layout(height=400)
                st.plotly_chart(fig_strength, use_container_width=True)

            # Search and filter
            st.markdown("### 🔍 Search Relationships")
            search_term = st.text_input("Search for concepts or keywords:", placeholder="Enter search term...")

            if search_term:
                filtered_relationships = []
                for rel in relationships:
                    if (search_term.lower() in rel['from'].lower() or
                            search_term.lower() in rel['to'].lower() or
                            search_term.lower() in rel['relationship'].lower()):
                        filtered_relationships.append(rel)

                st.write(f"Found {len(filtered_relationships)} relationships matching '{search_term}':")

                for rel in filtered_relationships:
                    st.markdown(f"""
                    <div class="relationship-card">
                        <strong>{rel['from']} → {rel['to']}</strong><br>
                        {rel['relationship']}
                    </div>
                    """, unsafe_allow_html=True)

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