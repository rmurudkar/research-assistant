import streamlit as st
import os
import tempfile
import json

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI



load_dotenv()

# Initialize knowledge graph session state
if 'kg_nodes' not in st.session_state:
    st.session_state.kg_nodes = []
if 'kg_edges' not in st.session_state:
    st.session_state.kg_edges = []
if 'kg_processed_file' not in st.session_state:
    st.session_state.kg_processed_file = None

st.title("Knowledge Graph Visualization")

# File uploader
uploaded_file = st.file_uploader("Upload PDF for Knowledge Graph", type="pdf")


# Function to extract knowledge graph from document
def extract_knowledge_graph(documents, num_concepts=10):
    # Join some document text
    doc_text = "\n\n".join([doc.page_content for doc in documents[:20]])  # Use first 20 chunks

    # Create knowledge graph extraction prompt
    kg_prompt = PromptTemplate(
        template="""Extract a knowledge graph from the following text. 
        Identify the {num_concepts} most important concepts and the relationships between them.

        Format your response as a JSON object with two lists:
        1. "nodes": Each node should have an "id" and "label" property
        2. "edges": Each edge should have "source", "target", and "label" properties

        Text:
        {text}

        JSON Knowledge Graph:
        """,
        input_variables=["text", "num_concepts"]
    )

    # Create LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    kg_chain = LLMChain(llm=llm, prompt=kg_prompt)

    # Extract knowledge graph
    kg_result = kg_chain.run(text=doc_text, num_concepts=num_concepts)

    # Parse JSON result
    try:
        kg_data = json.loads(kg_result)
        nodes = kg_data.get("nodes", [])
        edges = kg_data.get("edges", [])
        return nodes, edges
    except Exception as e:
        st.error(f"Error parsing knowledge graph: {e}")
        return [], []


# Process the file when uploaded
if uploaded_file and st.session_state.kg_processed_file != uploaded_file.name:
    with st.spinner("Extracting knowledge graph from document..."):
        # Create a temporary file
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)

        # Save the uploaded file
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the PDF
        loader = PyPDFLoader(temp_filepath)
        documents = loader.load()

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Extract knowledge graph
        num_concepts = st.sidebar.slider("Number of concepts to extract", 5, 20, 10)
        nodes, edges = extract_knowledge_graph(texts, num_concepts)

        st.session_state.kg_nodes = nodes
        st.session_state.kg_edges = edges
        st.session_state.kg_processed_file = uploaded_file.name

        st.success(f"Extracted {len(nodes)} concepts and {len(edges)} relationships!")

# Display the knowledge graph
if st.session_state.kg_nodes and st.session_state.kg_edges:
    st.header("Document Knowledge Graph")

    # Add visualization options
    viz_type = st.radio("Visualization Type", ["Network Graph", "Force-Directed Graph"])

    if viz_type == "Network Graph":
        # Create a NetworkX-style graph visualization using Pyvis
        st.write("### Interactive Network Graph")

        # Generate the HTML for the network graph
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Knowledge Graph</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
            <style>
                #mynetwork {{
                    width: 100%;
                    height: 600px;
                    border: 1px solid lightgray;
                }}
            </style>
        </head>
        <body>
            <div id="mynetwork"></div>
            <script>
                const nodes = new vis.DataSet({json.dumps(st.session_state.kg_nodes)});
                const edges = new vis.DataSet({json.dumps(st.session_state.kg_edges)});

                const container = document.getElementById("mynetwork");
                const data = {{
                    nodes: nodes,
                    edges: edges
                }};
                const options = {{
                    nodes: {{
                        shape: "dot",
                        size: 16,
                        font: {{
                            size: 14
                        }},
                        borderWidth: 2
                    }},
                    edges: {{
                        width: 1,
                        font: {{
                            size: 12,
                            align: "middle"
                        }},
                        arrows: {{
                            to: {{ enabled: true, scaleFactor: 0.5 }}
                        }}
                    }},
                    physics: {{
                        stabilization: true,
                        barnesHut: {{
                            gravitationalConstant: -80,
                            springConstant: 0.001,
                            springLength: 200
                        }}
                    }}
                }};
                const network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """

        # Display the HTML using a component
        st.components.v1.html(html_code, height=650)

    elif viz_type == "Force-Directed Graph":
        # Create a D3.js force-directed graph
        st.write("### Force-Directed Graph")

        # Generate the HTML for the D3 graph
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Knowledge Graph</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
            <style>
                #graph {{
                    width: 100%;
                    height: 600px;
                    border: 1px solid lightgray;
                }}
                .node {{
                    fill: #69b3a2;
                    stroke: #fff;
                    stroke-width: 2px;
                }}
                .link {{
                    stroke: #999;
                    stroke-opacity: 0.6;
                }}
                .label {{
                    font-family: Arial;
                    font-size: 12px;
                    pointer-events: none;
                }}
            </style>
        </head>
        <body>
            <div id="graph"></div>
            <script>
                const width = document.getElementById("graph").clientWidth;
                const height = 600;

                const nodes = {json.dumps(st.session_state.kg_nodes)};
                const links = {json.dumps(st.session_state.kg_edges)}.map(d => ({{
                    source: d.source,
                    target: d.target,
                    label: d.label
                }}));

                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);

                const g = svg.append("g");

                // Create zoom behavior
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on("zoom", (event) => {{
                        g.attr("transform", event.transform);
                    }});

                svg.call(zoom);

                // Create a force simulation
                const simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).distance(150))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collision", d3.forceCollide().radius(60));

                // Add links
                const link = g.selectAll(".link")
                    .data(links)
                    .enter()
                    .append("line")
                    .attr("class", "link");

                // Add link labels
                const linkLabel = g.selectAll(".link-label")
                    .data(links)
                    .enter()
                    .append("text")
                    .attr("class", "label link-label")
                    .attr("text-anchor", "middle")
                    .text(d => d.label);

                // Add nodes
                const node = g.selectAll(".node")
                    .data(nodes)
                    .enter()
                    .append("circle")
                    .attr("class", "node")
                    .attr("r", 10)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));

                // Add node labels
                const nodeLabel = g.selectAll(".node-label")
                    .data(nodes)
                    .enter()
                    .append("text")
                    .attr("class", "label node-label")
                    .attr("text-anchor", "middle")
                    .attr("dy", -15)
                    .text(d => d.label);

                // Update positions on each tick
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);

                    linkLabel
                        .attr("x", d => (d.source.x + d.target.x) / 2)
                        .attr("y", d => (d.source.y + d.target.y) / 2);

                    node
                        .attr("cx", d => d.x)
                        .attr("cy", d => d.y);

                    nodeLabel
                        .attr("x", d => d.x)
                        .attr("y", d => d.y);
                }});

                // Drag functions
                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}

                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}

                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
            </script>
        </body>
        </html>
        """

        # Display the HTML using a component
        st.components.v1.html(html_code, height=650)

    # Display a table of concepts and relationships
    with st.expander("View Concepts and Relationships"):
        st.subheader("Key Concepts")
        st.table([{"ID": node["id"], "Concept": node["label"]} for node in st.session_state.kg_nodes])

        st.subheader("Relationships")
        st.table([{
            "Source": edge["source"],
            "Relationship": edge["label"],
            "Target": edge["target"]
        } for edge in st.session_state.kg_edges])

    # Add concept exploration
    with st.expander("Explore a Concept"):
        selected_concept = st.selectbox(
            "Select a concept to explore:",
            options=[node["label"] for node in st.session_state.kg_nodes]
        )

        if st.button("Explore Concept"):
            # Create a prompt to explore the selected concept
            explore_prompt = PromptTemplate(
                template="""Based on the document, provide an in-depth analysis of the concept: {concept}

                Include:
                1. A detailed explanation of the concept
                2. How it relates to other concepts in the document
                3. Its significance in the context of the document
                4. Any nuances or complexities associated with it

                Format your response in a clear, structured way.
                """,
                input_variables=["concept"]
            )

            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            explore_chain = LLMChain(llm=llm, prompt=explore_prompt)

            with st.spinner(f"Exploring the concept: {selected_concept}..."):
                exploration = explore_chain.run(concept=selected_concept)
                st.write(exploration)