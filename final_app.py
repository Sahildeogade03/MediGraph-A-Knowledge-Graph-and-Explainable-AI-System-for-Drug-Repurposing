import os
import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs
from rdkit.Chem import AllChem
import base64
from io import BytesIO
import pandas as pd
import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
import numpy as np
import networkx as nx
import pickle
from ast import literal_eval
import lime
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_option_menu import option_menu
import logging
import json
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j AuraDB connection
URI = "neo4j+s://b09f418b.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "Y9-UEMVWae0ISwDFsKFAtLczklxpSgOKZfKRyyI-mDY"
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Use CPU only
device = torch.device('cpu')

# Hardcoded list of unique and relevant diseases
relevant_diseases = [
    "Sickle Cell Disease",
    "Beta-Thalassemia",
    "Thalassemia",
    "Iron Overload",
    "Anemia",
    "Vaso-occlusive Crisis",
    "Acute Pain"
]

# Chatbot configuration
DB_FAISS_PATH = "D:/EDAI/model/vectorestore/db_faiss"
HF_TOKEN = "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"

# Enhanced HAN Model
class EnhancedHAN(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata):
        super(EnhancedHAN, self).__init__()
        self.han1 = HANConv(in_channels_dict, hidden_channels, metadata=metadata, heads=4, dropout=0.2)
        self.linear_drug = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_disease = torch.nn.Linear(hidden_channels, out_channels)
        self.residual = torch.nn.Linear(list(in_channels_dict.values())[0], out_channels)
        self.node_types = list(in_channels_dict.keys())

    def forward(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device
        x = self.han1(x_dict, edge_index_dict)
        x = {k: torch.relu(v) if v is not None and torch.is_tensor(v) else torch.zeros(x_dict[k].shape[0], self.han1.out_channels, device=device)
             for k, v in x.items()}
        x_in = torch.cat([x_dict[nt] for nt in x_dict], dim=0)
        res = torch.relu(self.residual(x_in))
        total_drug_nodes = x.get("Drug", torch.zeros(0, 256, device=device)).size(0)
        total_disease_nodes = x.get("Disease", torch.zeros(0, 256, device=device)).size(0)
        res_drug = res[:total_drug_nodes]
        res_disease = res[total_drug_nodes:total_drug_nodes + total_disease_nodes]
        drug_emb = self.linear_drug(x["Drug"]) + res_drug if "Drug" in x and x["Drug"].size(0) > 0 else torch.zeros(0, 128, device=device)
        disease_emb = self.linear_disease(x["Disease"]) + res_disease if "Disease" in x and x["Disease"].size(0) > 0 else torch.zeros(0, 128, device=device)
        return drug_emb, disease_emb, None

    def forward_with_dropout(self, x_dict, edge_index_dict, n_samples=10):
        preds = []
        for _ in range(n_samples):
            x = self.han1(x_dict, edge_index_dict)
            x = {k: torch.relu(v) if v is not None and torch.is_tensor(v) else torch.zeros(x_dict[k].shape[0], self.han1.out_channels, device=device)
                 for k, v in x.items()}
            x_in = torch.cat([x_dict[nt] for nt in x_dict], dim=0)
            res = torch.relu(self.residual(x_in))
            total_drug_nodes = x.get("Drug", torch.zeros(0, 256, device=device)).size(0)
            total_disease_nodes = x.get("Disease", torch.zeros(0, 256, device=device)).size(0)
            res_drug = res[:total_drug_nodes]
            res_disease = res[total_drug_nodes:total_drug_nodes + total_disease_nodes]
            drug_emb = self.linear_drug(x["Drug"]) + res_drug if "Drug" in x and x["Drug"].size(0) > 0 else torch.zeros(0, 128, device=device)
            disease_emb = self.linear_disease(x["Disease"]) + res_disease if "Disease" in x and x["Disease"].size(0) > 0 else torch.zeros(0, 128, device=device)
            preds.append((drug_emb, disease_emb))
        mean_drug = torch.stack([d for d, _ in preds]).mean(dim=0)
        mean_disease = torch.stack([dis for _, dis in preds]).mean(dim=0)
        std_drug = torch.stack([d for d, _ in preds]).std(dim=0)
        std_disease = torch.stack([dis for _, dis in preds]).std(dim=0)
        return mean_drug, mean_disease, std_drug, std_disease

# Scoring Functions
def compute_tanimoto_coefficient(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1) if smiles1 else None
        mol2 = Chem.MolFromSmiles(smiles2) if smiles2 else None
        if mol1 is None or mol2 is None:
            logger.warning(f"Invalid SMILES: mol1={smiles1}, mol2={smiles2}")
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        logger.warning(f"Error computing Tanimoto coefficient: {str(e)}")
        return 0.0

# Subgraph Visualization Function
def plot_subgraph(drug_id, disease_id, nodes_df, driver, top_k=10):
    drug_name = nodes_df[nodes_df["id"] == drug_id]["display_name"].iloc[0] if drug_id in nodes_df["id"].values else drug_id
    disease_name = nodes_df[nodes_df["id"] == disease_id]["display_name"].iloc[0] if disease_id in nodes_df["id"].values else disease_id
    try:
        with driver.session() as session:
            query = """
            MATCH p = (d:Drug {id: $drug_id})-[r:TREATS|ASSOCIATED_WITH|HAS_SYMPTOM|INVOLVES|INTERACTS_WITH|BINDS|CAUSES|PALLIATES|TARGETS*1..2]-(dis:Disease {id: $disease_id})
            RETURN p
            UNION
            MATCH (d:Drug {id: $drug_id})-[r:TREATS|ASSOCIATED_WITH|HAS_SYMPTOM|INVOLVES|INTERACTS_WITH|BINDS|CAUSES|PALLIATES|TARGETS]-(n)
            WHERE n:Disease OR n:Symptom OR n:Gene OR n:Pathway
            RETURN d, r, n
            UNION
            MATCH (dis:Disease {id: $disease_id})-[r:TREATS|ASSOCIATED_WITH|HAS_SYMPTOM|INVOLVES|INTERACTS_WITH|BINDS|CAUSES|PALLIATES|TARGETS]-(n)
            WHERE n:Drug OR n:Symptom OR n:Gene OR n:Pathway
            RETURN dis, r, n
            """
            result = session.run(query, drug_id=drug_id, disease_id=disease_id)
            paths = list(result)
            logger.info(f"Subgraph query for drug {drug_id} and disease {disease_id} returned {len(paths)} paths")
            G = nx.DiGraph()
            for record in paths:
                path = record.get("p")
                if path:
                    for node in path.nodes:
                        node_id = str(node.element_id)
                        props = dict(node)
                        node_type = list(node.labels)[0] if node.labels else "Unknown"
                        node_name = nodes_df[nodes_df["id"] == props.get("id", node_id)]["display_name"].iloc[0] if props.get("id", node_id) in nodes_df["id"].values else props.get("name", node_id)
                        G.add_node(node_id, label=node_name, type=node_type, props=props)
                    for rel in path.relationships:
                        G.add_edge(str(rel.start_node.element_id), str(rel.end_node.element_id), type=rel.type)
                else:
                    node = record.get("d") or record.get("dis")
                    related_node = record.get("n")
                    rels = record.get("r", [])
                    node_id = str(node.element_id)
                    related_node_id = str(related_node.element_id)
                    node_type = list(node.labels)[0] if node.labels else "Unknown"
                    related_node_type = list(related_node.labels)[0] if related_node.labels else "Unknown"
                    node_props = dict(node)
                    related_props = dict(related_node)
                    node_name = nodes_df[nodes_df["id"] == node_props.get("id", node_id)]["display_name"].iloc[0] if node_props.get("id", node_id) in nodes_df["id"].values else node_props.get("name", node_id)
                    related_name = nodes_df[nodes_df["id"] == related_props.get("id", related_node_id)]["display_name"].iloc[0] if related_props.get("id", related_node_id) in nodes_df["id"].values else related_props.get("name", related_node_id)
                    if node_id not in G:
                        G.add_node(node_id, label=node_name, type=node_type, props=node_props)
                    if related_node_id not in G:
                        G.add_node(related_node_id, label=related_name, type=related_node_type, props=related_props)
                    for rel in rels:
                        G.add_edge(str(rel.start_node.element_id), str(rel.end_node.element_id), type=rel.type)
            if G.number_of_nodes() == 0:
                logger.warning(f"No subgraph data from Neo4j for drug {drug_id} and disease {disease_id}")
                raise Exception("No subgraph data retrieved from Neo4j")
            if G.number_of_nodes() > top_k:
                drug_disease_nodes = {drug_id, disease_id}
                connected_nodes = set()
                for n in G.nodes():
                    if n in drug_disease_nodes:
                        continue
                    if G.has_edge(drug_id, n) or G.has_edge(n, drug_id) or G.has_edge(disease_id, n) or G.has_edge(n, disease_id):
                        connected_nodes.add(n)
                keep_nodes = drug_disease_nodes | connected_nodes
                if len(keep_nodes) > top_k:
                    keep_nodes = drug_disease_nodes | set(list(connected_nodes)[:top_k - 2])
                G = G.subgraph(keep_nodes).copy()
            pos = nx.spring_layout(G, seed=42, k=0.5)
            color_map = {
                "Anatomy": "#FF0000", "BiologicalProcess": "#0000FF", "CellularComponent": "#008000",
                "Disease": "#FFFF00", "Drug": "#800080", "Gene": "#FFA500", "MolecularFunction": "#FFC1CC",
                "Pathway": "#A52A2A", "PharmacologicClass": "#00FFFF", "SideEffect": "#FF00FF",
                "Symptom": "#00FF00", "Target": "#008080", "Trial": "#808080", "Unknown": "#777777"
            }
            edge_x = []
            edge_y = []
            edge_text = []
            for src, tgt, data in G.edges(data=True):
                x0, y0 = pos[src]
                x1, y1 = pos[tgt]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(data["type"])
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='text',
                text=edge_text
            )
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_label = G.nodes[node]["label"]
                node_type = G.nodes[node]["type"]
                node_text.append(f"{node_label}<br>Type: {node_type}")
                node_colors.append(color_map.get(node_type, "#777777"))
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                text=[n for n in [G.nodes[n]["label"] for n in G.nodes()]],
                textposition="top center",
                marker=dict(size=20, color=node_colors),
                hoverinfo='text',
                hovertext=node_text
            )
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Subgraph for {drug_name} and {disease_name}",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            return fig
    except Exception as e:
        logger.warning(f"Failed to generate Neo4j subgraph: {str(e)}. Falling back to hardcoded Drug-TREATS-Disease graph with additional nodes.")
        G = nx.DiGraph()
        G.add_node(drug_id, label=drug_name, type="Drug")
        G.add_node(disease_id, label=disease_name, type="Disease")
        G.add_edge(drug_id, disease_id, type="TREATS")
        related_nodes = []
        for _, row in rels_df.iterrows():
            if row["source"] == drug_id or row["target"] == drug_id or row["source"] == disease_id or row["target"] == disease_id:
                related_id = row["source"] if row["source"] not in [drug_id, disease_id] else row["target"]
                related_type = nodes_df[nodes_df["id"] == related_id]["label"].iloc[0] if related_id in nodes_df["id"].values else "Unknown"
                if related_type in ["Symptom", "Gene", "Pathway"] and related_id not in G.nodes():
                    related_name = nodes_df[nodes_df["id"] == related_id]["display_name"].iloc[0] if related_id in nodes_df["id"].values else related_id
                    related_nodes.append((related_id, related_name, related_type, row["type"], row["source"], row["target"]))
        related_nodes.sort(key=lambda x: x[3] in ["TREATS", "ASSOCIATED_WITH"], reverse=True)
        for related_id, related_name, related_type, rel_type, src, tgt in related_nodes[:top_k - 2]:
            G.add_node(related_id, label=related_name, type=related_type)
            if src == related_id:
                G.add_edge(related_id, tgt, type=rel_type)
            else:
                G.add_edge(src, related_id, type=rel_type)
        pos = nx.spring_layout(G, seed=42, k=0.5)
        color_map = {
            "Drug": "#00FF00",
            "Disease": "#ADD8E6",
            "Symptom": "#00FF00", "Gene": "#FFA500", "Pathway": "#A52A2A", "Unknown": "#777777"
        }
        edge_x = []
        edge_y = []
        edge_text = []
        for src, tgt, data in G.edges(data=True):
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(data["type"])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='text',
            text=edge_text
        )
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_label = G.nodes[node]["label"]
            node_type = G.nodes[node]["type"]
            node_text.append(f"{node_label}<br>Type: {node_type}")
            node_colors.append(color_map.get(node_type, "#777777"))
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=[G.nodes[n]["label"] for n in G.nodes()],
            textposition="top center",
            marker=dict(size=20, color=node_colors),
            hoverinfo='text',
            hovertext=node_text
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Subgraph for {drug_name} and {disease_name}",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig

# Chatbot Functions
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": HF_TOKEN}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )
    return llm

def chatbot_main():
    st.header("MediBot Chatbot 💬")
    st.write("Ask questions about drug repurposing or connecting medicines with diseases.")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("Pass your prompt here:")
    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message("user").markdown(prompt)
        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            Answer as you are a Scientist trying to repurpose a drug or connect different medicines with different diseases.
            Context: {context}
            Question: {question}
            Start the answer directly. No small talk please.
        """
        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGING_FACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response = qa_chain.invoke({'query': prompt})
            result = response['result']
            source_documents = response['source_documents']
            formatted_sources = "\n".join([
                f"- **Page {doc.metadata.get('page', 'N/A')}** from *{doc.metadata.get('source', 'Unknown Source')}*"
                for doc in source_documents
            ])
            result_to_show = f"""
            **Response:**
            {result}

            **Source Documents:**
            {formatted_sources if formatted_sources else 'No relevant documents found.'}
            """
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            st.chat_message("assistant").markdown(result_to_show)
        except Exception as e:
            st.error(f"Error: {e}")

# Load data and model
@st.cache_resource
def load_model_and_data():
    try:
        nodes_df = pd.read_csv("model/nodes_df_retrained.csv")
        rels_df = pd.read_csv("model/rels_df_retrained.csv")
        def sanitize_dataframe(df):
            for col in df.columns:
                if col == "props":
                    df[col] = df[col].apply(
                        lambda x: str({"Value": str(x) if pd.notna(x) else "0"}) if pd.notna(x) and not x.strip().startswith("{") else str(x) if pd.notna(x) else '{"Value": "0"}'
                    )
                elif df[col].dtype == 'object':
                    df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    df[col] = df[col].astype(float).fillna(0.0).astype(str)
            return df
        nodes_df = sanitize_dataframe(nodes_df)
        rels_df = sanitize_dataframe(rels_df)
        def safe_get_name(props_str, index):
            try:
                props = literal_eval(props_str)
                if not isinstance(props, dict):
                    logger.warning(f"Row {index}: props is not a dict, got {type(props)} with value {props_str}")
                    return props_str
                return props.get("name", props_str)
            except (ValueError, SyntaxError) as e:
                logger.error(f"Row {index}: Failed to parse props {props_str} due to {e}")
                return props_str
        nodes_df["display_name"] = [safe_get_name(props, i) for i, props in enumerate(nodes_df["props"])]
        checkpoint = torch.load("model/han_drug_repurposing_retrained.pth", map_location=device, weights_only=True)
        metadata = checkpoint['metadata']
        in_channels_dict = checkpoint['in_channels_dict']
        data = HeteroData()
        with open('model/type_to_idx_retrained.pkl', 'rb') as f:
            node_type_to_idx = pickle.load(f)
        total_nodes = 0
        for node_type in nodes_df["label"].unique():
            type_nodes = nodes_df[nodes_df["label"] == node_type]
            if not type_nodes.empty:
                features = torch.stack([extract_features(literal_eval(row["props"])) for _, row in type_nodes.iterrows()])
                data[node_type].x = features.to(device) if features.shape[0] > 0 else torch.zeros((len(type_nodes), 516)).to(device)
                total_nodes += len(type_nodes)
            else:
                data[node_type].x = torch.zeros((0, 516)).to(device)
        edge_dict = {}
        for _, row in rels_df.iterrows():
            src_type = nodes_df[nodes_df["id"] == row["source"]]["label"].iloc[0]
            tgt_type = nodes_df[nodes_df["id"] == row["target"]]["label"].iloc[0]
            edge_type = (src_type, row["type"], tgt_type)
            src_idx = node_type_to_idx.get(src_type, {}).get(row["source"])
            tgt_idx = node_type_to_idx.get(tgt_type, {}).get(row["target"])
            if src_idx is not None and tgt_idx is not None:
                if edge_type not in edge_dict:
                    edge_dict[edge_type] = [[], []]
                edge_dict[edge_type][0].append(src_idx)
                edge_dict[edge_type][1].append(tgt_idx)
        for edge_type, (src_indices, tgt_indices) in edge_dict.items():
            if src_indices and tgt_indices:
                data[edge_type].edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long).to(device)
            else:
                data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
        data = data.to(device)
        model = EnhancedHAN(in_channels_dict, hidden_channels=256, out_channels=128, metadata=metadata).to(device)
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        with torch.no_grad():
            all_drug_emb, all_disease_emb, _ = model(data.x_dict, data.edge_index_dict)
            all_std_drug = torch.std(all_drug_emb, dim=0, keepdim=True).repeat(all_drug_emb.size(0), 1)
            all_std_disease = torch.std(all_disease_emb, dim=0, keepdim=True).repeat(all_disease_emb.size(0), 1)
        return model, data, nodes_df, rels_df, node_type_to_idx, all_drug_emb, all_disease_emb, all_std_drug, all_std_disease
    except Exception as e:
        st.error(f"Error loading model and data: {str(e)}")
        logger.error(f"Load error: {str(e)}")
        return None, None, None, None, None, None, None, None, None

def extract_features(props_dict):
    try:
        if "smiles" in props_dict and props_dict["smiles"]:
            mol = Chem.MolFromSmiles(props_dict["smiles"])
            fp_tensor = torch.zeros(512, dtype=torch.float32) if mol is None else torch.tensor(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)), dtype=torch.float32)
        else:
            fp_tensor = torch.zeros(512, dtype=torch.float32)
        value = float(props_dict.get("Value", "0").replace("nan", "0").replace("None", "0"))
        extra_features = [float(props_dict.get(key, 0.0)) for key in ["trial_count", "molecular_weight", "degree"]] + [value]
        extra_tensor = torch.tensor(extra_features, dtype=torch.float32)
        return torch.cat([fp_tensor, extra_tensor], dim=0).to(device)
    except Exception as e:
        st.warning(f"Error extracting features: {str(e)}")
        return torch.zeros(516, dtype=torch.float32).to(device)

def predict_drug_disease(drug_id, disease_id, node_type_to_idx, nodes_df, all_drug_emb, all_disease_emb, driver):
    try:
        drug_idx = node_type_to_idx["Drug"].get(drug_id)
        disease_idx = node_type_to_idx["Disease"].get(disease_id)
        if drug_idx is None or disease_idx is None:
            logger.error(f"Drug ID {drug_id} or Disease ID {disease_id} not found.")
            return None, None, None, "Drug or Disease ID not found."
        drug_emb = all_drug_emb[drug_idx]
        disease_emb = all_disease_emb[disease_idx]
        han_score = torch.sigmoid((drug_emb * disease_emb).sum()).item()
        drug_props = literal_eval(nodes_df[nodes_df["id"] == drug_id]["props"].iloc[0])
        drug_smiles = drug_props.get("smiles", "")
        reference_smiles = ""
        with driver.session() as session:
            reference_query = """
            MATCH (dis:Disease {id: $disease_id})<-[:TREATS]-(d:Drug)
            RETURN d.smiles AS smiles LIMIT 1
            """
            result = session.run(reference_query, disease_id=disease_id).single()
            if result and result["smiles"]:
                reference_smiles = result["smiles"]
            if not reference_smiles:
                reference_smiles = "CC"
        tc = compute_tanimoto_coefficient(drug_smiles, reference_smiles) if drug_smiles and reference_smiles else 0.0
        logger.info(f"Prediction for drug {drug_id} and disease {disease_id}: TC={tc}, HAN={han_score}")
        return tc, han_score, None, None
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, None, None, f"Prediction error: {str(e)}"

def visualize_subgraph(drug_id, disease_id, nodes_df, rels_df, driver):
    try:
        with driver.session() as session:
            query = """
            MATCH p = (d:Drug {id: $drug_id})-[r:TREATS|ASSOCIATED_WITH|HAS_SYMPTOM|INVOLVES|INTERACTS_WITH|BINDS|CAUSES|PALLIATES|TARGETS*1..5]-(dis:Disease {id: $disease_id})
            RETURN p
            """
            result = session.run(query, drug_id=drug_id, disease_id=disease_id)
            paths = list(result)
            logger.info(f"Primary query for drug {drug_id} and disease {disease_id} returned {len(paths)} paths")
            G = nx.MultiDiGraph()
            if paths:
                for record in paths:
                    path = record["p"]
                    for node in path.nodes:
                        node_id = str(node.element_id)
                        props = dict(node)
                        node_type = list(node.labels)[0] if node.labels else "Unknown"
                        if node_id not in G:
                            G.add_node(node_id, label=node_type, props=props)
                    for rel in path.relationships:
                        G.add_edge(str(rel.start_node.element_id), str(rel.end_node.element_id), type=rel.type)
            else:
                logger.warning(f"No direct paths found for drug {drug_id} and disease {disease_id}. Using fallback query.")
                fallback_query = """
                MATCH (d:Drug {id: $drug_id})-[r:*0..2]-(n)
                WHERE n:Disease OR n:Symptom OR n:Gene OR n:Pathway
                RETURN d, r, n
                UNION
                MATCH (dis:Disease {id: $disease_id})-[r:*0..2]-(n)
                WHERE n:Drug OR n:Symptom OR n:Gene OR n:Pathway
                RETURN dis, r, n
                """
                result = session.run(fallback_query, drug_id=drug_id, disease_id=disease_id)
                for record in result:
                    node = record[0] if "d" in record else record["dis"]
                    related_node = record["n"]
                    rels = record["r"] if record["r"] else []
                    node_id = str(node.element_id)
                    related_node_id = str(related_node.element_id)
                    node_type = list(node.labels)[0] if node.labels else "Unknown"
                    related_node_type = list(related_node.labels)[0] if related_node.labels else "Unknown"
                    if node_id not in G:
                        G.add_node(node_id, label=node_type, props=dict(node))
                    if related_node_id not in G:
                        G.add_node(related_node_id, label=related_node_type, props=dict(related_node))
                    for rel in rels:
                        G.add_edge(str(rel.start_node.element_id), str(rel.end_node.element_id), type=rel.type)
                logger.info(f"Fallback query returned {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            if G.number_of_nodes() == 0:
                logger.warning(f"No nodes found for drug {drug_id} and disease {disease_id} even with fallback query")
                st.warning("No subgraph data available despite extended query. Check database connectivity or graph structure.")
                return [], [], Config(width="100%", height=400, directed=True, physics=True)
            nodes = [
                Node(
                    id=n,
                    label=G.nodes[n]["props"].get("name", n),
                    title="\n".join(f"{k}: {v}" for k, v in G.nodes[n]["props"].items()),
                    color={
                        "Anatomy": "#FF0000", "BiologicalProcess": "#0000FF", "CellularComponent": "#008000",
                        "Disease": "#FFFF00", "Drug": "#800080", "Gene": "#FFA500", "MolecularFunction": "#FFC1CC",
                        "Pathway": "#A52A2A", "PharmacologicClass": "#00FFFF", "SideEffect": "#FF00FF",
                        "Symptom": "#00FF00", "Target": "#008080", "Trial": "#808080"
                    }.get(G.nodes[n]["label"], "#777777")
                ) for n in G.nodes()
            ]
            edges = [Edge(source=src, target=tgt, label=data["type"], arrows="to") for src, tgt, data in G.edges(data=True)]
            logger.info(f"Subgraph for drug {drug_id} and disease {disease_id}: {len(nodes)} nodes, {len(edges)} edges")
            return nodes, edges, Config(width="100%", height=400, directed=True, physics=True)
    except Exception as e:
        logger.error(f"Error visualizing subgraph: {str(e)}")
        st.error(f"Subgraph visualization failed: {str(e)}")
        return [], [], Config(width="100%", height=400, directed=True, physics=True)

def explain_prediction(drug_id, disease_id, nodes_df, rels_df):
    try:
        G = nx.MultiDiGraph()
        for _, row in nodes_df.iterrows():
            G.add_node(row["id"], label=row["label"], props=literal_eval(row["props"]))
        for _, row in rels_df.iterrows():
            G.add_edge(row["source"], row["target"], type=row["type"])
        path = nx.shortest_path(G, source=drug_id, target=disease_id) if nx.has_path(G, drug_id, disease_id) else [drug_id, disease_id]
        subgraph = nx.subgraph(G, path)
        nodes = [
            Node(
                id=str(n),
                label=G.nodes[n]["props"].get("name", n),
                title="\n".join(f"{k}: {v}" for k, v in G.nodes[n]["props"].items()),
                color={
                    "Anatomy": "#FF0000", "BiologicalProcess": "#0000FF", "CellularComponent": "#008000",
                    "Disease": "#FFFF00", "Drug": "#800080", "Gene": "#FFA500", "MolecularFunction": "#FFC1CC",
                    "Pathway": "#A52A2A", "PharmacologicClass": "#00FFFF", "SideEffect": "#FF00FF",
                    "Symptom": "#00FF00", "Target": "#008080", "Trial": "#808080"
                }.get(G.nodes[n]["label"], "#777777")
            ) for n in subgraph.nodes()
        ]
        edges = [Edge(source=str(src), target=str(tgt), label=data["type"], arrows="to") for src, tgt, data in subgraph.edges(data=True)]
        return nodes, edges, Config(width="100%", height=500, directed=True, physics=True, hierarchical=False)
    except Exception as e:
        logger.error(f"Error in explain_prediction: {str(e)}")
        return [], [], Config(width="100%", height=500, directed=True, physics=True, hierarchical=False)

def fetch_graph_data(cypher_query):
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            nodes = []
            edges = []
            node_ids = set()
            color_map = {
                "Anatomy": "#FF0000", "BiologicalProcess": "#0000FF", "CellularComponent": "#008000",
                "Disease": "#FFFF00", "Drug": "#800080", "Gene": "#FFA500", "MolecularFunction": "#FFC1CC",
                "Pathway": "#A52A2A", "PharmacologicClass": "#00FFFF", "SideEffect": "#FF00FF",
                "Symptom": "#00FF00", "Target": "#008080", "Trial": "#808080"
            }
            rel_styles = {
                "ASSOCIATED_WITH": "solid", "ASSOCIATES_WITH": "dashed", "BINDS": "solid",
                "CAUSES": "solid", "COVARIES_WITH": "dashed", "DOWNREGULATES": "solid",
                "EXPRESSES": "solid", "HAS_SYMPTOM": "solid", "INCLUDES": "dashed",
                "INTERACTS_WITH": "solid", "INVOLVES": "solid", "PALLIATES": "dashed",
                "PARTICIPATES_IN": "solid", "REGULATES": "solid", "RESEMBLES": "dashed",
                "TARGETS": "dashed", "TREATS": "dashed", "UPREGULATES": "solid"
            }
            for record in result:
                path = record["p"]
                if path:
                    for node in path.nodes:
                        node_id = str(node.element_id)
                        if node_id not in node_ids:
                            props = dict(node)
                            node_type = list(node.labels)[0] if node.labels else "Unknown"
                            nodes.append(Node(
                                id=node_id,
                                label=props.get("name", node_id),
                                title="\n".join(f"{k}: {v}" for k, v in props.items()),
                                color=color_map.get(node_type, "#777777")
                            ))
                            node_ids.add(node_id)
                    for rel in path.relationships:
                        edges.append(Edge(
                            source=str(rel.start_node.element_id),
                            target=str(rel.end_node.element_id),
                            label=rel.type,
                            dashes=rel_styles.get(rel.type, False),
                            arrows="to"
                        ))
            logger.info(f"Graph query returned {len(nodes)} nodes and {len(edges)} edges")
            return nodes, edges, Config(width="75%", height=600, directed=True, physics=True, hierarchical=False)
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            st.error(f"Query failed: {str(e)}")
            return [], [], None

def find_similar_drugs(drug_id, nodes_df, all_drug_emb, node_type_to_idx, top_k=3):
    try:
        drug_props = literal_eval(nodes_df[nodes_df["id"] == drug_id]["props"].iloc[0])
        drug_smiles = drug_props.get("smiles", "")
        if not drug_smiles:
            logger.warning(f"No SMILES for drug ID {drug_id}")
            return []
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES for drug ID {drug_id}")
            return []
        drug_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        drug_idx = node_type_to_idx["Drug"].get(drug_id)
        if drug_idx is None:
            logger.warning(f"Drug index not found for {drug_id}")
            return []
        ref_drug_emb = all_drug_emb[drug_idx]
        similarities = []
        drug_ids = nodes_df[nodes_df["label"] == "Drug"]["id"].tolist()
        for d_id in drug_ids:
            if d_id == drug_id:
                continue
            d_props = literal_eval(nodes_df[nodes_df["id"] == d_id]["props"].iloc[0])
            d_smiles = d_props.get("smiles", "")
            if d_smiles:
                d_mol = Chem.MolFromSmiles(d_smiles)
                if d_mol:
                    d_fp = AllChem.GetMorganFingerprintAsBitVect(d_mol, 2, nBits=2048)
                    sim = DataStructs.TanimotoSimilarity(d_fp, drug_fp)
                    d_idx = node_type_to_idx["Drug"].get(d_id)
                    if d_idx is not None:
                        d_emb = all_drug_emb[d_idx]
                        han_score = torch.cosine_similarity(ref_drug_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
                        similarities.append((d_id, sim, han_score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_drugs = similarities[:top_k]
        logger.info(f"Found {len(similar_drugs)} similar drugs for {drug_id}")
        return similar_drugs
    except Exception as e:
        st.error(f"Error finding similar drugs: {str(e)}")
        logger.error(f"Exception in find_similar_drugs: {str(e)}")
        return []

def fetch_chembl_data(drug_name):
    try:
        search_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
        headers = {"Accept": "application/json"}
        params = {"q": drug_name.lower().replace(" ", "+"), "limit": 1}
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"ChEMBL search response for {drug_name}: {data}")
        if data.get("molecules") and data["molecules"]:
            molecule = data["molecules"][0]
            chembl_id = molecule.get("molecule_chembl_id", "Not available")
            molecule_structures = molecule.get("molecule_structures", {})
            smiles = molecule_structures.get("canonical_smiles", "Not available") if molecule_structures else "Not available"
            synonyms = molecule.get("molecule_synonyms", [{}])[0].get("molecule_synonym", "Not available") if molecule.get("molecule_synonyms") else "Not available"
            return smiles, synonyms, chembl_id
        else:
            known_chembl_ids = {
                "PRASUGREL": "CHEMBL1201772",
                "FENTANYL CITRATE": "CHEMBL638",
                "DEFEROXAMINE": "CHEMBL465234",
                "DEFERIPRONE": "CHEMBL1201129",
                "DECITABINE": "CHEMBL1201129",
                "ACETAMINOPHEN": "CHEMBL112",
                "DEFERASIROX": "CHEMBL550348",
                "CRIZANLIZUMAB": "CHEMBL4297918",
                "LUSPATERCEPT": "CHEMBL3989899",
                "HYDROXYUREA": "CHEMBL1200576",
                "NITRIC OXIDE": "CHEMBL1200689"
            }
            chembl_id = known_chembl_ids.get(drug_name.upper(), "Not available")
            if chembl_id != "Not available":
                molecule_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}"
                response = requests.get(molecule_url, headers=headers, timeout=10)
                response.raise_for_status()
                molecule = response.json()
                molecule_structures = molecule.get("molecule_structures", {})
                smiles = molecule_structures.get("canonical_smiles", "Not available") if molecule_structures else "Not available"
                synonyms = molecule.get("molecule_synonyms", [{}])[0].get("molecule_synonym", "Not available") if molecule.get("molecule_synonyms") else "Not available"
                return smiles, synonyms, chembl_id
            return "Not available", "Not available", "Not available"
    except requests.exceptions.RequestException as e:
        logger.error(f"ChEMBL API error for {drug_name}: {str(e)}")
        return "Not available", "Not available", "Not available"

def get_molecular_image(smiles):
    try:
        if smiles == "Not available" or not smiles:
            logger.info(f"SMILES not available for image generation")
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to create molecule from SMILES: {smiles}")
            return None
        img = Draw.MolToImage(mol, size=(200, 200))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"Successfully generated image for SMILES: {smiles}")
        return img_str
    except Exception as e:
        logger.error(f"Error generating image for SMILES {smiles}: {str(e)}")
        return None

# Streamlit App
st.set_page_config(layout="wide", page_icon="🔬", page_title="MediGraph")
st.title("MediGraph Drug Repurposing Hub 💊")

with st.sidebar:
    selected = option_menu("MediGraph", ["KG Explorer", "Drug Repurposing Predictor", "Similar Drugs Explorer", "Retrieval Assistant"],icons=["graph-up", "activity", "search", "chat"], menu_icon="cast", default_index=0)

    st.header("Legends")
    with st.expander("Node Types"):
        for node_type, color in {
            "Anatomy": "#FF0000", "BiologicalProcess": "#0000FF", "CellularComponent": "#008000",
            "Disease": "#FFFF00", "Drug": "#800080", "Gene": "#FFA500", "MolecularFunction": "#FFC1CC",
            "Pathway": "#A52A2A", "PharmacologicClass": "#00FFFF", "SideEffect": "#FF00FF",
            "Symptom": "#00FF00", "Target": "#008080", "Trial": "#808080"
        }.items():
            st.write(f"<span style='display:inline-block;width:12px;height:12px;background-color:{color};border-radius:50%;margin-right:5px'></span> {node_type}", unsafe_allow_html=True)
    with st.expander("Relationship Types"):
        for rel_type, style in {
            "ASSOCIATED_WITH": "solid", "ASSOCIATES_WITH": "dashed", "BINDS": "solid",
            "CAUSES": "solid", "COVARIES_WITH": "dashed", "DOWNREGULATES": "solid",
            "EXPRESSES": "solid", "HAS_SYMPTOM": "solid", "INCLUDES": "dashed",
            "INTERACTS_WITH": "solid", "INVOLVES": "solid", "PALLIATES": "dashed",
            "PARTICIPATES_IN": "solid", "REGULATES": "solid", "RESEMBLES": "dashed",
            "TARGETS": "dashed", "TREATS": "dashed", "UPREGULATES": "solid"
        }.items():
            st.write(f"<span style='display:inline-block;width:20px;height:2px;border-top:2px {style} black;margin-right:5px'></span> {rel_type}", unsafe_allow_html=True)

model, data, nodes_df, rels_df, node_type_to_idx, all_drug_emb, all_disease_emb, all_std_drug, all_std_disease = load_model_and_data()

if model is None:
    st.error("Failed to load model and data. Please check the error messages and data files.")
elif selected == "KG Explorer":
    st.header("Interactive Knowledge Graph Explorer🌏")
    with st.container():
        st.markdown("<div style='border: 2px solid #ccc; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Drugs → Treats → Diseases"):
                st.session_state.query_result = fetch_graph_data("MATCH p=()-[:TREATS]->() RETURN p")
        with col2:
            if st.button("🧬 Gene → Associated with → Disease"):
                st.session_state.query_result = fetch_graph_data("MATCH p=()-[:ASSOCIATED_WITH]->() RETURN p")
        col3, col4 = st.columns(2)
        with col3:
            if st.button("💊 Drug → Similar To → Drug"):
                st.session_state.query_result = fetch_graph_data("MATCH p=()-[:RESEMBLES]->() RETURN p")
        with col4:
            if st.button("🏥 Clinical Trials → Involves → Drug"):
                st.session_state.query_result = fetch_graph_data("MATCH p=()-[:INVOLVES]->() RETURN p")
        st.markdown("</div>", unsafe_allow_html=True)
    if 'query_result' in st.session_state and st.session_state.query_result:
        nodes, edges, config = st.session_state.query_result
        st.write(f"Graph with {len(nodes)} nodes and {len(edges)} edges:")
        agraph(nodes=nodes, edges=edges, config=config)
        st.session_state.query_result = None
    cypher_query = st.text_area("Enter Cypher Query", "MATCH p=()-[:TREATS]->() RETURN p",
                                help="Enter a valid Cypher query. Example: MATCH (n:Drug)-[r]->(m:Disease) RETURN n, r, m")
    if st.button("Run Query"):
        nodes, edges, config = fetch_graph_data(cypher_query)
        if nodes or edges:
            st.write(f"Graph with {len(nodes)} nodes and {len(edges)} edges:")
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("No results or query failed.")
elif selected == "Drug Repurposing Predictor":
    st.header("Drug Repurposing Predictor with Explainable AI 🤖")
    col1, col2 = st.columns(2)
    with col1:
        selected_disease = st.selectbox("Select a Disease", relevant_diseases, key="disease_selector")
        disease_id = nodes_df[nodes_df["display_name"] == selected_disease]["id"].iloc[0] if not nodes_df[nodes_df["display_name"] == selected_disease].empty else None
        if disease_id is None:
            st.error(f"No disease found with name {selected_disease}")
    if st.button("Predict Top Drugs") and disease_id:
        with st.spinner("Predicting top drugs..."):
            top_drugs = []
            drug_ids = nodes_df[nodes_df["label"] == "Drug"]["id"].tolist()
            for drug_id in drug_ids:
                tc, han_score, _, error = predict_drug_disease(drug_id, disease_id, node_type_to_idx, nodes_df, all_drug_emb, all_disease_emb, driver)
                if error is None and tc is not None and han_score is not None:
                    top_drugs.append((drug_id, tc, han_score))
            top_drugs.sort(key=lambda x: x[1], reverse=True)
            top_drugs = top_drugs[:6]
            if top_drugs:
                scores_data = []
                info_data = []
                drug_ids = [drug_id for drug_id, _, _ in top_drugs]
                for drug_id, tc, han_score in top_drugs:
                    drug_name = nodes_df[nodes_df["id"] == drug_id]["display_name"].iloc[0]
                    smiles, synonyms, chembl_id = fetch_chembl_data(drug_name)
                    image = get_molecular_image(smiles) if smiles != "Not available" else None
                    scores_data.append({
                        "Drug": drug_name,
                        "Tanimoto Coefficient": round(tc, 4),
                        "HAN Model Score": round(han_score, 4)
                    })
                    info_data.append({
                        "Drug": drug_name,
                        "SMILES": smiles,
                        "Synonyms": synonyms,
                        "ChEMBL ID": chembl_id,
                        "Image": image
                    })
                scores_df = pd.DataFrame(scores_data)
                info_df = pd.DataFrame(info_data)
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.subheader("Drug Scores")
                    st.dataframe(scores_df[["Drug", "Tanimoto Coefficient", "HAN Model Score"]])
                    st.subheader("Drug Information")
                    st.dataframe(info_df[["Drug", "SMILES", "Synonyms", "ChEMBL ID"]])
                    csv_scores = scores_df.to_csv(index=False)
                    csv_info = info_df.drop(columns=["Image"]).to_csv(index=False)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Scores as CSV",
                            data=csv_scores,
                            file_name=f"drug_scores_{selected_disease}.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            label="Download Drug Info as CSV",
                            data=csv_info,
                            file_name=f"drug_info_{selected_disease}.csv",
                            mime="text/csv"
                        )
                with col_right:
                    st.subheader("Subgraph Visualization")
                    top_drug_id = drug_ids[0]
                    subgraph_fig = plot_subgraph(top_drug_id, disease_id, nodes_df, driver, top_k=10)
                    if subgraph_fig:
                        st.plotly_chart(subgraph_fig, use_container_width=True)
                    else:
                        st.write("No subgraph available.")
                st.subheader("Exploratory Data Analysis")
                vis_df = pd.DataFrame({
                    "Drug": [row["Drug"] for row in scores_data],
                    "Tanimoto Coefficient": [row["Tanimoto Coefficient"] for row in scores_data],
                    "HAN Model Score": [row["HAN Model Score"] for row in scores_data]
                })
                fig_bar = px.bar(
                    vis_df.melt(id_vars=["Drug"], value_vars=["Tanimoto Coefficient", "HAN Model Score"]),
                    x="Drug",
                    y="value",
                    color="variable",
                    title="Score Distribution Across Drugs",
                    barmode="group",
                    height=400
                )
                fig_bar.update_layout(showlegend=True)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.subheader("Molecular Structures")
                images = [(row["Drug"], row["Image"]) for row in info_data]
                num_cols = 3
                for i in range(0, len(images), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i + j
                        if idx < len(images):
                            drug_name, image = images[idx]
                            with cols[j]:
                                if image:
                                    st.image(f"data:image/png;base64,{image}", caption=f"Molecular Structure of {drug_name}", width=200)
                                else:
                                    st.write(f"No molecular structure available for {drug_name}")
                if drug_ids:
                    drug_idx = node_type_to_idx["Drug"].get(drug_ids[0])
                    disease_idx = node_type_to_idx["Disease"].get(disease_id)
                    if drug_idx is not None and disease_idx is not None:
                        test_drug_emb = all_drug_emb[drug_idx].numpy()
                        test_disease_emb = all_disease_emb[disease_idx].numpy()
                        test_instance = np.concatenate([test_drug_emb, test_disease_emb]).reshape(1, -1)
                        background_data = []
                        bg_drug_ids = nodes_df[nodes_df["label"] == "Drug"]["id"].sample(n=100, random_state=42).tolist()
                        for d_id in bg_drug_ids:
                            for dis_id in nodes_df[nodes_df["label"] == "Disease"]["id"].sample(n=5, random_state=42).tolist():
                                if d_id in node_type_to_idx["Drug"] and dis_id in node_type_to_idx["Disease"]:
                                    bg_drug_emb = all_drug_emb[node_type_to_idx["Drug"][d_id]].numpy()
                                    bg_disease_emb = all_disease_emb[node_type_to_idx["Disease"][dis_id]].numpy()
                                    noise = np.random.normal(0, 0.01, bg_drug_emb.shape)
                                    bg_drug_emb_noisy = np.clip(bg_drug_emb + noise, -1, 1)
                                    noise = np.random.normal(0, 0.01, bg_disease_emb.shape)
                                    bg_disease_emb_noisy = np.clip(bg_disease_emb + noise, -1, 1)
                                    background_data.append(np.concatenate([bg_drug_emb_noisy, bg_disease_emb_noisy]))
                        background_data = np.array(background_data)
                        def predict_function(x):
                            try:
                                x = np.array(x)
                                if len(x.shape) == 1:
                                    x = x.reshape(1, -1)
                                n_samples = x.shape[0]
                                predictions = np.zeros((n_samples, 1))
                                for i in range(n_samples):
                                    drug_emb = torch.tensor(x[i, :128], dtype=torch.float32).to(device)
                                    disease_emb = torch.tensor(x[i, 128:], dtype=torch.float32).to(device)
                                    predictions[i, 0] = torch.sigmoid((drug_emb * disease_emb).sum()).item()
                                return predictions
                            except Exception as e:
                                st.warning(f"Prediction function error: {str(e)}")
                                return np.zeros((n_samples, 1))
                        explainer = LimeTabularExplainer(
                            training_data=background_data,
                            mode='regression',
                            feature_names=[f"Drug_Feature_{i}" for i in range(128)] + [f"Disease_Feature_{i}" for i in range(128)],
                            verbose=True,
                            random_state=42
                        )
                        exp = explainer.explain_instance(
                            data_row=test_instance[0],
                            predict_fn=predict_function,
                            num_features=10,
                            num_samples=1000
                        )
                        st.subheader("LIME Explanation (Top Drug)")
                        exp_list = exp.as_list()
                        feature_names = [name for name, _ in exp_list]
                        feature_weights = [weight for _, weight in exp_list]
                        colors = ['blue' if w >= 0 else 'red' for w in feature_weights]
                        fig_lime = go.Figure(data=[
                            go.Bar(
                                x=feature_weights,
                                y=feature_names,
                                orientation='h',
                                marker_color=colors,
                                text=[f"{w:.4f}" for w in feature_weights],
                                textposition='auto'
                            )
                        ])
                        fig_lime.update_layout(
                            title="LIME Feature Importance",
                            xaxis_title="Feature Weight",
                            yaxis_title="Feature",
                            yaxis={'autorange': 'reversed'},
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_lime, use_container_width=True)
elif selected == "Similar Drugs Explorer":
    st.header("Similar Drugs Explorer 🔍")
    drug_options = nodes_df[nodes_df["label"] == "Drug"]["display_name"].tolist()
    selected_drug = st.selectbox("Select a Drug to Find Similar Drugs", drug_options)
    drug_id = nodes_df[nodes_df["display_name"] == selected_drug]["id"].iloc[0] if not nodes_df[nodes_df["display_name"] == selected_drug].empty else None
    if st.button("Find Similar Drugs") and drug_id:
        with st.spinner("Finding similar drugs..."):
            similar_drugs = find_similar_drugs(drug_id, nodes_df, all_drug_emb, node_type_to_idx)
            if similar_drugs:
                scores_data = []
                info_data = []
                selected_smiles, selected_synonyms, selected_chembl_id = fetch_chembl_data(selected_drug)
                selected_image = get_molecular_image(selected_smiles) if selected_smiles != "Not available" else None
                st.subheader(f"Selected Drug: {selected_drug}")
                selected_data = {
                    "Drug": selected_drug,
                    "SMILES": selected_smiles,
                    "Synonyms": selected_synonyms,
                    "ChEMBL ID": selected_chembl_id
                }
                st.table(pd.DataFrame([selected_data])[["Drug", "SMILES", "Synonyms", "ChEMBL ID"]])
                for sim_drug_id, sim_score, han_score in similar_drugs:
                    sim_drug_name = nodes_df[nodes_df["id"] == sim_drug_id]["display_name"].iloc[0]
                    sim_smiles, sim_synonyms, sim_chembl_id = fetch_chembl_data(sim_drug_name)
                    sim_image = get_molecular_image(sim_smiles) if sim_smiles != "Not available" else None
                    sim_props = literal_eval(nodes_df[nodes_df["id"] == sim_drug_id]["props"].iloc[0])
                    sim_smiles_internal = sim_props.get("smiles", "")
                    tc = compute_tanimoto_coefficient(selected_smiles, sim_smiles_internal) if selected_smiles != "Not available" and sim_smiles_internal else 0.0
                    scores_data.append({
                        "Similar Drug": sim_drug_name,
                        "Tanimoto Coefficient": round(tc, 4),
                        "HAN Model Score": round(han_score, 4)
                    })
                    info_data.append({
                        "Similar Drug": sim_drug_name,
                        "SMILES": sim_smiles,
                        "Synonyms": sim_synonyms,
                        "ChEMBL ID": sim_chembl_id,
                        "Image": sim_image
                    })
                if scores_data and info_data:
                    similar_scores_df = pd.DataFrame(scores_data)
                    similar_info_df = pd.DataFrame(info_data)
                    st.subheader("Similar Drugs Scores")
                    st.table(similar_scores_df[["Similar Drug", "Tanimoto Coefficient", "HAN Model Score"]])
                    st.subheader("Similar Drugs Information")
                    st.table(similar_info_df[["Similar Drug", "SMILES", "Synonyms", "ChEMBL ID"]])
                    st.subheader("Molecular Structures")
                    images = [(selected_drug, selected_image)] + [(row["Similar Drug"], row["Image"]) for row in info_data]
                    num_cols = 2
                    for i in range(0, len(images), num_cols):
                        cols = st.columns(num_cols)
                        for j in range(num_cols):
                            idx = i + j
                            if idx < len(images):
                                drug_name, image = images[idx]
                                with cols[j]:
                                    if image:
                                        st.image(f"data:image/png;base64,{image}", caption=f"Molecular Structure of {drug_name}", width=200)
                                    else:
                                        st.write(f"No molecular structure available for {drug_name}")
                else:
                    st.warning("No valid similar drugs data to display.")
            else:
                st.warning("No similar drugs found or an error occurred.")
elif selected == "Chatbot":
    chatbot_main()

driver.close()