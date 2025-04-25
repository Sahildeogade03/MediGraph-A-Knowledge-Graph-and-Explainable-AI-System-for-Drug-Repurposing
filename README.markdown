# MediGraph: Drug Repurposing with Knowledge Graph & AI

MediGraph predicts drug-disease treatment relationships using a knowledge graph (KG) and EnhancedHAN model (AUC ~0.8519). Built with ChEMBL, Hetionet, and ClinicalTrials.gov data, it offers a Streamlit webapp with KG Explorer, Drug Repurposing Predictor, Similar Drugs Explorer, and Retrieval Assistant. LIME ensures explainable predictions.

## Features

- **Knowledge Graph**: Neo4j KG with 1,657 nodes, 2,420 edges.
- **EnhancedHAN Model**: PyTorch Geometric, 256 channels, 4 heads, AUC ~0.8519.
- **Explainable AI**: LIME for prediction insights via Plotly charts.
- **Webapp Modules**:
  - KG Explorer: Visualizes relationships.
  - Drug Repurposing: Ranks top 6 drugs with scores, LIME.
  - Similar Drugs: Finds 3 similar drugs.
  - Retrieval Assistant: Chatbot with LangChain, Mistral-7B.
- **Real-time Data**: ChEMBL API integration.
- **Scalable**: Supports rarer diseases, new sources.

## Installation

### Prerequisites

- Python 3.8+
- Neo4j AuraDB
- ChEMBL API
- HuggingFace account
- GPU (optional)

### Setup

1. Clone repo:
   ```bash
   git clone https://github.com/your-username/medigraph.git
   cd medigraph
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download SpaCy models:
   ```bash
   python -m spacy download en_ner_bc5cdr_md en_ner_bionlp13cg_md
   ```

5. Configure Neo4j:
   - Edit `config/neo4j_config.json`:
     ```json
     {"uri": "neo4j+s://your-instance.databases.neo4j.io", "username": "neo4j", "password": "your-password"}
     ```

6. Set HuggingFace token:
   ```bash
   echo "HUGGINGFACE_TOKEN=your-token" > .env
   ```

7. Add pre-trained model:
   - Place `han_drug_repurposing_retrained.pth` in `models/`.

## Usage

### Run Webapp

1. Start Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Access: `http://localhost:8501`

### Webapp Modules

- **KG Explorer**: Select Cypher query, view graph.
- **Drug Repurposing**: Pick disease, see top 6 drugs, LIME charts.
- **Similar Drugs**: Select drug, view 3 analogs.
- **Retrieval Assistant**: Ask drug repurposing questions.

### Train Model

1. Ensure `data/nodes_df.csv`, `data/rels_df.csv`, `data/type_to_idx.pkl`.
2. Run:
   ```bash
   python scripts/train_model.py
   ```
3. Outputs: `models/han_drug_repurposing_retrained.pth`, plots in `plots/`.

## Screenshots

- ![KG Schema](https://github.com/user-attachments/assets/3a650a11-d8d0-411d-9718-51ecd283dcfe)
- ![KG Explorer](https://github.com/user-attachments/assets/e099ec85-043c-4505-8bd5-55decfb59c32)
- ![Drug Repurposing](https://github.com/user-attachments/assets/839409c7-3996-4372-911d-fa84db1f2b06)
- ![Similar Drugs](https://github.com/user-attachments/assets/14101534-e636-462a-b77d-6a1dbdec1492)
- ![Retrieval Assistant](https://github.com/user-attachments/assets/50d6613d-cbe3-44c8-9c71-5339abb7dd0e)
- ![LIME](https://github.com/user-attachments/assets/59a84272-365d-471d-8b05-a16c33232a50)

## Datasets

- **ChEMBL**: Molecules from `chembl_35.db`.
- **Hetionet**: Entities, relationships from `hetionet-v1.0-nodes.tsv`, `edges.sif`.
- **ClinicalTrials.gov**: Trials from `ctg-studies.csv`.
- **Preprocessing**: SpaCy for entities, RDKit for fingerprints.

## Model Details

- **Architecture**: HANConv (256 channels, 4 heads), 128D embeddings, 1.92M parameters.
- **Training**: Adam (lr=0.001), BCEWithLogitsLoss, 500 epochs, AUC ~0.8519.
- **Scoring**:
  - Tanimoto: $T(A, B) = \frac{\mathbf{f}_A \cdot \mathbf{f}_B}{|\mathbf{f}_A|^2 + |\mathbf{f}_B|^2 - \mathbf{f}_A \cdot \mathbf{f}_B}$
  - HAN Score: $S(d, s) = \sigma(\mathbf{e}_d \cdot \mathbf{e}_s)$

## Webapp Functionalities

- **KG Explorer**: ~18 nodes/15 edges per query.
- **Drug Repurposing**: Predicts drugs (e.g., Fludarabine).
- **Similar Drugs**: Finds analogs (e.g., Luspatercept).
- **Retrieval Assistant**: Mistral-7B, FAISS-based answers.

## Explainability

- **LIME**: Highlights top 10 embedding dimensions.
- **Visualization**: Plotly bar charts (positive/negative weights).

## License

MIT License. See [LICENSE](LICENSE).