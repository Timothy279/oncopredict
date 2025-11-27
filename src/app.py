import streamlit as st
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os

from config import Config
from model import DrugResponseModel
from dataset import smile_to_graph

# Page config
st.set_page_config(page_title="OncoPredict", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    """Loads the model from config path."""
    model = DrugResponseModel()
    if os.path.exists(Config.MODEL_PATH):
        try:
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
            model.to(Config.DEVICE)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Échec du chargement de l'architecture du modèle : {e}")
            return None
    return None

def predict_ic50(model, smiles, gene_vector):
    # 1. Input Sanitization
    clean_smiles = smiles.strip().replace("\n", "").replace(" ", "")
    
    # 2. Graph Conversion
    graph_data = smile_to_graph(clean_smiles)
    if graph_data is None:
        return None
        
    x, edge_index = graph_data
    
    # 3. Tensor Prep
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    x = x.to(Config.DEVICE)
    edge_index = edge_index.to(Config.DEVICE)
    batch = batch.to(Config.DEVICE)
    x_gen = torch.tensor(np.array([gene_vector]), dtype=torch.float).to(Config.DEVICE)

    # 4. Inference
    with torch.no_grad():
        pred = model(x, edge_index, batch, x_gen)
    
    return pred.item()

# --- UI Layout ---
st.title("🧬 OncoPredict : Prédiction de Réponse aux Médicaments")
st.markdown("""
Cet outil utilise un **Graph Attention Network (GAT)** pour prédire le score IC50 d'une molécule 
basé sur le profil génétique d'une tumeur.
""")

model = load_model()

if model is None:
    st.warning(f"Modèle introuvable à {Config.MODEL_PATH}. Veuillez d'abord entraîner le modèle.")
    st.stop()

# Layout
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Molécule (SMILES)")
    default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    smiles_input = st.text_area("Entrez le code SMILES", value=default_smiles, height=100)

    if smiles_input:
        clean_s = smiles_input.strip().replace("\n", "").replace(" ", "")
        mol = Chem.MolFromSmiles(clean_s)
        if mol:
            st.image(Draw.MolToImage(mol), caption="Structure Chimique", width=400)
        else:
            st.error("Format SMILES invalide.")

with col_right:
    st.subheader("2. Profil Génomique")
    
    # Mapping des options pour la logique interne vs affichage
    profile_options = {
        "Profil Moyen (Référence)": "Average",
        "Mutation Résistante": "Resistant",
        "Mutation Sensible": "Sensitive",
        "Aléatoire": "Random"
    }
    
    profile_display = st.selectbox("Sélectionnez le Profil Tumoral", list(profile_options.keys()))
    profile_mode = profile_options[profile_display]
    
    # Logic to generate gene vector
    if profile_mode == "Random":
        gene_vector = np.random.randn(Config.NUM_GENES)
    elif profile_mode == "Resistant":
        gene_vector = np.ones(Config.NUM_GENES) * 2.0
    elif profile_mode == "Sensitive":
        gene_vector = np.ones(Config.NUM_GENES) * -2.0
    else:
        # Average
        gene_vector = np.zeros(Config.NUM_GENES)

    st.markdown("---")
    
    if st.button("Lancer la Prédiction", type="primary"):
        with st.spinner("Traitement en cours..."):
            result = predict_ic50(model, smiles_input, gene_vector)
        
        if result is None:
            st.error("Erreur : Impossible de traiter la molécule. Vérifiez la syntaxe SMILES.")
        else:
            st.success("Inférence Réussie")
            st.metric("IC50 Prédit (Log)", f"{result:.4f}")
            st.caption("Des valeurs IC50 plus faibles indiquent une plus grande puissance du médicament.")
