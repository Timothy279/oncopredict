import torch
import os

class Config:
    # Chemins
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset_with_smiles.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    
    # Hyperparamètres Modèle
    # Calculé : 10 types + 6 degrés + 5 H + 1 Aroma = 22
    NUM_NODE_FEATURES = 22  
    NUM_GENES = 498
    HIDDEN_DIM = 64
    HEADS = 2
    DROPOUT = 0.3
    
    # Hyperparamètres Entraînement
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    PATIENCE = 3 # Pour le scheduler
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Création automatique du dossier models
os.makedirs(Config.MODEL_DIR, exist_ok=True)

