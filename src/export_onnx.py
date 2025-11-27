import torch
from config import Config
from model import DrugResponseModel

def export_to_onnx():
    print("[INFO] Chargement du modèle pour export ONNX...")
    
    # 1. Charger l'architecture et les poids
    model = DrugResponseModel()
    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location="cpu"))
        model.eval()
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le modèle : {e}")
        return

    # 2. Créer des fausses données (Dummy Input) pour tracer le graphe
    # On simule 1 molécule avec 10 atomes et 1 profil génétique
    dummy_num_atoms = 10
    dummy_x = torch.randn(dummy_num_atoms, Config.NUM_NODE_FEATURES)
    # Une liaison simple entre atome 0 et 1
    dummy_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # Batch vector (tous les atomes appartiennent à la molécule 0)
    dummy_batch = torch.zeros(dummy_num_atoms, dtype=torch.long)
    # Profil génétique
    dummy_x_gen = torch.randn(1, Config.NUM_GENES)

    # 3. Export ONNX
    output_path = "models/architecture_visual.onnx"
    
    print("[INFO] Exportation en cours...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_edge_index, dummy_batch, dummy_x_gen), # Les entrées
        output_path,
        export_params=True,        # Stocker les poids
        opset_version=16,          # Version plus récente pour supporter scatter_reduce
        do_constant_folding=True,  # Simplifier le graphe
        input_names=['Atom_Features', 'Edge_Index', 'Batch_Map', 'Genetics'],
        output_names=['IC50_Prediction'],
        dynamic_axes={
            'Atom_Features': {0: 'num_atoms'},
            'Edge_Index': {1: 'num_edges'},
            'Batch_Map': {0: 'num_atoms'}
        }
    )
    
    print(f"[SUCCÈS] Modèle visuel sauvegardé : {output_path}")
    print("-> Ouvrez ce fichier sur https://netron.app")

if __name__ == "__main__":
    export_to_onnx()

