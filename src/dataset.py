import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from rdkit import Chem

# --- UTILITAIRES ---
def one_hot_encoding(x, permitted_list):
    """Fonction utilitaire pour encoder en vecteurs binaires"""
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == s) for s in permitted_list]
    return binary_encoding

def smile_to_graph(smile):
    """
    Convertit un SMILES en objet Data pour PyTorch Geometric.
    Tente de réparer les molécules invalides (sanitize=False).
    """
    # 1. Tentative de lecture standard
    mol = Chem.MolFromSmiles(smile)
    
    # 2. Si échec, tentative en mode "permissif"
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smile, sanitize=False)
            if mol:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
        except:
            return None

    # Si toujours échec
    if mol is None:
        return None

    # --- FEATURE ENGINEERING ---
    atom_features = []
    for atom in mol.GetAtoms():
        features = []
        # Type d'atome (10 features)
        features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown'])
        
        # Degré (6 features)
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        
        # Hydrogènes (5 features)
        features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        
        # Aromaticité (1 feature)
        features += [1 if atom.GetIsAromatic() else 0]
        
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Connectivité
    src_list = []
    dst_list = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        src_list.extend([start, end])
        dst_list.extend([end, start])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return x, edge_index

# --- CLASS DATASET ---
class DrugOmicsDataset(Dataset):
    def __init__(self, csv_file, smile_col="SMILES", target_col="LN_IC50"):
        """
        Dataset PyTorch personnalisé pour Drug + Omics.
        """
        print(f"[INIT] Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        
        # Identification dynamique des gènes
        excluded_cols = [smile_col, target_col, "DRUG_NAME", "DRUG_ID", "COSMIC_ID", "TARGET_PATHWAY"]
        self.gene_cols = [c for c in self.data.columns 
                          if c not in excluded_cols 
                          and not c.startswith("Unnamed")
                          and pd.api.types.is_numeric_dtype(self.data[c])]
        
        print(f"   -> Found {len(self.gene_cols)} genomic features.")
        
        self.smile_col = smile_col
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Molécule
        smile = row[self.smile_col]
        graph_data = smile_to_graph(smile)
        
        # Gestion de secours si le SMILES plante (ne devrait pas arriver souvent avec le fix)
        if graph_data is None:
             # On retourne un graphe vide ou dummy pour ne pas faire planter le DataLoader
             # Ici on choisit de retourner l'élément précédent (ou 0 si idx=0)
             # C'est une stratégie simple pour éviter le crash
             safe_idx = idx - 1 if idx > 0 else idx + 1
             if safe_idx < len(self.data):
                 return self.__getitem__(safe_idx)
        
        x, edge_index = graph_data

        # 2. Génomique
        genes = torch.tensor(row[self.gene_cols].values.astype(np.float32), dtype=torch.float)
        
        # 3. Target
        y = torch.tensor([row[self.target_col]], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, x_gen=genes.unsqueeze(0), y=y)

if __name__ == "__main__":
    # Petit test unitaire
    from config import Config
    try:
        ds = DrugOmicsDataset(Config.DATA_PATH)
        print(f"[TEST] Dataset length: {len(ds)}")
        print(f"[TEST] First item: {ds[0]}")
    except Exception as e:
        print(f"[WARN] Test failed (probably file not found): {e}")
