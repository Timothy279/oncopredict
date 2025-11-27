import pandas as pd
import pubchempy as pcp
import os
import time
import sys

# Force UTF-8 encoding for stdout on Windows to support emojis
sys.stdout.reconfigure(encoding='utf-8')

# CONFIG
PROCESSED_DIR = "data/processed"
INPUT_FILE = "merged_dataset.csv"
OUTPUT_FILE = "final_dataset_with_smiles.csv"

def fetch_smiles():
    print("🧪 Démarrage de la récupération des SMILES via PubChem...")
    
    # 1. Charger le dataset actuel
    path = os.path.join(PROCESSED_DIR, INPUT_FILE)
    if not os.path.exists(path):
        print(f"❌ Fichier non trouvé : {path}")
        return

    df = pd.read_csv(path)
    print(f"   -> Dataset chargé : {len(df)} lignes.")
    
    # 2. Identifier les médicaments uniques
    unique_drugs = df['DRUG_NAME'].unique()
    print(f"   -> {len(unique_drugs)} médicaments uniques à identifier.")
    
    # 3. Boucle de récupération (API Call)
    drug_to_smiles = {}
    print("   -> Interrogation de l'API PubChem (patience)...")
    
    for i, drug_name in enumerate(unique_drugs):
        try:
            # On cherche le composé par son nom
            compounds = pcp.get_compounds(drug_name, 'name')
            if compounds:
                # On prend le premier résultat (Isomeric SMILES est le plus précis)
                smiles = compounds[0].isomeric_smiles
                drug_to_smiles[drug_name] = smiles
                print(f"      ✅ ({i+1}/{len(unique_drugs)}) {drug_name} -> Trouvé")
            else:
                print(f"      ⚠️ ({i+1}/{len(unique_drugs)}) {drug_name} -> Non trouvé")
                drug_to_smiles[drug_name] = None
        except Exception as e:
            print(f"      ❌ Erreur sur {drug_name}: {e}")
            drug_to_smiles[drug_name] = None
            
        # Petite pause pour ne pas se faire bannir de l'API
        time.sleep(0.2)
        
    # 4. Mapping et Nettoyage
    print("\n🔗 Fusion des SMILES dans le dataset principal...")
    # On crée la colonne SMILES en mappant le dictionnaire
    df['SMILES'] = df['DRUG_NAME'].map(drug_to_smiles)
    
    # On supprime les lignes où on n'a pas trouvé de SMILES (on ne peut pas entraîner dessus)
    initial_len = len(df)
    df_clean = df.dropna(subset=['SMILES'])
    lost = initial_len - len(df_clean)
    
    print(f"   -> Lignes initiales : {initial_len}")
    print(f"   -> Lignes après nettoyage : {len(df_clean)} (Perdu : {lost})")
    
    # 5. Sauvegarde Finale
    save_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)
    df_clean.to_csv(save_path, index=False)
    print(f"💾 Sauvegardé sous : {save_path}")
    print("🚀 Prêt pour la Phase 2 (Modélisation GNN) !")

if __name__ == "__main__":
    fetch_smiles()

