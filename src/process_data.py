import pandas as pd
import os
import gc # Garbage Collector pour libérer la mémoire
import sys

# Force UTF-8 encoding for stdout on Windows
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Noms exacts des fichiers (Vérifiez qu'ils matchent les vôtres !)
FILE_IC50 = "GDSC2_fitted_dose_response_27Oct23.xlsx"
FILE_DRUGS = "screened_compounds_rel_8.5.csv"
FILE_GENOMICS = "Cell_line_RMA_proc_basalExp.txt.zip"

def process_data():
    print("🚀 Démarrage du Data Pipeline (Oncologie)...")

    # 1. CHARGEMENT DES DROGUES
    print("\n🧪 1. Chargement des Molécules (Noms & Cibles)...")
    try:
        df_drugs = pd.read_csv(os.path.join(RAW_DIR, FILE_DRUGS))
        # On renomme pour standardiser
        # NOTE: La colonne SMILES n'est pas présente dans la v8.5, on garde le nom et la voie de signalisation
        df_drugs = df_drugs[['DRUG_ID', 'DRUG_NAME', 'TARGET_PATHWAY']].dropna()
        df_drugs['DRUG_ID'] = df_drugs['DRUG_ID'].astype(int)
        print(f"   -> {len(df_drugs)} molécules trouvées.")
    except Exception as e:
        print(f"❌ Erreur Molécules: {e}")
        return

    # 2. CHARGEMENT DE LA RÉPONSE (IC50)
    print("\n🎯 2. Chargement des IC50 (Cibles)...")
    try:
        # Lecture Excel (peut être long)
        df_ic50 = pd.read_excel(os.path.join(RAW_DIR, FILE_IC50))
        # On ne garde que les colonnes utiles
        df_ic50 = df_ic50[['COSMIC_ID', 'DRUG_ID', 'LN_IC50']].dropna()
        df_ic50['COSMIC_ID'] = df_ic50['COSMIC_ID'].astype(int)
        df_ic50['DRUG_ID'] = df_ic50['DRUG_ID'].astype(int)
        print(f"   -> {len(df_ic50)} expériences trouvées.")
    except Exception as e:
        print(f"❌ Erreur IC50: {e}")
        return

    # 3. MERGE 1 : IC50 + Infos Drogues
    print("\n🔗 3. Fusion IC50 + Infos Drogues...")
    merged_df = df_ic50.merge(df_drugs, on='DRUG_ID', how='inner')
    print(f"   -> {len(merged_df)} paires valides (Drug+IC50).")
    
    # Nettoyage mémoire
    del df_drugs, df_ic50
    gc.collect()

    # 4. CHARGEMENT GÉNOMIQUE (Optimisé)
    print("\n🧬 4. Chargement Génomique (Lourd)...")
    try:
        # On lit le ZIP direct avec Pandas
        # Format: Index=Gene, Cols=CellLines(COSMIC_ID)
        df_gen = pd.read_csv(os.path.join(RAW_DIR, FILE_GENOMICS), sep='\t', compression='zip')
        
        # Nettoyage Index (Gènes)
        df_gen = df_gen.set_index(df_gen.columns[1]) # Colonne GENE_title
        df_gen = df_gen.drop(columns=[df_gen.columns[0]]) # Drop ID interne
        
        # Transpose : On veut Lignes=Cellules, Cols=Gènes
        df_gen = df_gen.T
        
        # Nettoyage des IDs Cellules ("DATA.906826" -> 906826)
        # Gestion robuste : suppression du préfixe "DATA." et conversion float -> int pour gérer les cas comme "123.0" ou "123.1"
        df_gen.index = df_gen.index.str.replace("DATA.", "", regex=False)
        df_gen.index = df_gen.index.map(lambda x: int(float(x)))
        
        # FEATURE SELECTION (Crucial pour ne pas exploser la RAM)
        # On garde les 500 gènes les plus variables (ceux qui différencient les cancers)
        print("   -> Sélection des 500 gènes les plus importants...")
        top_genes = df_gen.var().nlargest(500).index
        df_gen = df_gen[top_genes]
        
        print(f"   -> Génomique prête : {df_gen.shape}")
        
    except Exception as e:
        print(f"❌ Erreur Génomique: {e}")
        return

    # 5. MERGE FINAL
    print("\n🔗 5. Fusion Finale (Génomique + Le reste)...")
    # On merge sur l'index de df_gen (COSMIC_ID) et la colonne COSMIC_ID de merged_df
    final_df = merged_df.merge(df_gen, left_on='COSMIC_ID', right_index=True, how='inner')
    
    print("-" * 30)
    print(f"✅ DATASET FINAL : {final_df.shape}")
    print(f"   - {final_df.shape[0]} exemples d'entraînement.")
    print(f"   - {final_df.shape[1]} colonnes (Features).")
    
    # 6. SAUVEGARDE
    save_path = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    final_df.to_csv(save_path, index=False)
    print(f"\n💾 Sauvegardé sous : {save_path}")

if __name__ == "__main__":
    # Vérification des librairies
    try:
        import openpyxl
    except ImportError:
        print("⚠️ Installation de openpyxl requise...")
        os.system("pip install openpyxl")
        
    process_data()

