import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from config import Config
from dataset import DrugOmicsDataset
from model import DrugResponseModel

def train_pipeline():
    print(f"[INFO] Device detected: {Config.DEVICE}")
    print("[INFO] Initializing dataset...")
    
    dataset = DrugOmicsDataset(Config.DATA_PATH)
    
    # Validation split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"[INFO] Training on {len(train_dataset)} samples, Validating on {len(val_dataset)}")

    # Model Init
    model = DrugResponseModel().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=Config.PATIENCE, verbose=True)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    
    print("[INFO] Starting training loop...")
    
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(Config.DEVICE)
            optimizer.zero_grad()
            
            output = model(batch.x, batch.edge_index, batch.batch, batch.x_gen)
            target = batch.y.view(-1, 1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(Config.DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch, batch.x_gen)
                target = batch.y.view(-1, 1)
                val_loss = criterion(out, target)
                total_val_loss += val_loss.item() * batch.num_graphs
        
        avg_val_loss = total_val_loss / len(val_dataset)
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {duration:.1f}s")

        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"   >>> Model saved to {Config.MODEL_PATH}")

    print(f"[DONE] Training finished. Best Validation Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_pipeline()
