import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from architectures.v2_diffusion.model import DenseDenoisingGNN
from architectures.v2_diffusion.noise_scheduler import DiscreteNoiseScheduler
import numpy as np

# Prevent numpy serialization errors when loading the .pt file
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

class CalabiYauDataset(Dataset):
    """
    Loads harvested string theory manifolds from disk and dynamically 
    translates their continuous simplices into padded discrete graph matrices.
    """
    def __init__(self, data_path, num_nodes=50):
        print(f"Loading harvested universes from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find {data_path}. Run DeepSpaceHarvester.py first!")
            
        
        self.raw_data = torch.load(data_path, weights_only=False)
        self.num_nodes = num_nodes
        
        # FIX 1: Calculate pos_weight empirically from actual data
        # instead of using an arbitrary hardcoded guess.
        total_ones = 0
        total_zeros = 0
        
        print(f"Successfully loaded {len(self.raw_data)} universes into memory.")
        print("Calculating class balance from training data...")
        for universe in self.raw_data:
            simplices = universe["X_simplices"]
            adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
            for simplex in simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        u, v = simplex[i], simplex[j]
                        if u < self.num_nodes and v < self.num_nodes:
                            adj[u, v] = 1.0
                            adj[v, u] = 1.0
            total_ones += adj.sum().item()
            total_zeros += (adj == 0).sum().item()
        
        self.pos_weight = (total_zeros / total_ones) if total_ones > 0 else 1.0
        print(f"Empirical pos_weight: {self.pos_weight:.2f}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        universe = self.raw_data[idx]
        simplices = universe["X_simplices"]
        
        adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        
        for simplex in simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    u, v = simplex[i], simplex[j]
                    if u < self.num_nodes and v < self.num_nodes:
                        adj[u, v] = 1.0
                        adj[v, u] = 1.0
                        
        return adj



def train_step(model, scheduler, optimizer, clean_graphs, device, pos_weight):
    model.train()
    optimizer.zero_grad()
    
    batch_size, n, _ = clean_graphs.shape
    t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device).long()
    noisy_graphs = scheduler.add_noise(clean_graphs, t)
    
    predicted_clean_logits = model(noisy_graphs, t)
    
    mask = (clean_graphs.sum(dim=-1, keepdim=True) > 0).float()
    mask2d = torch.bmm(mask, mask.transpose(1, 2))
    
    weight_tensor = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_tensor, reduction='none')
    
    raw_loss = loss_fn(predicted_clean_logits, clean_graphs)
    masked_loss = (raw_loss * mask2d).sum() / mask2d.sum()
    
    total_loss = masked_loss
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Discrete Graph Diffusion Model on Calabi-Yau manifolds.")
    parser.add_argument('-f', '--file', type=str, required=True, 
                        help="Path to the harvested .pt data file (e.g., data/deep_space_1000.pt)")
    parser.add_argument('-e', '--epochs', type=int, default=50, 
                        help="Number of training epochs (default: 50)")
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help="Batch size for training (default: 32)")
    parser.add_argument('-n', '--nodes', type=int, default=50, 
                        help="Matrix padding size for the graph nodes (default: 50)")
    parser.add_argument('--hidden_dim', type=int, default=256, 
                        help="Hidden dimension size for the GNN (default: 256)")
    
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing V2 Training Pipeline on: {device}")
    
    # Load dataset — pos_weight is calculated empirically inside __init__
    dataset = CalabiYauDataset(data_path=args.file, num_nodes=args.nodes)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,        # Parallel data loading
        pin_memory=False      # Must be False for MPS
    )
    
    scheduler = DiscreteNoiseScheduler(num_timesteps=100)
    model = DenseDenoisingGNN(num_nodes=args.nodes, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n--- INITIATING DISCRETE DIFFUSION ON {args.file} ---")
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0
            
            for batch_idx, clean_graphs in enumerate(dataloader):
                clean_graphs = clean_graphs.to(device)
                # FIX 4: Actually pass pos_weight into train_step.
                # Previously the function signature had a default of 5.0
                # but it was never passed from the training loop, so the
                # empirical value was silently ignored.
                loss = train_step(model, scheduler, optimizer, clean_graphs, device, dataset.pos_weight)
                epoch_loss += loss
                
            avg_loss = epoch_loss / len(dataloader)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{args.epochs} | Avg Loss: {avg_loss:.4f}")
                
        print("\n✅ V2 ARCHITECTURE FULLY TRAINED ON REAL GEOMETRY!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user! Salvaging current weights...")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/v2_diffusion_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model weights securely saved to {save_path}")