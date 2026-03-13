import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Prevent numpy serialization errors on Mac (Compatible with NumPy 1.x and 2.x)
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except AttributeError:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 1. The Conditional VAE Architecture
class UniverseCVAE(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim=128):
        super(UniverseCVAE, self).__init__()
        
        # ENCODER: Compresses Geometry + Physics into a probability cloud
        self.enc1 = nn.Linear(x_dim + y_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        # DECODER: Expands Latent Cloud + Physics back into Geometry
        self.dec1 = nn.Linear(latent_dim + y_dim, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, x_dim)

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = F.relu(self.enc1(xy))
        h = F.relu(self.enc2(h))
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = F.relu(self.dec1(zy))
        h = F.relu(self.dec2(h))
        return self.dec3(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def cvae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the God Mode CVAE Engine.")
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Path to the raw harvested dataset (e.g., data/deep_space_50k.pt)")
    parser.add_argument('-o', '--output', type=str, default="checkpoints/universe_generator.pth", 
                        help="Path to save the trained model weights.")
    parser.add_argument('-e', '--epochs', type=int, default=50, 
                        help="Number of training epochs (default: 50)")
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help="Training batch size (default: 64)")
    
    args = parser.parse_args()
    
    print("🌌 Igniting the God Mode CVAE Engine...")

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found.")
        exit(1)

    # 2. Load the Massive Harvest
    print(f"📥 Loading dataset from: {args.input}")
    raw_data = torch.load(args.input, weights_only=False)

    # Build Master Physics Vocabulary
    all_keys = set()
    for u in raw_data:
        all_keys.update(u["Y_physics"].keys())
    master_keys = sorted(list(all_keys))
    y_dim = len(master_keys)

    # Pad the Geometry
    max_simplices = max(len(u["X_simplices"]) for u in raw_data)
    x_dim = max_simplices * 5 # 5 vertices per simplex
    print(f"📏 Max Universe Size: {max_simplices} Simplices. Flattened Blueprint Size: {x_dim}")

    X_data = []
    Y_data = []

    for u in raw_data:
        flat_x = [v for simplex in u["X_simplices"] for v in simplex]
        padded_x = flat_x + [0] * (x_dim - len(flat_x))
        X_data.append(padded_x)
        
        y_dense = [0.0] * y_dim
        for k, v in u["Y_physics"].items():
            if k in master_keys:
                idx = master_keys.index(k)
                y_dense[idx] = float(v)
        Y_data.append(y_dense)

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Initialize Training
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UniverseCVAE(x_dim, y_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"🧠 Training God Mode Generator on {len(raw_data)} universes...")
    print(f"⚙️  Settings: {args.epochs} Epochs | Batch Size: {args.batch_size} | Device: {device}")

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_y)
            loss = cvae_loss(recon_x, batch_x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader.dataset)
        
        # Dynamic print interval based on epochs
        print_interval = max(1, args.epochs // 10)
        if (epoch + 1) % print_interval == 0 or epoch == args.epochs - 1:
            print(f"   > Epoch {epoch+1}/{args.epochs} | Generative Loss: {avg_loss:.4f}")

    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    torch.save({
        'model_state': model.state_dict(),
        'x_dim': x_dim,
        'y_dim': y_dim,
        'master_keys': master_keys
    }, args.output)

    print("\n" + "="*50)
    print("✅ GOD MODE ENGINE SAVED")
    print(f"Weights written to: {args.output}")
    print("We are ready to generate custom reality.")
    print("="*50)   

