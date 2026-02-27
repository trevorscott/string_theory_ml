import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

print("🌌 Igniting the God Mode CVAE Engine...")

# 1. Load the Massive Harvest
raw_data = torch.load("deep_space_50k.pt", weights_only=False)

# Build Master Physics Vocabulary
all_keys = set()
for u in raw_data:
    all_keys.update(u["Y_physics"].keys())
master_keys = sorted(list(all_keys))
y_dim = len(master_keys)

# 2. Pad the Geometry (Find the biggest universe to set the boundary)
max_simplices = max(len(u["X_simplices"]) for u in raw_data)
x_dim = max_simplices * 5 # 5 vertices per simplex
print(f"Max Universe Size: {max_simplices} Simplices. Flattened Blueprint Size: {x_dim}")

X_data = []
Y_data = []

for u in raw_data:
    # Flatten and pad X (The Geometry)
    flat_x = [v for simplex in u["X_simplices"] for v in simplex]
    padded_x = flat_x + [0] * (x_dim - len(flat_x))
    X_data.append(padded_x)
    
    # Extract Y (The Physics)
    y_dense = [0.0] * y_dim
    for k, v in u["Y_physics"].items():
        if k in master_keys:
            idx = master_keys.index(k)
            y_dense[idx] = float(v)
    Y_data.append(y_dense)

X_tensor = torch.tensor(X_data, dtype=torch.float32)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 3. The Conditional VAE Architecture
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
        # Concatenate geometry and physics
        xy = torch.cat([x, y], dim=1)
        h = F.relu(self.enc1(xy))
        h = F.relu(self.enc2(h))
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        # The "Probability Cloud" sampling trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # Concatenate sample and physics
        zy = torch.cat([z, y], dim=1)
        h = F.relu(self.dec1(zy))
        h = F.relu(self.dec2(h))
        return self.dec3(h) # Returns the reconstructed geometry

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# The Loss Function for a VAE is two parts: Reconstruction + KL Divergence (Cloud Shaping)
def cvae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = UniverseCVAE(x_dim, y_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Train the Generator
epochs = 50
print(f"🧠 Training God Mode Generator on {len(raw_data)} universes...")

model.train()
for epoch in range(epochs):
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
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Generative Loss: {avg_loss:.4f}")

# Save the generator and the vocabulary lengths
torch.save({
    'model_state': model.state_dict(),
    'x_dim': x_dim,
    'y_dim': y_dim,
    'master_keys': master_keys
}, "universe_generator.pth")

print("\n✅ God Mode Engine Saved. We are ready to generate custom reality.")