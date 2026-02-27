import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import numpy as np

# Prevent numpy serialization errors on Mac
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 1. Redefine the Architecture (Must match exactly to load the weights)
class UniverseCVAE(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim=128):
        super(UniverseCVAE, self).__init__()
        # ENCODER (Needed for loading weights, though we won't use it to generate)
        self.enc1 = nn.Linear(x_dim + y_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        # DECODER (The actual Universe Generator)
        self.dec1 = nn.Linear(latent_dim + y_dim, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, x_dim)

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = torch.relu(self.dec1(zy))
        h = torch.relu(self.dec2(h))
        return self.dec3(h)

# 2. Wake up the God Mode Engine
print("🔮 Waking up the God Mode Engine...")
checkpoint = torch.load("universe_generator.pth", weights_only=False)
x_dim = checkpoint['x_dim']
y_dim = checkpoint['y_dim']
master_keys = checkpoint['master_keys']

model = UniverseCVAE(x_dim, y_dim)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 3. Define the Physics You Want!
# For this test run, we will set a blank slate, and intentionally spike the first three 
# physical intersection properties to highly specific, arbitrary numbers.
target_y = torch.zeros(1, y_dim)

if y_dim >= 3:
    target_y[0, 0] = 24.0  # e.g., We want a universe with a property of 24
    target_y[0, 1] = -12.0 # We want a negative curvature here
    target_y[0, 2] = 8.0   

# 4. Pull a seed from the Quantum Cloud
# We sample a random point in the 128-dimensional latent space
z = torch.randn(1, 128)

# 5. Generate the Universe
print("⚡ Commanding the math to build the geometry...")
with torch.no_grad():
    generated_flat_x = model.decode(z, target_y)[0].numpy()

# 6. Format the mathematical output back into structural Geometry
# The AI outputs continuous decimals, but vertices are strict integers. 
vertices = np.round(generated_flat_x).astype(int)

simplices = []
for i in range(0, len(vertices), 5):
    simplex = vertices[i:i+5].tolist()
    
    # Filter out the empty padded areas we used for training
    if sum(simplex) != 0:
        # We take the absolute value because vertex IDs can't be negative
        simplex = [abs(v) for v in simplex]
        simplices.append(simplex)

print("\n" + "="*50)
print("✨ CUSTOM UNIVERSE GENERATED ✨")
print(f"Target Physics Applied: [24.0, -12.0, 8.0, ...]")
print(f"Generated a Manifold with {len(simplices)} Simplices.")
print("Structural Blueprint (First 10 Simplices):")
for i, s in enumerate(simplices[:10]):
    print(f"  Simplex {i}: {s}")
if len(simplices) > 10:
    print("  ... (blueprint truncated for screen)")
print("="*50)