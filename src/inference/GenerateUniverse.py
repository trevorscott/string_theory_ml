import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# Prevent numpy serialization errors on Mac (Compatible with NumPy 1.x and 2.x)
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except AttributeError:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 1. Redefine the Architecture (Must match exactly to load the weights)
class UniverseCVAE(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim=128):
        super(UniverseCVAE, self).__init__()
        self.enc1 = nn.Linear(x_dim + y_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        self.dec1 = nn.Linear(latent_dim + y_dim, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, x_dim)

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = torch.relu(self.dec1(zy))
        h = torch.relu(self.dec2(h))
        return self.dec3(h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a custom universe using the God Mode CVAE.")
    parser.add_argument('-m', '--model', type=str, default="checkpoints/universe_generator.pth", 
                        help="Path to the trained God Mode model weights.")
    parser.add_argument('-p', '--physics', type=float, nargs='+', default=[24.0, -12.0, 8.0],
                        help="A list of target physical properties (e.g., -p 24.0 -12.0 8.0)")
    
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model checkpoint '{args.model}' not found. Did you train it yet?")
        exit(1)

    print("🔮 Waking up the God Mode Engine...")
    checkpoint = torch.load(args.model, weights_only=False)
    x_dim = checkpoint['x_dim']
    y_dim = checkpoint['y_dim']
    master_keys = checkpoint['master_keys']

    model = UniverseCVAE(x_dim, y_dim)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 3. Define the Physics You Want!
    target_y = torch.zeros(1, y_dim)
    
    # Map the user's requested physics to the target tensor
    for i, val in enumerate(args.physics):
        if i < y_dim:
            target_y[0, i] = val
        else:
            print(f"⚠️ Warning: You provided more physics targets than the model's vocabulary ({y_dim}). Truncating.")
            break

    # 4. Pull a seed from the Quantum Cloud
    z = torch.randn(1, 128)

    # 5. Generate the Universe
    print(f"⚡ Commanding the math to build the geometry...")
    with torch.no_grad():
        generated_flat_x = model.decode(z, target_y)[0].numpy()

    # 6. Format the mathematical output back into structural Geometry
    vertices = np.round(generated_flat_x).astype(int)

    simplices = []
    for i in range(0, len(vertices), 5):
        simplex = vertices[i:i+5].tolist()
        
        # Filter out the empty padded areas
        if sum(simplex) != 0:
            simplex = [abs(v) for v in simplex]
            simplices.append(simplex)

    
    print("\n" + "="*50)
    print("✨ CUSTOM UNIVERSE GENERATED ✨")
    print(f"Target Physics Applied: {args.physics[:y_dim]}")
    print(f"Generated a Manifold with {len(simplices)} Simplices.")
    print("Structural Blueprint (First 10 Simplices):")
    for i, s in enumerate(simplices[:10]):
        print(f"  Simplex {i}: {s}")
    if len(simplices) > 10:
        print("  ... (blueprint truncated for screen)")
    print("="*50)