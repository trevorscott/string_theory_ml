import torch
import numpy as np
from torch.utils.data import TensorDataset

# Allow numpy scalars
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
raw_data = torch.load("universal_multiverse_5k.pt", weights_only=False)

print("📂 Processing 5,000 Universes...")

# 1. Build Global Y-Vocabulary
all_keys = set()
for universe in raw_data:
    all_keys.update(universe["Y_physics"].keys())
master_keys = sorted(list(all_keys))
y_dim = len(master_keys)
print(f"Global Y-Vector Size: {y_dim} couplings.")

# 2. Find Max X-Length
max_simplices = max(len(u["X_simplices"]) for u in raw_data)
max_x_len = (max_simplices * 5) + 1 # +1 for the Scaffold ID
print(f"Max X-Length (with ID): {max_x_len}")

X_tensors = []
Y_tensors = []

for universe in raw_data:
    # --- Process X ---
    x_flat = torch.tensor(universe["X_simplices"], dtype=torch.float32).flatten()
    
    # Pad and add Scaffold ID at the very end
    x_padded = torch.zeros(max_x_len)
    x_padded[:len(x_flat)] = x_flat
    x_padded[-1] = float(universe["scaffold_id"]) # The "Context" flag
    X_tensors.append(x_padded)
    
    # --- Process Y ---
    y_dense = torch.zeros(y_dim)
    for key, value in universe["Y_physics"].items():
        idx = master_keys.index(key)
        y_dense[idx] = float(value)
    Y_tensors.append(y_dense)

X_all = torch.stack(X_tensors)
Y_all = torch.stack(Y_tensors)

print(f"✅ Final Tensors - X: {X_all.shape}, Y: {Y_all.shape}")

torch.save({
    "X": X_all, "Y": Y_all, 
    "input_dim": max_x_len, "output_dim": y_dim
}, "universal_5k_tensors.pt")