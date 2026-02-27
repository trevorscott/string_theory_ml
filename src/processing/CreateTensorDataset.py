import torch
import numpy as np
from torch.utils.data import TensorDataset

# Allow numpy scalars for the load
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
raw_data = torch.load("polytope5_multiverse.pt", weights_only=False)

# 1. Build Y-Vocabulary (Same as before)
all_keys = set()
for universe in raw_data:
    all_keys.update(universe["Y_physics"].keys())
master_keys = sorted(list(all_keys))
y_dim = len(master_keys)

# 2. Find the Max X-Size (PADDING PREP)
max_simplices = max(len(u["X_simplices"]) for u in raw_data)
max_x_len = max_simplices * 5
print(f"Max simplices: {max_simplices} (Total X-length: {max_x_len})")

X_tensors = []
Y_tensors = []

for universe in raw_data:
    # --- Process X with Padding ---
    x_flat = torch.tensor(universe["X_simplices"], dtype=torch.float32).flatten()
    
    # Create a zero-tensor of the max size and 'paste' our data into it
    x_padded = torch.zeros(max_x_len)
    x_padded[:len(x_flat)] = x_flat
    X_tensors.append(x_padded)
    
    # --- Process Y (Same as before) ---
    y_dense = torch.zeros(y_dim)
    for key, value in universe["Y_physics"].items():
        idx = master_keys.index(key)
        y_dense[idx] = float(value)
    Y_tensors.append(y_dense)

X_all = torch.stack(X_tensors)
Y_all = torch.stack(Y_tensors)

print(f"✅ Final Shapes - X: {X_all.shape}, Y: {Y_all.shape}")

# Save the datasets (train/test split logic remains the same)
torch.save({
    "X": X_all, "Y": Y_all, 
    "input_dim": max_x_len, "output_dim": y_dim
}, "polytope5_tensors.pt")