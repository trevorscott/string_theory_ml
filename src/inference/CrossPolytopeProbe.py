import torch
import numpy as np
from cytools import fetch_polytopes
import os

# 1. Environment Fix
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Tell Torch that NumPy scalars are 'safe' for loading
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 2. Setup Device (M2 Max GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 3. Match the Training Architecture exactly
class IntersectionNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IntersectionNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            
            torch.nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 4. Load Metadata and Model
print("📂 Loading model and metadata...")
data_meta = torch.load("polytope5_tensors.pt", weights_only=False)
input_dim = data_meta["input_dim"]   # Should be 170
output_dim = data_meta["output_dim"] # Should be 165

model = IntersectionNet(input_dim, output_dim).to(device)
model.load_state_dict(torch.load("intersection_model.pth", weights_only=False))
model.eval()

# 5. Fetch the "Unseen" Polytope
print("🧐 Fetching a new, unseen scaffold (Polytope #6)...")
polys = fetch_polytopes(h11=7, limit=50, as_list=True)
p_new = polys[6] 

# 6. Generate one universe from the new scaffold
t_new = next(p_new.random_triangulations_fast(N=1))
simplices = [list(s) for s in t_new.simplices()]

# 7. THE CLIP/PAD TRICK
# Convert to flat tensor
x_raw = torch.tensor(simplices, dtype=torch.float32).flatten()

# Prepare a fixed-size input of 170
x_input = torch.zeros(input_dim)

# Determine how much to copy:
# If x_raw > 170, we CLIP it. If x_raw < 170, the rest stays ZEROS (Padding).
size_to_copy = min(len(x_raw), input_dim)
x_input[:size_to_copy] = x_raw[:size_to_copy]

# Add batch dimension and move to M2 GPU
x_input = x_input.unsqueeze(0).to(device)

# 8. Predict vs Ground Truth
print("🔮 Predicting Intersection Numbers...")
with torch.no_grad():
    prediction = model(x_input).cpu().numpy().flatten()

# Get true physics for comparison
actual_dict = t_new.get_cy().intersection_numbers()

print("\n--- TEST RESULTS ---")
print(f"Scaffold Input Size: {len(x_raw)} features (Clipped/Padded to {input_dim})")
print(f"Max Prediction Value: {np.max(prediction):.4f}")
print(f"Min Prediction Value: {np.min(prediction):.4f}")

# Check if it's outputting anything significant
if np.mean(np.abs(prediction)) < 0.01:
    print("\n⚠️  The model is outputting near-zero. It doesn't recognize this geometry.")
else:
    print("\n✅ The model is projecting its 'intuition' onto the new scaffold!")
    print("Top 5 Predicted Couplings:", prediction[:5])