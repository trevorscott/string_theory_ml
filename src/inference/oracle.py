import torch
import numpy as np
from cytools import fetch_polytopes
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load Architecture & Metadata
class UniversalIntersectionNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UniversalIntersectionNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim)
        )
    def forward(self, x): return self.net(x)

data_meta = torch.load("universal_5k_tensors.pt", weights_only=False)
input_dim = data_meta["input_dim"]
output_dim = data_meta["output_dim"]

model = UniversalIntersectionNet(input_dim, output_dim).to(device)
model.load_state_dict(torch.load("universal_model.pth", weights_only=False))
model.eval()

# 2. Fetch a completely UNSEEN Polytope
print("🔮 Summoning the Oracle for Polytope #205 (Unseen Geometry)...")
polys = fetch_polytopes(h11=7, limit=210, as_list=True)
p_new = polys[205] 
t_new = next(p_new.random_triangulations_fast(N=1))

# 3. Format the Input
simplices = [list(s) for s in t_new.simplices()]
x_raw = torch.tensor(simplices, dtype=torch.float32).flatten()

x_input = torch.zeros(input_dim)
size_to_copy = min(len(x_raw), input_dim - 1) # -1 to leave room for Scaffold ID
x_input[:size_to_copy] = x_raw[:size_to_copy]
x_input[-1] = 205.0 # The new Scaffold ID

x_input = x_input.unsqueeze(0).to(device)

# 4. Predict
with torch.no_grad():
    prediction = model(x_input).cpu().numpy().flatten()

print("\n--- ORACLE PREDICTION vs REALITY ---")
# Let's just look at the highest magnitude predictions to see if it's confident
top_indices = np.argsort(np.abs(prediction))[-5:][::-1]

print("Top 5 Predicted Couplings (Index -> Value):")
for idx in top_indices:
    print(f"  Vocabulary Index [{idx:3d}]: Predicted {prediction[idx]:.2f}")

print("\n(Note: The AI calculated this instantly, bypassing the TOPCOM C++ backend.)")