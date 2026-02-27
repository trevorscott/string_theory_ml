import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from cytools import fetch_polytopes

# 1. SETUP HARDWARE
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. MATCH THE NEW ARCHITECTURE (Must be identical to BBN.py)
class BottNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(36 * 4, 1024), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),         
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x.view(-1, 36 * 4))

# Load the weights
model = BottNet().to(device)
model.load_state_dict(torch.load("bott_net_balanced.pth"))
model.eval()

# 3. RUN THE DYNAMIC STRESS TEST
h11_target = random.randint(5, 25)
print(f"🎲 Stress Testing with h11 target: {h11_target}...")

test_list = fetch_polytopes(h11=h11_target, limit=20, as_list=True)

if test_list:
    p = random.choice(test_list)
    actual_h11 = p.h11(lattice="N")
    actual_h21 = p.h21(lattice="N")
    
    # NORMALIZATION: This must match the harvest script (/ 10.0)
    test_verts = np.zeros((1, 36, 4))
    v = p.vertices()
    test_verts[0, :v.shape[0], :] = v / 10.0 
    
    test_input = torch.tensor(test_verts, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred = model(test_input)
    
    print("-" * 35)
    print(f"ACTUAL HODGE:    ({actual_h11}, {actual_h21})")
    print(f"PREDICTED HODGE: ({pred[0][0]:.2f}, {pred[0][1]:.2f})")
    print("-" * 35)
    
    # Calculate Error
    d1 = abs(actual_h11 - pred[0][0].item())
    d2 = abs(actual_h21 - pred[0][1].item())
    print(f"Error Delta: h11 (+/- {d1:.2f}), h21 (+/- {d2:.2f})")
else:
    print("Could not fetch a test case.")