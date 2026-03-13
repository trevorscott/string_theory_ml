import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from cytools import fetch_polytopes

# --- CONFIG ---
TARGET_COUNT = 100000    # Goal for tonight
DATA_FILE = "landscape_100k.pt"
MODEL_FILE = "bott_net.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================
# STAGE 1: THE HARVESTER
# ==========================================
def get_data():
    if os.path.exists(DATA_FILE):
        print(f"📂 Found local library! Loading {DATA_FILE}...")
        return torch.load(DATA_FILE)

    print(f"🛰️ Library not found. Harvesting {TARGET_COUNT} universes from KS Database...")
    x_list, y_list = [], []
    
    # Using the docs: as_list=False for streaming efficiency
    g = fetch_polytopes(lattice="N", limit=TARGET_COUNT, as_list=False)
    
    for i, p in enumerate(g):
        try:
            h11, h21 = p.h11(lattice="N"), p.h21(lattice="N")
            verts = p.vertices()
            if verts.shape[0] <= 36:
                padded = np.zeros((36, 4))
                padded[:verts.shape[0], :] = verts
                x_list.append(padded)
                y_list.append([h11, h21])
        except: continue
        
        if i % 1000 == 0 and i > 0:
            print(f"✅ Processed {i} shapes...")

    X = torch.tensor(np.array(x_list), dtype=torch.float32)
    Y = torch.tensor(np.array(y_list), dtype=torch.float32)
    torch.save((X, Y), DATA_FILE)
    return X, Y

# ==========================================
# STAGE 2: THE BRAIN (BottNet)
# ==========================================
class BottNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(36 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x.view(-1, 36 * 4))

# ==========================================
# STAGE 3: ATTACK
# ==========================================
X, Y = get_data()
X, Y = X.to(device), Y.to(device)

model = BottNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

print(f"\n🔥 M2 Max ignited. Training on {len(X)} samples...")

try:
    for epoch in range(2001):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), MODEL_FILE)
except KeyboardInterrupt:
    print("\n🛑 Saving progress...")
    torch.save(model.state_dict(), MODEL_FILE)

print("🎯 Mission Complete. You have a trained String Landscape Radar.")

