import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# --- CONFIG ---
DATA_FILE = "balanced_landscape.pt"
MODEL_FILE = "bott_net_balanced.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load the Stratified Data
X, Y = torch.load(DATA_FILE)
X, Y = X.to(device), Y.to(device)

# Split into Train (90%) and Validation (10%)
n_train = int(0.9 * len(X))
X_train, X_val = X[:n_train], X[n_train:]
Y_train, Y_val = Y[:n_train], Y[n_train:]

# 2. Re-Initialize Model (Fresh Brain)
class BottNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(36 * 4, 1024), # Widened for complex patterns
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),         # Prevents memorization
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x.view(-1, 36 * 4))

model = BottNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()

print(f"🔥 Training on Stratified Data ({len(X_train)} samples)...")

# 3. Training Loop
for epoch in range(1001):
    model.train()
    optimizer.zero_grad()
    
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        # Check Validation Accuracy
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val)
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

torch.save(model.state_dict(), MODEL_FILE)
print("🎯 Balanced training complete.")

