import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# 1. Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Igniting Universal Training on: {device}")

# Allow numpy scalars to load
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 2. Load the Global Tensors
data = torch.load("universal_5k_tensors.pt", weights_only=False)
X, Y = data["X"], data["Y"]
dataset = TensorDataset(X, Y)

# Dynamic 80/20 Split (724 universes -> ~579 Train, ~145 Validate)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. The Universal Architecture
class UniversalIntersectionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UniversalIntersectionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Extra deep layer for the global dataset
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512, output_dim) # Predicting 364 global couplings
        )
    
    def forward(self, x):
        return self.net(x)

input_dim = data["input_dim"]
output_dim = data["output_dim"]

model = UniversalIntersectionNet(input_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Training Loop
print(f"🏋️  Training on {train_size} universes, validating on {val_size}...")
epochs = 150

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Print stats every 15 epochs
    if (epoch + 1) % 15 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += criterion(v_out, vy).item()
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

torch.save(model.state_dict(), "universal_model.pth")
print("\n✅ Universal Model Saved! The landscape has been mapped.")

