import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

# 1. Setup Device (M2 Max GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Training on: {device}")

# 2. Load Data
data = torch.load("polytope5_tensors.pt", weights_only=False)
X, Y = data["X"], data["Y"]
dataset = TensorDataset(X, Y)

# Split 80/20
train_idx, val_idx = random_split(dataset, [396, 99])
train_loader = DataLoader(train_idx, batch_size=16, shuffle=True)
val_loader = DataLoader(val_idx, batch_size=16)

# 3. The Model
class IntersectionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IntersectionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512, output_dim) # Predicting 165 couplings
        )
    
    def forward(self, x):
        return self.net(x)

model = IntersectionNet(data["input_dim"], data["output_dim"]).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 4. Training Loop
print("🏋️  Starting training...")
for epoch in range(100):
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
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += criterion(v_out, vy).item()
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "intersection_model.pth")
print("\n✅ Training Complete. Model saved as intersection_model.pth")