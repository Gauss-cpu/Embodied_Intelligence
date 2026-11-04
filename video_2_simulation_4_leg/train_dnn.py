import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time

DATA_PATH = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/features/leg_dataset.npz"
MODEL_DIR = Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 2048
EPOCHS = 20
LR = 1e-3

print(f"device used: {DEVICE}")

# 1. Uploading and primarily process the data
print("loading dataset...")
data = np.load(DATA_PATH, allow_pickle=True)
X_raw = data["X"]
Y_raw = data["Y"]


print(f"Original X: {X_raw.shape}, Y: {Y_raw.shape}")

X_train, X_val, Y_train, Y_val = train_test_split(
    X_raw, Y_raw, test_size=0.1, random_state=42
)

# Standard Scaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_Y = StandardScaler()
Y_train = scaler_Y.fit_transform(Y_train)
Y_val = scaler_Y.transform(Y_val)

with open(MODEL_DIR / "scalers.pkl", "wb") as f:
    pickle.dump({"scaler_X": scaler_X, "scaler_Y": scaler_Y}, f)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=BATCH_SIZE)


# 2. Defining the DNN model
class TorqueNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


model = TorqueNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"number of model parameters: {sum(p.numel() for p in model.parameters())}")

# 3. training epoch
train_losses = []
val_losses = []

print("starting to train...")
for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(xb)

    tr_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            val_loss += criterion(pred, yb).item() * len(xb)
    val_loss /= len(val_loader.dataset)

    train_losses.append(tr_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1:2d} | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f} | time used: {time.time() - t0:.1f}s")

# 4. Saving the model
torch.save(model.state_dict(), MODEL_DIR / "dnn_torque_model.pt")
print("saved: dnn_torque_model.pt")

# 5. Drawing out the loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("DNN 行为克隆训练曲线")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()