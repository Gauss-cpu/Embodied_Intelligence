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

# ==================== 配置 ====================
DATA_PATH = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/features/leg_dataset.npz"
MODEL_DIR = Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 2048  # 400万数据，可以开大一点
EPOCHS = 20
LR = 1e-3

print(f"使用设备: {DEVICE}")

# ==================== 1. 加载并标准化数据 ====================
print("加载数据...")
data = np.load(DATA_PATH, allow_pickle=True)
X_raw = data["X"]
Y_raw = data["Y"]

# 400万数据如果太大，为了快速验证，你可以先截取前 50 万条测试
# X_raw = X_raw[:500000]
# Y_raw = Y_raw[:500000]

print(f"原始 X: {X_raw.shape}, Y: {Y_raw.shape}")

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(
    X_raw, Y_raw, test_size=0.1, random_state=42
)

# 标准化 (非常重要)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_Y = StandardScaler()
Y_train = scaler_Y.fit_transform(Y_train)
Y_val = scaler_Y.transform(Y_val)

# 保存标准化器，供以后物理引擎推理时使用
with open(MODEL_DIR / "scalers.pkl", "wb") as f:
    pickle.dump({"scaler_X": scaler_X, "scaler_Y": scaler_Y}, f)

# 转成 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=BATCH_SIZE)


# ==================== 2. 定义 DNN 模型 ====================
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

print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# ==================== 3. 训练循环 ====================
train_losses = []
val_losses = []

print("开始训练...")
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

    # 验证
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

    print(f"Epoch {epoch + 1:2d} | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f} | 耗时: {time.time() - t0:.1f}s")

# ==================== 4. 保存模型 ====================
torch.save(model.state_dict(), MODEL_DIR / "dnn_torque_model.pt")
print("已保存模型: dnn_torque_model.pt")

# ==================== 5. 画 Loss 曲线 ====================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("DNN 行为克隆训练曲线")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (标准化后)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()