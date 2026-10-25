import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

df = pd.read_csv('events.csv')
texts = df['text'].tolist()
labels = df['valence'].values.astype(np.float32)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True)
print(f"embeddings shape: {embeddings.shape}")

x_train, x_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.15, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

print(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

class ValenceNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        out = self.net(x)
        return out.view(out.size(0), -1)

device = torch.device('cpu')
model = ValenceNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 100
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)

    model.eval()
    with torch.no_grad():
        pred_val = model(x_val.to(device))
        val_loss = loss_fn(pred_val, y_val.to(device)).item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'valence_model.pt')
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

print(f"\n训练完成，最佳 val_loss = {best_val_loss:.4f}")
print("已保存: valence_model.pt")
print("已保存: （sentence-transformer 编码器不用重训，直接复用）")