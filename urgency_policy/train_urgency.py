import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("读取数据集...")
df = pd.read_csv("urgency_dataset.csv")
texts = df["event"].tolist()
labels = torch.tensor(df["urgency"].values, dtype=torch.float32).unsqueeze(1)

print("加载编码器...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("正在对文本进行编码...")
embeddings = encoder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

dataset = TensorDataset(embeddings, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


class UrgencyNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 0~1
        )

    def forward(self, x):
        return self.net(x)


device = "mps" if torch.backends.mps.is_available() else "cpu"
model = UrgencyNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print("开始训练...")
for epoch in range(500):
    total_loss = 0
    for batch_emb, batch_labels in dataloader:
        batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        preds = model(batch_emb)
        loss = criterion(preds, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "urgency_model.pth")
print("✅ 训练完成，模型已保存为 urgency_model.pth")