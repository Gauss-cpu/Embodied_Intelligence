import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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
        return self.net(x)

device = "mps" if torch.backends.mps.is_available() else "cpu"

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = ValenceNet().to(device)
model.load_state_dict(torch.load("valence_model.pt", map_location=device))
model.eval()

def predict_valence(text):
    """输入一个事件文本，输出 valence 分数（-1 到 +1）"""
    emb = encoder.encode([text], convert_to_tensor=True).to(device)
    with torch.no_grad():
        score = model(emb).item()
    return score

# 测试
test_events = [
    "I got promoted at work!",
    "My dog died yesterday.",
    "Just had a normal day.",
    "Someone insulted me in public.",
    "I won the lottery!",
    "I'm exhausted and can't sleep.",
]

print("测试推理：")
for event in test_events:
    v = predict_valence(event)
    bar = "█" * int(abs(v) * 20)
    sign = "+" if v > 0 else "-"
    print(f"  {v:+.3f} {sign}{bar}  {event}")

import torch.nn as nn

class PPOModulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, 1)

    def forward(self, x):
        return self.mu_head(self.shared(x))

_ppo = None
def modulate_scale(valence, arousal, fear):
    """情绪 → 动作缩放因子"""
    global _ppo
    if _ppo is None:
        _ppo = PPOModulator().to(device)
        _ppo.load_state_dict(torch.load("ppo_modulator.pt", map_location=device))
        _ppo.eval()
    with torch.no_grad():
        s = torch.tensor([[valence, arousal, fear]],
                         dtype=torch.float32, device=device)
        return float(_ppo(s).item())