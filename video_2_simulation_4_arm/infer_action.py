import json
import pickle
import numpy as np
from pathlib import Path


# ==================== 加载分类器 ====================
with open("motion_classifier_rf.pkl", "rb") as f:
    bundle = pickle.load(f)

MODEL = bundle["model"]
LABEL_TO_IDX = bundle["label_to_idx"]
IDX_TO_LABEL = bundle["idx_to_label"]


# ==================== 特征提取（和训练时一致） ====================
def extract_features(traj):
    q1 = np.array([qf["q"][0] for qf in traj])
    q2 = np.array([qf["q"][1] for qf in traj])
    q4 = np.array([qf["q"][3] for qf in traj])

    feats = []
    for q in [q1, q2, q4]:
        dq = np.gradient(q)
        ddq = np.gradient(dq)
        feats.extend([
            q.mean(), q.std(), q.max(), q.min(),
            q.max() - q.min(),
            np.abs(dq).mean(),
            np.abs(ddq).mean(),
        ])
    return np.array(feats, dtype=np.float32)


# ==================== 推理 ====================
def predict_action(traj):
    """
    输入：trajectory (list of frame dicts, 每个有 'q' 字段)
    输出：(label, confidence, top3)
    """
    if len(traj) < 15:
        return None, 0.0, []

    feat = extract_features(traj).reshape(1, -1)

    # 预测
    proba = MODEL.predict_proba(feat)[0]
    best_idx = int(np.argmax(proba))
    best_label = IDX_TO_LABEL[best_idx]
    best_conf = float(proba[best_idx])

    # Top-3
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(IDX_TO_LABEL[i], float(proba[i])) for i in top3_idx]

    return best_label, best_conf, top3


def classify_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    traj = data["trajectory"]
    label, conf, top3 = predict_action(traj)
    return label, conf, top3, data.get("category", "?")


# ==================== 测试：扫一遍验证集 ====================
if __name__ == "__main__":
    DATASET = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
    EXCLUDE = {"strum_guitar"}

    correct = 0
    total = 0
    by_cat = {}

    print(f"{'真实类别':<20s} {'预测':<20s} {'置信度':>8s}")
    print("-" * 60)

    for cat_dir in sorted(DATASET.glob("penn_*")):
        cat_name = cat_dir.name.replace("penn_", "")
        if cat_name in EXCLUDE:
            continue

        cat_correct = 0
        cat_total = 0

        for json_file in sorted(cat_dir.glob("*.json"))[:10]:   # 每类抽 10 条测试
            label, conf, top3, true_cat = classify_json(json_file)
            if label is None:
                continue
            if label == true_cat:
                correct += 1
                cat_correct += 1
            total += 1
            cat_total += 1

            mark = "✓" if label == true_cat else "✗"
            print(f"{true_cat:<20s} {label:<20s} {conf:>8.3f}  {mark}")

        by_cat[cat_name] = (cat_correct, cat_total)

    print("-" * 60)
    print(f"总准确率: {correct}/{total} = {correct/total:.3f}\n")
    print("按类别准确率:")
    for cat, (c, t) in sorted(by_cat.items()):
        print(f"  {cat:<20s}: {c}/{t} = {c/t:.2f}")


    # ==================== 演示：给一条 JSON 输出动作 + 力矩 ====================
    print("\n" + "=" * 60)
    print("演示：对新轨迹做完整推理")
    print("=" * 60)

    # 挑一条没见过的
    test_path = sorted((DATASET / "penn_tennis_serve").glob("*.json"))[5]
    label, conf, top3, true = classify_json(test_path)

    print(f"输入: {test_path.name}")
    print(f"真实类别: {true}")
    print(f"预测类别: {label}  (置信度 {conf:.3f})")
    print(f"Top-3:")
    for l, p in top3:
        print(f"  {l:<20s}: {p:.3f}")