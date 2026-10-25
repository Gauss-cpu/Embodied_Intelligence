import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================== 配置 ====================
DATASET = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
EXCLUDE = {"strum_guitar"}
MIN_FRAMES = 15


# ==================== 特征提取 ====================
def extract_features(traj):
    """
    21 维统计特征：q1 (偏航)、q2 (抬升)、q4 (肘) 各 7 个统计量。
    """
    q1 = np.array([qf["q"][0] for qf in traj])
    q2 = np.array([qf["q"][1] for qf in traj])
    q4 = np.array([qf["q"][3] for qf in traj])

    feats = []
    for q in [q1, q2, q4]:
        dq = np.gradient(q)
        ddq = np.gradient(dq)
        feats.extend([
            q.mean(),
            q.std(),
            q.max(),
            q.min(),
            q.max() - q.min(),         # peak-to-peak
            np.abs(dq).mean(),          # 平均速度
            np.abs(ddq).mean(),         # 平均加速度
        ])
    return np.array(feats, dtype=np.float32)


# ==================== 加载 + 提特征 ====================
print("加载数据...")
X_list = []
y_list = []
labels_set = set()

for cat_dir in sorted(DATASET.glob("penn_*")):
    cat_name = cat_dir.name.replace("penn_", "")
    if cat_name in EXCLUDE:
        continue
    for json_file in cat_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        traj = data["trajectory"]
        if len(traj) < MIN_FRAMES:
            continue
        X_list.append(extract_features(traj))
        y_list.append(cat_name)
        labels_set.add(cat_name)

labels_list = sorted(labels_set)
label_to_idx = {l: i for i, l in enumerate(labels_list)}
idx_to_label = {i: l for l, i in label_to_idx.items()}
n_classes = len(labels_list)

X = np.stack(X_list)
y = np.array([label_to_idx[c] for c in y_list], dtype=np.int64)

print(f"总样本: {len(X)}")
print(f"特征维度: {X.shape[1]}")
print(f"类别数: {n_classes}\n")

for cat in labels_list:
    cnt = sum(1 for c in y_list if c == cat)
    print(f"  {cat:<20s}: {cnt}")


# ==================== 划分 ====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {len(X_train)}, 验证集: {len(X_val)}\n")


# ==================== 训练两个模型 ====================
print("=" * 50)
print("随机森林")
print("=" * 50)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_val_acc = rf.score(X_val, y_val)
rf_train_acc = rf.score(X_train, y_train)
print(f"训练准确率: {rf_train_acc:.3f}")
print(f"验证准确率: {rf_val_acc:.3f}")


print("\n" + "=" * 50)
print("Gradient Boosting")
print("=" * 50)
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)
gb.fit(X_train, y_train)
gb_val_acc = gb.score(X_val, y_val)
gb_train_acc = gb.score(X_train, y_train)
print(f"训练准确率: {gb_train_acc:.3f}")
print(f"验证准确率: {gb_val_acc:.3f}")


# ==================== 选最好的 ====================
if rf_val_acc >= gb_val_acc:
    best_model = rf
    best_name = "RandomForest"
    best_acc = rf_val_acc
else:
    best_model = gb
    best_name = "GradientBoosting"
    best_acc = gb_val_acc

print(f"\n★ 最佳模型: {best_name}  验证准确率 = {best_acc:.3f}")


# ==================== 混淆矩阵 ====================
y_pred = best_model.predict(X_val)
print("\n分类报告:")
print(classification_report(y_val, y_pred, target_names=labels_list))

print("混淆矩阵:")
cm = confusion_matrix(y_val, y_pred)
print(f"{'':>16s}", end="")
for lab in labels_list:
    print(f"{lab[:6]:>8s}", end="")
print()
for i, lab in enumerate(labels_list):
    print(f"{lab:>16s}", end="")
    for j in range(len(labels_list)):
        print(f"{cm[i, j]:>8d}", end="")
    print()


# ==================== 特征重要性 ====================
print("\n特征重要性 (RandomForest):")
feat_names = [
    "q1_mean", "q1_std", "q1_max", "q1_min", "q1_ptp", "q1_vel", "q1_acc",
    "q2_mean", "q2_std", "q2_max", "q2_min", "q2_ptp", "q2_vel", "q2_acc",
    "q4_mean", "q4_std", "q4_max", "q4_min", "q4_ptp", "q4_vel", "q4_acc",
]
importances = rf.feature_importances_
for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 100)
    print(f"  {name:>10s}: {imp:.4f}  {bar}")


# ==================== 保存 ====================
import pickle
with open("motion_classifier_rf.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feat_names": feat_names,
    }, f)
print(f"\n已保存: motion_classifier_rf.pkl")