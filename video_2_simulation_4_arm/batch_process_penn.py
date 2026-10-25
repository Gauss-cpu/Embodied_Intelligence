import json
import numpy as np
import math
import csv
import time
from pathlib import Path

# ==================== 配置 ====================
PENN_ROOT = Path.home() / "Desktop/Physical_Intelligence/dataset_raw_penn/OpenDataLab___Penn_Action/penn_extracted/0.3"
OUT_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
OUT_DIR.mkdir(exist_ok=True)

TARGET_CATS = [1, 2, 5, 7, 9, 10, 11, 12, 13, 14]
CAT_NAMES = {
    1: "baseball_pitch", 2: "pullup", 5: "strum_guitar",
    7: "tennis_serve", 9: "tennis_forehand", 10: "bowl",
    11: "jumping_jacks", 12: "baseball_swing",
    13: "clean_and_jerk", 14: "golf_swing",
}
VIDEOS_PER_CAT = 50
F_PIXELS = 550.0
H_REAL = 1.75
IMG_W, IMG_H = 480, 270

# ==================== 关键点索引 ====================
KP_HEAD = 0
KP_SHOULDER_L, KP_SHOULDER_R = 1, 2
KP_ELBOW_L, KP_ELBOW_R = 3, 4
KP_WRIST_L, KP_WRIST_R = 5, 6
KP_ANKLE_L, KP_ANKLE_R = 11, 12


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0, 0, -1.0])


def process_video(frames):
    if len(frames) < 10:
        return None

    q_frames = []
    prev_q1 = 0.0

    for i, fr in enumerate(frames):
        # 找 keypoints2d
        kps = None
        for item in fr["ground_truth"]:
            if item.get("type") == "keypoints2d":
                kps = item["points"]
                break
        if kps is None:
            continue

        sh = kps[KP_SHOULDER_R]
        el = kps[KP_ELBOW_R]
        wr = kps[KP_WRIST_R]
        head = kps[KP_HEAD]
        ankle_l = kps[KP_ANKLE_L]
        ankle_r = kps[KP_ANKLE_R]

        # vis 过滤：只保留完全可见帧
        if sh[2] < 2 or el[2] < 2 or wr[2] < 2:
            continue

        # 尺度锚定：头顶到脚踝
        head_y = head[1]
        feet_y = (ankle_l[1] + ankle_r[1]) / 2.0
        H_pixel = feet_y - head_y
        if H_pixel < 30:
            continue

        Z_person = F_PIXELS * H_REAL / H_pixel

        # 平面假设：整条手臂在同一深度
        def kp_to_3d(kp):
            Z = Z_person
            X = (kp[0] - IMG_W / 2) * Z / F_PIXELS
            Y = (kp[1] - IMG_H / 2) * Z / F_PIXELS
            return np.array([X, Y, Z])

        p_sh = kp_to_3d(sh)
        p_el = kp_to_3d(el)
        p_wr = kp_to_3d(wr)

        upper_dir = norm(p_el - p_sh)
        fore_dir = norm(p_wr - p_el)

        # 视频坐标 → arm3d 坐标
        up3d = norm(np.array([upper_dir[2], upper_dir[0], -upper_dir[1]]))
        fo3d = norm(np.array([fore_dir[2], fore_dir[0], -fore_dir[1]]))

        q2 = math.acos(np.clip(-up3d[2], -1, 1))
        sin_q2 = math.sqrt(up3d[0] ** 2 + up3d[1] ** 2)
        if sin_q2 < 0.15:
            q1 = prev_q1
        else:
            q1 = math.atan2(up3d[1], up3d[0])
        prev_q1 = q1
        q4 = math.acos(np.clip(np.dot(up3d, fo3d), -1, 1))

        q_frames.append({
            "frame": i,
            "t": i / 30.0,
            "q": [q1, q2, 0.0, q4, 0.0, 0.0, 0.0, 0.0],
        })

    if len(q_frames) < 10:
        return None

    # q1 unwrap
    q1_arr = np.unwrap(np.array([qf["q"][0] for qf in q_frames]))
    for j, qf in enumerate(q_frames):
        qf["q"][0] = float(q1_arr[j])

    return {
        "fps": 30.0,
        "valid_frames": len(q_frames),
        "trajectory": q_frames,
    }


# ==================== 主流程 ====================
print("加载 train.json...")
with open(PENN_ROOT / "annotations/json/train.json") as f:
    train_data = json.load(f)
print(f"train 视频数: {len(train_data['samples'])}")

print("加载 test.json...")
with open(PENN_ROOT / "annotations/json/test.json") as f:
    test_data = json.load(f)
print(f"test 视频数: {len(test_data['samples'])}")

all_videos = train_data["samples"] + test_data["samples"]
print(f"合并后: {len(all_videos)} 个视频\n")

# 按类别分组
from collections import defaultdict

by_cat = defaultdict(list)
for video in all_videos:
    if not video:
        continue
    first = video[0]
    cat_id = None
    for item in first["ground_truth"]:
        if "categories" in item:
            cat_id = item["categories"][0]["category_id"]
            break
    if cat_id in TARGET_CATS:
        src = first["media"]["source"]
        vid_id = src.split("/")[1]
        by_cat[cat_id].append((vid_id, video))

print("各目标类别的视频数:")
for cat in TARGET_CATS:
    print(f"  {cat:2d} {CAT_NAMES[cat]:<20s}: {len(by_cat[cat])}")
print()

# 处理
index_rows = []
t_start = time.time()

for cat in TARGET_CATS:
    cat_name = CAT_NAMES[cat]
    out_cat = OUT_DIR / f"penn_{cat_name}"
    out_cat.mkdir(exist_ok=True)

    videos = by_cat[cat][:VIDEOS_PER_CAT]
    print(f"\n=== {cat_name} ({len(videos)} 个视频) ===")

    for vi, (vid_id, video) in enumerate(videos):
        out_file = out_cat / f"video_{vid_id}_q.json"
        if out_file.exists():
            print(f"  [{vi+1}/{len(videos)}] {vid_id}  已存在，跳过")
            continue

        t0 = time.time()
        try:
            result = process_video(video)
            if result:
                result["video_id"] = vid_id
                result["category"] = cat_name
                result["category_id"] = cat
                with open(out_file, "w") as f:
                    json.dump(result, f)
                index_rows.append({
                    "category": cat_name, "category_id": cat,
                    "video_id": vid_id, "frames": result["valid_frames"],
                    "json": str(out_file),
                })
                dt = time.time() - t0
                print(f"  [{vi+1}/{len(videos)}] {vid_id}  "
                      f"{result['valid_frames']} 帧  {dt:.1f}s")
            else:
                print(f"  [{vi+1}/{len(videos)}] {vid_id}  失败")
        except Exception as e:
            print(f"  [{vi+1}/{len(videos)}] {vid_id}  错误: {e}")

# 索引
index_path = OUT_DIR / "dataset_index.csv"
with open(index_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "category_id",
                                            "video_id", "frames", "json"])
    writer.writeheader()
    writer.writerows(index_rows)

elapsed = (time.time() - t_start) / 60
print(f"\n总耗时: {elapsed:.1f} 分钟")
print(f"成功: {len(index_rows)} 个视频")
print(f"索引: {index_path}")