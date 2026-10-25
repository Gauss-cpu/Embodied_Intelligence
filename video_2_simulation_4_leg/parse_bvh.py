import json
import os
import math
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/cmu-mocap/data"
OUT_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/processed_legs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_single_bvh(bvh_path):
    try:
        with open(bvh_path, 'r') as f:
            lines = f.readlines()

        joint_channels = {}
        current_joint = None
        channel_count = 0

        for line in lines:
            line = line.strip()
            if line.startswith("JOINT") or line.startswith("ROOT"):
                current_joint = line.split()[1]
                joint_channels[current_joint] = {"start_idx": channel_count, "channels": []}
            elif line.startswith("CHANNELS"):
                parts = line.split()
                num_channels = int(parts[1])
                ch_names = parts[2:]
                if current_joint:
                    joint_channels[current_joint]["channels"] = ch_names
                    joint_channels[current_joint]["start_idx"] = channel_count
                    channel_count += num_channels
            elif line == "MOTION":
                break

        thigh_name = "LeftUpLeg"
        shank_name = "LeftLeg"

        if thigh_name not in joint_channels or shank_name not in joint_channels:
            return None

        def get_rot_idx(joint, axis="Xrotation"):
            if joint not in joint_channels: return -1
            chs = joint_channels[joint]["channels"]
            return joint_channels[joint]["start_idx"] + (chs.index(axis) if axis in chs else 0)

        thigh_idx = get_rot_idx(thigh_name)
        shank_idx = get_rot_idx(shank_name)

        motion_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "MOTION":
                motion_start_idx = i
                break

        frame_time = float(lines[motion_start_idx + 3].split()[2])
        q_traj = []

        for i in range(motion_start_idx + 4, len(lines)):
            line = lines[i].strip()
            if not line: continue
            parts = line.split()
            if len(parts) < channel_count: continue

            val_thigh = float(parts[thigh_idx])
            val_shank = float(parts[shank_idx])

            # 转弧度并调整正负号以符合我们的坐标系
            th1 = -math.radians(val_thigh)
            th2 = max(0.0, -math.radians(val_shank))  # 防止膝盖反折

            q_traj.append({
                "t": round(i * frame_time, 4),  # 这里直接用帧号*i
                "q": [th1, th2]
            })

        out_data = {
            "source": bvh_path.name,
            "fps": 1 / frame_time,
            "total_frames": len(q_traj),
            "trajectory": q_traj
        }

        # 保存路径按原文件夹分类
        subject = bvh_path.parent.name
        out_subject_dir = OUT_DIR / subject
        out_subject_dir.mkdir(exist_ok=True)

        out_file = out_subject_dir / f"{bvh_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(out_data, f, indent=2)

        return {"subject": subject, "file": bvh_path.name, "frames": len(q_traj)}
    except Exception as e:
        return {"subject": bvh_path.parent.name, "file": bvh_path.name, "error": str(e)}


if __name__ == "__main__":
    print("扫描所有 BVH 文件...")
    bvh_files = list(BASE_DIR.rglob("*.bvh"))
    print(f"共找到 {len(bvh_files)} 个 BVH 文件。开始并行处理...")

    results = []
    # 使用多进程加速
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, res in enumerate(executor.map(parse_single_bvh, bvh_files)):
            if res:
                results.append(res)
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} / {len(bvh_files)}")

    # 写索引
    index_path = OUT_DIR / "index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "file", "frames", "error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n全部完成！数据已保存至: {OUT_DIR}")