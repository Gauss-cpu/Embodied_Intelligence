import json
import os
import math
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/cmu-mocap/data"
OUT_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/processed_legs_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parse_single_bvh(bvh_path):
    try:
        with open(bvh_path, 'r') as f:
            lines = f.readlines()

        joints = {}
        current_joint = None
        channel_count = 0

        for line in lines:
            line = line.strip()
            if line.startswith("ROOT") or line.startswith("JOINT"):
                current_joint = line.split()[1]
                joints[current_joint] = {"offset": np.zeros(3), "channels": [], "start_idx": channel_count}
            elif line.startswith("OFFSET"):
                if current_joint:
                    joints[current_joint]["offset"] = np.array([float(x) for x in line.split()[1:]])
            elif line.startswith("CHANNELS"):
                parts = line.split()
                num_channels = int(parts[1])
                ch_names = parts[2:]
                if current_joint:
                    joints[current_joint]["channels"] = ch_names
                    joints[current_joint]["start_idx"] = channel_count
                    channel_count += num_channels
            elif line == "MOTION":
                break

        # 加上脚踝
        target_joints = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
        for t in target_joints:
            if t not in joints:
                return None

        motion_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "MOTION":
                motion_start_idx = i
                break

        frame_time = float(lines[motion_start_idx + 3].split()[2])
        q_traj = []

        def get_val(joint_name, axis_name, parts):
            j = joints[joint_name]
            if axis_name in j["channels"]:
                idx = j["channels"].index(axis_name)
                return float(parts[j["start_idx"] + idx])
            return 0.0

        for i in range(motion_start_idx + 4, len(lines)):
            line = lines[i].strip()
            if not line: continue
            parts = line.split()
            if len(parts) < channel_count: continue

            # Hips
            R_root = rot_z(math.radians(get_val("Hips", "Zrotation", parts))) @ \
                     rot_x(math.radians(get_val("Hips", "Xrotation", parts))) @ \
                     rot_y(math.radians(get_val("Hips", "Yrotation", parts)))

            frame_data = {"Hips": [0.0, 0.0, 0.0]}

            # LeftUpLeg
            R_lu = R_root @ (rot_z(math.radians(get_val("LeftUpLeg", "Zrotation", parts))) @ \
                             rot_x(math.radians(get_val("LeftUpLeg", "Xrotation", parts))) @ \
                             rot_y(math.radians(get_val("LeftUpLeg", "Yrotation", parts))))
            pos_lu = R_root @ joints["LeftUpLeg"]["offset"]
            frame_data["LeftUpLeg"] = pos_lu.tolist()

            # LeftLeg
            R_ll = R_lu @ (rot_z(math.radians(get_val("LeftLeg", "Zrotation", parts))) @ \
                           rot_x(math.radians(get_val("LeftLeg", "Xrotation", parts))) @ \
                           rot_y(math.radians(get_val("LeftLeg", "Yrotation", parts))))
            pos_ll = pos_lu + R_lu @ joints["LeftLeg"]["offset"]
            frame_data["LeftLeg"] = pos_ll.tolist()

            # LeftFoot
            R_lf = R_ll @ (rot_z(math.radians(get_val("LeftFoot", "Zrotation", parts))) @ \
                           rot_x(math.radians(get_val("LeftFoot", "Xrotation", parts))) @ \
                           rot_y(math.radians(get_val("LeftFoot", "Yrotation", parts))))
            pos_lf = pos_ll + R_ll @ joints["LeftFoot"]["offset"]
            frame_data["LeftFoot"] = pos_lf.tolist()

            # RightUpLeg
            R_ru = R_root @ (rot_z(math.radians(get_val("RightUpLeg", "Zrotation", parts))) @ \
                             rot_x(math.radians(get_val("RightUpLeg", "Xrotation", parts))) @ \
                             rot_y(math.radians(get_val("RightUpLeg", "Yrotation", parts))))
            pos_ru = R_root @ joints["RightUpLeg"]["offset"]
            frame_data["RightUpLeg"] = pos_ru.tolist()

            # RightLeg
            R_rl = R_ru @ (rot_z(math.radians(get_val("RightLeg", "Zrotation", parts))) @ \
                           rot_x(math.radians(get_val("RightLeg", "Xrotation", parts))) @ \
                           rot_y(math.radians(get_val("RightLeg", "Yrotation", parts))))
            pos_rl = pos_ru + R_ru @ joints["RightLeg"]["offset"]
            frame_data["RightLeg"] = pos_rl.tolist()

            # RightFoot
            R_rf = R_rl @ (rot_z(math.radians(get_val("RightFoot", "Zrotation", parts))) @ \
                           rot_x(math.radians(get_val("RightFoot", "Xrotation", parts))) @ \
                           rot_y(math.radians(get_val("RightFoot", "Yrotation", parts))))
            pos_rf = pos_rl + R_rl @ joints["RightFoot"]["offset"]
            frame_data["RightFoot"] = pos_rf.tolist()

            q_traj.append(frame_data)

        out_data = {
            "source": bvh_path.name,
            "fps": 1 / frame_time,
            "total_frames": len(q_traj),
            "trajectory": q_traj
        }

        subject = bvh_path.parent.name
        out_subject_dir = OUT_DIR / subject
        out_subject_dir.mkdir(exist_ok=True)

        with open(out_subject_dir / f"{bvh_path.stem}.json", "w") as f:
            json.dump(out_data, f)

        return {"subject": subject, "file": bvh_path.name, "frames": len(q_traj)}
    except Exception as e:
        return {"subject": bvh_path.parent.name, "file": bvh_path.name, "error": str(e)}


if __name__ == "__main__":
    bvh_files = list(BASE_DIR.rglob("*.bvh"))
    print(f"共找到 {len(bvh_files)} 个文件，开始提取完整 3D 关节...")
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, res in enumerate(executor.map(parse_single_bvh, bvh_files)):
            if res: results.append(res)
            if (i + 1) % 200 == 0: print(f"进度: {i + 1}/{len(bvh_files)}")
    print(f"全部完成！数据在 {OUT_DIR}")