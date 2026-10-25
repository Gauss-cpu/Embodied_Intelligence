import pandas as pd
import re
import numpy as np

df = pd.read_parquet("hf://datasets/Holmeister/AEB_2_FT/data/test-00000-of-00001.parquet")

def extract_snippet(instruction):
    """支持 Snippet: / Tweet: / 引号三种格式"""
    for keyword in ["Snippet:", "snippet:", "Tweet:", "tweet:"]:
        idx = instruction.find(keyword)
        if idx != -1:
            text = instruction[idx + len(keyword):].strip()
            # 去掉尾部的 "Intensity score:" 之类
            for tail in ["Intensity score:", "Score:", "score:", "Rating:"]:
                tidx = text.find(tail)
                if tidx != -1:
                    text = text[:tidx].strip()
            if text:
                return text
    # 兜底：引号里的内容
    m = re.search(r'"([^"]+)"', instruction)
    if m:
        return m.group(1).strip()
    return None

def parse_output(s):
    """支持 '1.5' 和 '-2: very negative' 两种格式"""
    if s is None:
        return None
    s = str(s).strip()
    # 尝试直接转
    try:
        return float(s)
    except:
        pass
    # 处理 '-2: very negative' 格式
    m = re.match(r'^\s*(-?\d+(?:\.\d+)?)', s)
    if m:
        return float(m.group(1))
    return None

df["text"] = df["instruction"].apply(extract_snippet)
df["raw_score"] = df["output"].apply(parse_output)

print(f"总行数: {len(df)}")
print(f"text 提取失败: {df['text'].isna().sum()}")
print(f"score 解析失败: {df['raw_score'].isna().sum()}")
print(f"两者都成功: {(df['text'].notna() & df['raw_score'].notna()).sum()}")

# 丢掉失败行
df = df.dropna(subset=["text", "raw_score"]).reset_index(drop=True)

# 看 raw_score 的分布
print(f"\nraw_score 分布:")
print(df["raw_score"].describe())

# 归一化到 [-1, 1]
# 但注意——现在数据是两种格式混合：
#   A 格式：output = -2.1 ~ +2.85（大概 ±3 范围）
#   B 格式：output = -2 ~ +1（整数，来自 "-2: very negative" 这种）
# 先看实际范围
raw_min, raw_max = df["raw_score"].min(), df["raw_score"].max()
print(f"\n原始范围: [{raw_min:.2f}, {raw_max:.2f}]")

SCALE = 3.0
df["valence"] = (df["raw_score"] / SCALE).clip(-1.0, 1.0)

# 清理
df = df[["text", "raw_score", "valence"]].reset_index(drop=True)

print(f"\n清理后: {len(df)} 行")
print(f"\nvalence 统计:")
print(df["valence"].describe())
print(f"\n前 8 条:")
for i in range(min(8, len(df))):
    row = df.iloc[i]
    print(f"  [{row['valence']:+.2f}] {row['text'][:80]}")

df.to_csv("events.csv", index=False)
print(f"\n已保存: events.csv")