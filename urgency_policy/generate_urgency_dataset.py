import pandas as pd

# 精心构造的高质量数据集：事件 -> 紧急度 (0.0 ~ 1.0)
data = [
    # --- 悠闲散步 (0.0 - 0.2) ---
    ("I am strolling in the park", 0.05),
    ("Taking a leisurely walk", 0.1),
    ("Enjoying the beautiful scenery", 0.0),
    ("Walking my dog slowly", 0.15),
    ("Window shopping on the weekend", 0.1),
    ("Listening to music while walking", 0.05),
    ("Walking barefoot on the grass", 0.0),
    ("Taking a slow walk after dinner", 0.1),
    ("Walking around the neighborhood", 0.15),
    ("Casually exploring the city", 0.1),
    ("Looking at the flowers", 0.0),
    ("Walking in the rain with an umbrella", 0.2),
    ("Going for a morning stroll", 0.1),
    ("Walking the dog in the park", 0.15),
    ("Wandering through the forest", 0.05),
    ("Walking slowly to enjoy the breeze", 0.0),
    ("Browsing a bookstore", 0.1),
    ("Walking to the kitchen for water", 0.05),
    ("Walking from the bedroom to the living room", 0.0),
    ("Taking a relaxed walk on the beach", 0.05),

    # --- 正常通勤 / 稍微赶时间 (0.3 - 0.5) ---
    ("Walking to the subway station", 0.4),
    ("Going to school in the morning", 0.35),
    ("Heading to the office", 0.3),
    ("Looking for a restroom", 0.45),
    ("Walking fast to catch the elevator", 0.5),
    ("Trying to find a lost key", 0.4),
    ("Slightly late for a meeting", 0.5),
    ("Carrying heavy groceries home", 0.4),
    ("Walking to the bus stop before it leaves", 0.5),
    ("Heading to the cafeteria for lunch", 0.3),
    ("Going to the gym", 0.25),
    ("Walking across the crosswalk before the light turns red", 0.45),
    ("Moving through a crowded street", 0.4),
    ("Trying to find a specific store", 0.3),
    ("Walking with a purpose", 0.35),
    ("Going to meet a friend", 0.25),
    ("Walking to the train platform", 0.4),
    ("Carrying luggage at the airport", 0.4),
    ("Navigating through the mall", 0.3),
    ("Going to grab a coffee", 0.25),

    # --- 明显着急 / 轻度奔跑 (0.6 - 0.8) ---
    ("I am late for my exam!", 0.85),
    ("Running to catch the last train", 0.8),
    ("Hurry up, we are running out of time", 0.75),
    ("Sprinting to the bus stop", 0.7),
    ("Slightly jogging to warm up", 0.6),
    ("Running to deliver an urgent document", 0.75),
    ("Late for a job interview", 0.8),
    ("Trying to catch the departing bus", 0.7),
    ("Rushing to the hospital for a mild injury", 0.8),
    ("Escaping sudden heavy rain", 0.6),
    ("Running because the meeting started 5 minutes ago", 0.75),
    ("Chasing a taxi", 0.7),
    ("Running late for the flight", 0.8),
    ("Hurrying to the restroom", 0.65),
    ("Running to grab the phone before it stops ringing", 0.6),
    ("Rushing to finish a project before the deadline", 0.7),
    ("Jogging fast because I overslept", 0.8),
    ("Running to catch the package delivery", 0.65),
    ("Hurrying to pick up a child from school", 0.7),
    ("Moving rapidly through the crowd", 0.6),

    # --- 极度恐慌 / 逃命 (0.9 - 1.0) ---
    ("A dog is chasing me!", 0.99),
    ("Fire alarm is ringing, need to evacuate", 0.95),
    ("Someone is chasing me", 0.98),
    ("Running for my life!", 1.0),
    ("Earthquake! Need to get outside", 1.0),
    ("The building is on fire", 1.0),
    ("Being followed by a stranger", 0.95),
    ("Escaping from a dangerous animal", 1.0),
    ("Tornado approaching, take cover", 1.0),
    ("Gunshots heard nearby, run!", 1.0),
    ("Emergency evacuation due to gas leak", 0.98),
    ("Zombie apocalypse, run!", 1.0),
    ("Robbery happening, flee the scene", 0.95),
    ("Tsunami warning, run to higher ground", 1.0),
    ("Being chased by an angry mob", 0.98),
    ("Running away from an explosion", 1.0),
    ("Flood is coming rapidly", 0.95),
    ("Wildfire spreading towards me", 1.0),
    ("Escaping a collapsing structure", 1.0),
    ("Sprinting to save someone's life", 1.0),
]

# 增加一些日常变化的模板，让数据更平滑
templates = [
    ("I am {} to the store", 0.3),
    ("I am {} in the park", 0.1),
    ("{} to the office", 0.4),
    ("{} away from danger", 0.95),
    ("{} towards the exit", 0.7),
    ("{} to catch the bus", 0.6),
]

verbs = ["walking slowly", "jogging", "running fast", "sprinting", "hurrying", "rushing", "moving quickly", "strolling"]

for t, base in templates:
    for v in verbs:
        if "slowly" in v or "strolling" in v:
            score = min(base, 0.2)
        elif "sprinting" in v or "running fast" in v:
            score = min(base + 0.4, 1.0)
        elif "jogging" in v or "moving quickly" in v:
            score = min(base + 0.2, 0.9)
        else:
            score = base
        data.append((t.format(v), round(score, 2)))

# 去重并保存为 CSV
df = pd.DataFrame(data, columns=["event", "urgency"])
df = df.drop_duplicates(subset=["event"]).reset_index(drop=True)
df.to_csv("urgency_dataset.csv", index=False)

print(f"✅ 成功生成数据集，共 {len(df)} 条数据，已保存至 urgency_dataset.csv")