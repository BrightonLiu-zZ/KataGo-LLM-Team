from pathlib import Path
import re
import random

# 修改为你的SGF目录
INPUT_DIR = Path(r"D:\katago_old\9x9_online_go_game\test")

move_pattern = re.compile(rb";\s*[BW]\[[^\]]*\]")

# 建立 {步数: [文件路径,...]} 的字典
move_dict = {}

for p in INPUT_DIR.glob("*.sgf"):
    try:
        data = p.read_bytes()
        moves = move_pattern.findall(data)
        num_moves = len(moves)
        move_dict.setdefault(num_moves, []).append(p)
    except Exception:
        continue

# 输入目标步数
n = int(input("请输入要抽样的总步数 n: "))

if n in move_dict:
    f = random.choice(move_dict[n])
    print(f"随机抽到的棋谱: {f.name}")
else:
    print(f"没有找到总步数为 {n} 的棋谱")
