from pathlib import Path
import re
import matplotlib.pyplot as plt

# 修改为你的SGF目录
INPUT_DIR = Path(r"D:\katago_old\9x9_online_go_game\test")

move_pattern = re.compile(rb";\s*[BW]\[[^\]]*\]")

all_moves = []

for p in INPUT_DIR.glob("*.sgf"):
    try:
        data = p.read_bytes()
        moves = move_pattern.findall(data)
        all_moves.append(len(moves))
    except Exception:
        continue

plt.hist(all_moves, bins=50, edgecolor="black")
plt.xlabel("total moves")
plt.ylabel("frequency")
plt.title("total moves histrogram")
plt.show()
