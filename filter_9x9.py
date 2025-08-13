# filter_9x9_sgf.py
# Python 3.x
from pathlib import Path
import re, hashlib, shutil

# === 1) 修改为你的SGF所在文件夹 & 输出文件夹 ===
INPUT_DIR = Path(r"D:\katago_old\9x9_online_go_game")   # 这里改成你的SGF文件夹
OUTPUT_DIR = Path(r"D:\katago_old\9x9_online_go_game\output")   # 这里改成你想保存9路SGF的文件夹
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 2) 正则：匹配 9x9 棋盘 ===
# 兼容 SZ[9] 或 SZ[9:9]，允许空白；大小写不敏感
re_sz9 = re.compile(rb"SZ\s*\[\s*9(\s*:\s*9)?\s*\]", re.IGNORECASE)

# 可选：如果出现 GM 标签，则要求 GM[1]（围棋）
re_gm  = re.compile(rb"GM\s*\[\s*(\d+)\s*\]", re.IGNORECASE)

def is_go_game(data: bytes) -> bool:
    """若文件中出现 GM[]，则要求为 GM[1]；若未出现 GM，则默认当作围棋"""
    m = re_gm.search(data)
    if not m:
        return True
    try:
        return int(m.group(1)) == 1
    except Exception:
        return True

def is_9x9(data: bytes) -> bool:
    return re_sz9.search(data) is not None

def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

total, kept, skipped, bad = 0, 0, 0, 0
seen_hashes = set()

for p in INPUT_DIR.rglob("*.sgf"):
    total += 1
    try:
        data = p.read_bytes()
    except Exception:
        bad += 1
        continue

    # 粗检查：应当含有括号与节点
    if b"(" not in data or b")" not in data:
        bad += 1
        continue

    if not is_go_game(data):
        skipped += 1
        continue

    if not is_9x9(data):
        skipped += 1
        continue

    # 去重：按内容哈希保存，避免同名覆盖
    h = sha1_bytes(data)
    if h in seen_hashes:
        continue
    seen_hashes.add(h)

    out_path = OUTPUT_DIR / f"{h}.sgf"
    try:
        out_path.write_bytes(data)
        kept += 1
    except Exception:
        bad += 1

print(f"Scanned: {total}")
print(f"Kept 9x9: {kept}")
print(f"Skipped (non-9x9 / non-Go): {skipped}")
print(f"Bad / unreadable: {bad}")
print(f"Output folder: {OUTPUT_DIR.resolve()}")
