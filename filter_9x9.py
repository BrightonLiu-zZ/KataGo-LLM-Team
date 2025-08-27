from pathlib import Path
import re, hashlib, shutil

INPUT_DIR = Path(r"D:\katago_old\9x9_online_go_game")   # change it to the folder where your sgf files are located
OUTPUT_DIR = Path(r"D:\katago_old\9x9_online_go_game\output")   # change it to the folder where you want to save for sgf files of board size 9
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# match sgf of board size 9
re_sz9 = re.compile(rb"SZ\s*\[\s*9(\s*:\s*9)?\s*\]", re.IGNORECASE)

re_gm  = re.compile(rb"GM\s*\[\s*(\d+)\s*\]", re.IGNORECASE)

def is_go_game(data: bytes) -> bool:
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

    if b"(" not in data or b")" not in data:
        bad += 1
        continue

    if not is_go_game(data):
        skipped += 1
        continue

    if not is_9x9(data):
        skipped += 1
        continue

    # eliminate repeated file name using hashes
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
