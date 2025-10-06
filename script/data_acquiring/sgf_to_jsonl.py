# pip install sgfmill before running
from pathlib import Path
import json
from sgfmill import sgf

SGF_DIR = Path(r"D:\katago_old\lizzie") # change to the folder where your sgf are located
OUT_JSONL = Path(r"D:\katago_old\lizzie\json_output.jsonl")  # path to the .jsonl file that you will give KataGO
STEP_INTERVAL = 1 # generate an request for each move in a game

def coords_to_gtp(move_tuple):
    if move_tuple is None:
        return "pass"
    row, col = move_tuple
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    col_letter = letters[col]   
    return f"{col_letter}{row+1}"

def parse_ab_point(p):
    # tuple/list of ints
    if isinstance(p, (tuple, list)) and len(p) == 2 \
       and all(isinstance(x, int) for x in p):
        return (p[0], p[1])

    # bytes -> str
    if isinstance(p, (bytes, bytearray)):
        p = p.decode('ascii', errors='ignore')

    # 'dd' style
    if isinstance(p, str) and len(p) >= 2:
        col = ord(p[0]) - ord('a')
        row = ord(p[1]) - ord('a')
        return (row, col)

    raise ValueError(f"Unrecognized AB point format: {p!r}")

def sgf_to_requests_one_file(path: Path):
    # change each sgf to several KataGO analysis request
    try:
        game = sgf.Sgf_game.from_bytes(path.read_bytes())
    except Exception as e:
        print(f"[Failed to parse] {path.name} -> {e}")
        return []

    size = game.get_size()
    if size != 9:
        # skip if board size is not 9
        return []

    root = game.get_root()
    rules = (root.get("RU") or "Chinese").strip()
    try:
        komi = float(root.get("KM")) if root.get("KM") is not None else 7.0
    except Exception:
        komi = 7.0

    initial_stones = []
    try:
        ha = int(root.get("HA")) if root.get("HA") is not None else 0
    except Exception:
        ha = 0
    if ha > 0:
        ab_list = root.get("AB") or []
        for item in ab_list:
            try:
                rc = parse_ab_point(item)
                initial_stones.append(["B", coords_to_gtp(rc)])
            except Exception:
                pass

    main_nodes = game.get_main_sequence()
    moves = []
    for node in main_nodes[1:]:
        color, move = node.get_move()  # color: 'b'/'w' or None, move: (row,col) or None(pass)
        if color is None:
            continue
        col = 'B' if color.lower() == 'b' else 'W'
        gtp = coords_to_gtp(move)
        moves.append([col, gtp])

    if len(moves) == 0:
        print(f"[no move] {path.name}")
        return []

    # create request for the situation after each move
    reqs = []
    for t in range(0, len(moves)+1, STEP_INTERVAL):
        req = {
            "id": f"{path.stem}_pos{t}",
            "boardXSize": size,
            "boardYSize": size,
            "rules": rules if rules else "Chinese",
            "komi": komi,
            "initialStones": initial_stones,
            "moves": moves[:t], 
            "analyzeTurns": [t], 
            "includePolicy": True,
            "numTopMoves": 10
        }
        reqs.append(req)
    print(f"[OK] {path.name} moves={len(moves)} -> requests={len(reqs)}")
    return reqs

def main():
    total_games, total_reqs = 0, 0
    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for p in SGF_DIR.glob("*.sgf"):
            total_games += 1
            reqs = sgf_to_requests_one_file(p)
            for r in reqs:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_reqs += len(reqs)
    print(f"scanned sgf: {total_games}；requested: {total_reqs} -> {OUT_JSONL}")

if __name__ == "__main__":
    main()