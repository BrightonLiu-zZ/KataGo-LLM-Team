import json
import sys
from pathlib import Path
try:
    from sgfmill import boards, ascii_boards
except ImportError:
    print("Error: sgfmill not found. Please run: pip install sgfmill")
    sys.exit(1)

# ================= 配置路径 =================
# 建议先跑 short 版本测试，没问题了再改成完整版
#INPUT_FILE = r"C:\git_repo\KataGo-LLM-Team\data\json_output_with_topk_short.jsonl"
INPUT_FILE = r"C:\git_repo\KataGo-LLM-Team\data\json_output_with_topk.jsonl" 

OUTPUT_FILE = r"C:\git_repo\KataGo-LLM-Team\data\training_ready_data.jsonl"
BOARD_SIZE = 9
# ===========================================

def parse_gtp_coord(coord_str, size=9):
    """
    解析 GTP 坐标 (如 "C7", "pass") 为 (row, col)。
    返回: (row, col) 或 None (如果是 pass 或无效值)
    """
    if not isinstance(coord_str, str): 
        return None
    coord_str = coord_str.strip().upper()
    if coord_str == "PASS":
        return None
    
    if len(coord_str) < 2: 
        return None

    col_char = coord_str[0]
    row_str = coord_str[1:]
    
    # SGF/GTP 坐标: A-J (跳过 I)
    cols = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if col_char not in cols:
        return None
    
    col = cols.index(col_char)
    try:
        row = int(row_str) - 1
    except ValueError:
        return None
        
    if 0 <= row < size and 0 <= col < size:
        return (row, col)
    return None

def render_board_with_coords(board):
    """
    自定义渲染 ASCII 棋盘，带坐标轴，方便 LLM 理解位置。
    """
    size = board.side
    lines = []
    # 顶部坐标轴
    lines.append("   A B C D E F G H J")
    
    for r in range(size - 1, -1, -1):
        row_chars = []
        for c in range(size):
            color = board.get(r, c)
            if color == 'b':
                row_chars.append("X") # 黑棋
            elif color == 'w':
                row_chars.append("O") # 白棋
            else:
                row_chars.append(".") # 空点
        # 行号 + 棋盘内容
        line_str = f" {r+1} " + " ".join(row_chars)
        lines.append(line_str)
    
    return "\n".join(lines)

def get_next_player(initial_stones, moves):
    """推断下一手颜色"""
    if not moves:
        # 如果没有历史着法，看是否有初始黑子
        has_black = any(s[0].upper() == 'B' for s in initial_stones)
        has_white = any(s[0].upper() == 'W' for s in initial_stones)
        # 如果只有黑子(让子棋)，通常白先；否则黑先
        if has_black and not has_white:
            return "W"
        return "B"
    else:
        # 取最后一手的反色
        last_color = moves[-1][0].upper()
        return "W" if last_color == "B" else "B"

def process_line(line_str):
    try:
        raw = json.loads(line_str)
    except json.JSONDecodeError:
        return None

    # 1. 初始化棋盘
    b = boards.Board(BOARD_SIZE)
    
    # 2. 放置初始子 (Batch Setup - 修复点 1)
    black_points = []
    white_points = []
    
    for item in raw.get("initialStones", []):
        color_str, coord_str = item
        pt = parse_gtp_coord(coord_str, BOARD_SIZE)
        if pt:
            if color_str.upper() == 'B':
                black_points.append(pt)
            else:
                white_points.append(pt)
    
    # 一次性应用 setup，避免报错
    if black_points or white_points:
        b.apply_setup(black_points, white_points, [])

    # 3. 模拟行棋 (Replay Moves - 修复点 2)
    game_moves = raw.get("moves", [])
    for item in game_moves:
        color_str, coord_str = item
        color = 'b' if color_str.upper() == 'B' else 'w'
        pt = parse_gtp_coord(coord_str, BOARD_SIZE)
        
        if pt:
            # 必须解包元组 (row, col)
            try:
                b.play(pt[0], pt[1], color)
            except ValueError:
                # 忽略非法着手（如打劫未找劫材），保持程序运行
                pass
        else:
            # Pass move: 棋盘状态不变，不需要调用 b.play
            pass

    # 4. 准备 Prompt
    next_p_code = get_next_player(raw.get("initialStones", []), game_moves)
    next_p_full = "Black" if next_p_code == "B" else "White"
    
    # 渲染带坐标的棋盘
    board_str = render_board_with_coords(b)
    
    # 格式化历史
    recent_moves = game_moves[-8:]
    # 将列表 ["B", "C7"] 转换为字符串 "B C7"
    formatted_moves = [f"{m[0]} {m[1]}" for m in recent_moves] 
    history_str = "; ".join(formatted_moves)
    if not history_str:
        history_str = "None (Start of game)"

    prompt_text = (
        f"[Current 9x9 Board State]\n"
        f"{board_str}\n\n"
        f"[Match History]\n"
        f"Last moves: {history_str}\n\n"
        f"[Instruction]\n"
        f"Player to move: {next_p_full}. \n"
        f"Analyze the board structure (territory, shapes, liberties). \n"
        f"Then, identify the best move coordinate (e.g. D4) and explain your reasoning."
    )
    # === 5. 提取新的全盘数据并生成合法坐标列表 ===
    katago_evals = raw.get("katago_evals", {})
    root_winrate = raw.get("root_winrate")
    root_scoreLead = raw.get("root_scoreLead")

    # 找出棋盘上所有的空点 (Valid Empty Coordinates)
    valid_empty_coords = []
    cols = "ABCDEFGHJ"
    for r in range(9):
        for c in range(9):
            if b.get(r, c) is None: # 如果这个点是空的
                valid_empty_coords.append(f"{cols[c]}{r+1}")
    
    # 把它加到我们给 LLM 的提示词末尾
    prompt_text += f"\n\nValid empty coordinates are: [{', '.join(valid_empty_coords)}]"

    if not katago_evals:
        return None

    return {
        "original_id": raw.get("id"),
        "user_prompt": prompt_text,
        "katago_evals": katago_evals,      # 传递完整的大字典
        "root_winrate": root_winrate,      # 传递当前盘面胜率
        "root_scoreLead": root_scoreLead   # 传递当前盘面目差
    }

def main():
    in_path = Path(INPUT_FILE)
    out_path = Path(OUTPUT_FILE)
    
    print(f"Reading from: {in_path}")
    print(f"Writing to:   {out_path}")
    
    count = 0
    errors = 0
    
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        
        for line_num, line in enumerate(fin):
            line = line.strip()
            if not line: continue
            
            try:
                processed = process_line(line)
                if processed:
                    fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                # 打印具体哪一行出错，方便调试
                print(f"[Warning] Failed at line {line_num}: {e}")
                errors += 1
                if errors > 10:
                    print("Too many errors, stopping early.")
                    break
                
    print(f"Done! Successfully processed {count} lines.")
    if errors > 0:
        print(f"Encountered {errors} errors.")

if __name__ == "__main__":
    main()