import json
import copy
import os

# ================= 配置区 =================
# 请修改为你的实际路径
INPUT_FILE = r'C:\git_repo\KataGo-LLM-Team\data\training_data_quality_filtered.jsonl'
OUTPUT_FILE = r'C:\git_repo\KataGo-LLM-Team\data\training_data_augmented.jsonl'

# ⚠️ 关键设置：仅当 katago_all 代表 "绝对黑棋胜率" 时设为 True。
# 你确认是 Winrate of Player to Move，所以设为 False。
INVERT_WINRATE_ON_COLOR_FLIP = False 
# =========================================

# 围棋坐标配置 (9x9, 跳过 'I')
COLS = "ABCDEFGHJ"
COL_MAP = {c: i for i, c in enumerate(COLS)}

def parse_coord(coord_str):
    """解析坐标 (e.g., 'C4') -> (x, y) 0-based"""
    if not isinstance(coord_str, str): return None, None
    coord_str = coord_str.strip().upper()
    
    # 处理特殊情况
    if coord_str == "PASS": return None, None
    if len(coord_str) < 2: return None, None
    
    col_char = coord_str[0]
    if col_char not in COL_MAP: return None, None
    
    col = COL_MAP[col_char]
    try:
        # 围棋坐标 1-9 对应 index 0-8
        # 注意：这里假设输入已经是标准围棋坐标 (1在下, 9在上)
        # 如果原始数据是 1-based, 这里的 row index 需要是 row_num - 1
        row = int(coord_str[1:]) - 1
    except ValueError:
        return None, None
        
    return col, row

def to_coord_str(x, y):
    """(x, y) -> 'C4'"""
    if x is None or y is None: return "pass"
    if not (0 <= x < 9 and 0 <= y < 9): return "pass" # 越界保护
    return f"{COLS[x]}{y+1}"

def transform_xy(x, y, op):
    """
    对 (x,y) 应用 D4 群变换 (9x9)
    op: 0-7
    0: Identity
    1: Rot90 (顺时针)
    2: Rot180
    3: Rot270
    4: Flip Horizontal
    5: Flip H + Rot90
    6: Flip H + Rot180
    7: Flip H + Rot270
    """
    if x is None or y is None: return None, None # Pass
    
    # 基础变换函数 (9x9 网格，中心在 (4,4))
    # 顺时针旋转90度: (x, y) -> (y, 8-x)
    # 例如 A1(0,0) -> A9(0,8) ... 等等，需仔细核对
    # 修正：A1(0,0) 旋转90度应该是 J1(8,0)? 不对。
    # 让我们用矩阵思维：
    # x (col) 0..8, y (row) 0..8. 
    # Rot90: new_x = y, new_y = 8 - x
    # 验证: (0,0)[A1] -> (0,8)[A9]. 
    # 等等，围棋坐标 A1 是左下角。如果 y=0 是 Row 1。
    # A1 (0,0) Rot90 -> 左上角 A9 (0,8). 正确。
    # J9 (8,8) Rot90 -> 右下角 J1 (8,0).
    # New_x = 8 -> y=8. New_y = 8-8=0. -> (8,0). 正确。
    def rot90(cx, cy): return cy, 8 - cx
    
    # 水平翻转 (左右镜像)
    # A1(0,0) -> J1(8,0)
    # new_x = 8 - x, new_y = y
    def flip_h(cx, cy): return 8 - cx, cy
    
    tx, ty = x, y
    
    # 先处理翻转
    if op >= 4:
        tx, ty = flip_h(tx, ty)
        rot_times = op - 4
    else:
        rot_times = op
        
    # 再处理旋转
    for _ in range(rot_times):
        tx, ty = rot90(tx, ty)
        
    return tx, ty

def transform_board_lines(board_lines, op, invert_color=False):
    """变换 ASCII 棋盘"""
    # 1. 解析成 9x9 网格
    grid = [['.' for _ in range(9)] for _ in range(9)]
    
    # 你的数据格式通常是:
    # " 9 . . . . . . . . ."
    # ...
    # " 1 . . . . . . . . ."
    # 需要解析出 y 坐标。行标 '9' 对应 y=8, '1' 对应 y=0
    
    valid_rows = 0
    parsed_lines = []
    
    for line in board_lines:
        parts = line.strip().split()
        if not parts or not parts[0].isdigit(): 
            continue # 跳过表头 "A B C..."
        
        try:
            row_label = int(parts[0])
            y = row_label - 1 # 1-based -> 0-based
            stones = parts[1:]
            
            # 安全填充
            for x, s in enumerate(stones):
                if x < 9 and 0 <= y < 9:
                    grid[y][x] = s
            valid_rows += 1
        except ValueError:
            continue

    # 2. 几何变换 + 颜色翻转
    new_grid = [['.' for _ in range(9)] for _ in range(9)]
    for y in range(9):
        for x in range(9):
            val = grid[y][x]
            
            # 计算原坐标 (x,y) 变换后的新坐标 (nx, ny)
            # 注意：这里的逻辑是 "原图的(x,y) 跑到了 新图的(nx,ny)"
            # 还是 "新图的(nx,ny) 来自 原图的(x,y)"?
            # 通常 Transform 是 Forward: Source -> Target
            # grid[y][x] 是源。
            nx, ny = transform_xy(x, y, op)
            
            # 颜色翻转 (石头)
            if invert_color:
                if val == 'X': val = 'O'
                elif val == 'O': val = 'X'
            
            if nx is not None and ny is not None:
                new_grid[ny][nx] = val

    # 3. 重建字符串
    new_lines = ["   A B C D E F G H J"]
    for i in range(9):
        y = 8 - i # 打印顺序 9 down to 1
        row_str = f" {y+1} " + " ".join(new_grid[y])
        new_lines.append(row_str)
        
    return "\n".join(new_lines)

def transform_history(hist_line, op, invert_color=False):
    """变换历史记录 e.g. 'Last moves: B E4; W B6'"""
    # 简单判定
    if "Last moves:" not in hist_line: return hist_line
    
    prefix = "Last moves:"
    try:
        content = hist_line.split(prefix)[1].strip()
    except IndexError:
        return hist_line # 格式不对，原样返回
        
    if not content: return hist_line
    
    moves = content.split(';')
    new_moves = []
    
    for m in moves:
        m = m.strip()
        if not m: continue
        parts = m.split()
        
        # 处理 "B E4" 或 "W pass"
        if len(parts) >= 2:
            color = parts[0]
            coord = parts[1]
            
            # 颜色翻转 (Player)
            if invert_color:
                color = 'W' if color == 'B' else 'B'
                
            # 坐标变换
            x, y = parse_coord(coord)
            if x is not None:
                nx, ny = transform_xy(x, y, op)
                new_coord = to_coord_str(nx, ny)
            else:
                new_coord = coord # Pass 或其他
                
            new_moves.append(f"{color} {new_coord}")
        else:
            new_moves.append(m) # 无法解析，保留原样
            
    return prefix + " " + "; ".join(new_moves)

def process_one_record(record):
    prompt = record.get('user_prompt', "")
    katago_all = record.get('katago_all', {})
    
    if not prompt: return []

    # --- 1. 拆解 Prompt ---
    lines = prompt.split('\n')
    board_start = -1
    board_end = -1
    hist_idx = -1
    instr_idx = -1
    
    for i, line in enumerate(lines):
        # 寻找棋盘头 "A B C..."
        if "A B C D" in line:
            board_start = i
        # 寻找棋盘尾 (通常是 row 1)
        if board_start != -1 and line.strip().startswith("1 "):
            board_end = i 
        if "Last moves:" in line:
            hist_idx = i
        if "Player to move:" in line:
            instr_idx = i

    # 如果找不到棋盘，直接返回空或报错
    if board_start == -1 or board_end == -1: 
        # Fallback: 也许数据格式只有一行？跳过
        return []

    # 提取各部分
    header = lines[:board_start]
    raw_board_lines = lines[board_start : board_end + 1]
    
    # 提取 History 和 Instruction，如果没找到 index，就设为空字符串
    raw_history = lines[hist_idx] if hist_idx != -1 else ""
    raw_instr = lines[instr_idx] if instr_idx != -1 else ""
    
    # 提取剩下的尾部 (例如 Instruction 之后的行)
    # 这里比较 tricky，为了简单，我们假设结构是 Header -> Board -> History -> Instruction
    # 我们重新构建整个 Prompt 字符串
    
    augmented_list = []

    # --- 2. 循环生成 (8 几何 x 2 颜色) ---
    for op in range(8):
        for color_flip in [False, True]:
            
            # A. 变换 Prompt 组件
            # 1. Board
            new_board_str = transform_board_lines(raw_board_lines, op, color_flip)
            
            # 2. History
            new_hist_str = transform_history(raw_history, op, color_flip)
            
            # 3. Instruction (Player)
            new_instr_str = raw_instr
            if color_flip:
                # 简单文本替换 Black <-> White
                # 注意防止 "White" 变成 "Black" 后再变回 "White"
                if "Black" in new_instr_str:
                    new_instr_str = new_instr_str.replace("Black", "TEMP_HOLDER")
                    new_instr_str = new_instr_str.replace("White", "Black")
                    new_instr_str = new_instr_str.replace("TEMP_HOLDER", "White")
                elif "White" in new_instr_str:
                    new_instr_str = new_instr_str.replace("White", "TEMP_HOLDER")
                    new_instr_str = new_instr_str.replace("Black", "White")
                    new_instr_str = new_instr_str.replace("TEMP_HOLDER", "Black")
            
            # B. 重组 Prompt
            # 顺序: Header -> New Board -> New History -> New Instruction
            # 并在中间补上原有的空行
            
            new_prompt_parts = []
            new_prompt_parts.extend(header)
            new_prompt_parts.append(new_board_str)
            new_prompt_parts.append("") # 空行
            if new_hist_str: new_prompt_parts.append(new_hist_str)
            new_prompt_parts.append("") # 空行
            if new_instr_str: new_prompt_parts.append(new_instr_str)
            
            full_prompt = "\n".join(new_prompt_parts)
            
            # C. 变换 Katago 数据
            new_katago = {}
            if katago_all:
                for k, v in katago_all.items():
                    # 坐标变换
                    # 修复点：这里 parse_coord 现在返回 (None, None)，解包安全了
                    kx, ky = parse_coord(k)
                    
                    if kx is not None:
                        nkx, nky = transform_xy(kx, ky, op)
                        new_k = to_coord_str(nkx, nky)
                        
                        # 胜率变换
                        new_v = v
                        if color_flip and INVERT_WINRATE_ON_COLOR_FLIP:
                            new_v = 1.0 - v
                        
                        new_katago[new_k] = new_v
                    else:
                        # 对于 pass 或其他无法解析的坐标，保留原样
                        new_katago[k] = v 
                
            # D. 打包结果
            new_rec = copy.deepcopy(record)
            new_rec['user_prompt'] = full_prompt
            new_rec['katago_all'] = new_katago
            
            # 生成唯一 ID
            orig_id = record.get('original_id', 'unknown')
            suffix = f"_aug{op}" + ("_inv" if color_flip else "")
            new_rec['original_id'] = f"{orig_id}{suffix}"
            
            augmented_list.append(new_rec)
            
    return augmented_list

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Starting Augmentation...")
    print(f"Input: {INPUT_FILE}")
    print(f"Mode: 8x Geometric * 2x Color = 16x Total")
    print(f"Winrate Inversion: {'ENABLED' if INVERT_WINRATE_ON_COLOR_FLIP else 'DISABLED (Default)'}")
    
    count = 0
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                if not line.strip(): continue
                record = json.loads(line)
                
                aug_records = process_one_record(record)
                for rec in aug_records:
                    fout.write(json.dumps(rec) + '\n')
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} source records...")
                    
        print(f"Done! Generated {count * 16} records in {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()