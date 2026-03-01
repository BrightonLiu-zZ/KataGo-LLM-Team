import re

# ================= 1. 复制过来的辅助函数 =================
VALID_COLS = "ABCDEFGHJ"
VALID_ROWS = "123456789"

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str):
    match = re.search(r"MOVE:\s*([A-HJ-Z][1-9]|PASS)", text, re.IGNORECASE)
    return match.group(1).upper() if match and is_valid_9x9_coord(match.group(1).upper()) else None

def extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def check_spatial_legality(prompt_text: str, move_str: str) -> float:
    if move_str == "PASS": return 0.0
    lines = prompt_text.split('\n')
    board_lines = [line for line in lines if re.match(r"^\s*[1-9]\s+([.XO]\s+)*[.XO]", line)]
    if len(board_lines) != 9: return -0.1
    
    col_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}
    row_idx = 9 - int(move_str[1])
    col_idx = col_mapping.get(move_str[0])
    
    if col_idx is None: return -0.5
    
    row_elements = board_lines[row_idx].strip().split()
    if len(row_elements) >= 10:
        if row_elements[col_idx + 1] != '.':
            return -0.8 # 下在已有棋子上
    return 0.0

# ================= 2. 复制过来的奖励函数 =================
def format_and_legality_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, content in zip(prompts, completions):
        score = 0.2 if "<think>" in content and "</think>" in content else 0.0
        move = extract_move(content)
        if not move: score -= 1.0
        else: score += check_spatial_legality(prompt, move)
        rewards.append(score)
    return rewards

def thinking_quality_reward_func(prompts, completions, katago_all, **kwargs):
    rewards = []
    for prompt, content, k_data in zip(prompts, completions, katago_all):
        score = 0.0
        move = extract_move(content)
        think_text = extract_think_content(content)
        
        if not move or check_spatial_legality(prompt, move) < 0:
            rewards.append(0.0)
            continue
            
        mentioned = [m.upper() for m in re.findall(r'\b([A-HJ-Z][1-9]|PASS)\b', think_text, re.IGNORECASE)]
        if move not in mentioned:
            score -= 0.5
        else:
            score += 0.2
            l = len(think_text)
            if 50 <= l <= 800: score += 0.1
            if move in k_data: score += 0.2
        rewards.append(score)
    return rewards

def outcome_regret_reward_func(prompts, completions, katago_all, katago_best, **kwargs):
    rewards = []
    for content, k_data, best_wr in zip(completions, katago_all, katago_best):
        move = extract_move(content)
        if not move:
            rewards.append(0.0)
            continue
            
        best_wr_val = float(best_wr)
        if move in k_data:
            chosen_wr = float(k_data[move])
            score = chosen_wr - 1.5 * (best_wr_val - chosen_wr)
        else:
            worst = min([float(v) for v in k_data.values()]) if k_data else 0.0
            fallback = max(0.0, worst - 0.1)
            score = fallback - 1.5 * (best_wr_val - fallback) - 0.2
        rewards.append(max(-1.0, min(1.0, score)))
    return rewards

# ================= 3. 模拟测试区 (Mock Data) =================
if __name__ == "__main__":
    # 模拟一个棋盘：A1 是空位(.)，A2 是黑棋(X)
    mock_prompt = """[Current 9x9 Board State]
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . X . . . . .
 6 . O X . O X X . .
 5 X X O O O X O . .
 4 O X O X X O O O .
 3 . O O O X X O . .
 2 X O X X X X O . .
 1 . X . X . . X O .
"""
    # 我们将同一个棋盘复制 3 份，因为要测试 3 个不同的回答
    prompts = [mock_prompt, mock_prompt, mock_prompt]
    
    # 模拟 3 种模型的回答
    completions = [
        # 回答 1：完美。提到了 A1，格式对，A1 是空位。
        "<think> The corner at A1 looks empty and promising. </think>\nMOVE: A1",
        
        # 回答 2：空间幻觉。提到了 A2，格式对，但 A2 已经有 X 了！
        "<think> I should defend at A2. </think>\nMOVE: A2",
        
        # 回答 3：不守规矩/逻辑断裂。没有按格式写 MOVE。
        "<think> Maybe B1? </think>\nI will play at B1."
    ]
    
    # 模拟 KataGo 的数据：最佳胜率 0.95，A1 的胜率是 0.90
    katago_all = [{'A1': 0.90, 'B1': 0.85}, {'A1': 0.90, 'B1': 0.85}, {'A1': 0.90, 'B1': 0.85}]
    katago_best = [0.95, 0.95, 0.95]

    print("🚀 正在运行 Reward 函数测试...\n")
    
    # 运行函数并打包结果以便查看
    r1 = format_and_legality_reward_func(prompts, completions)
    r2 = thinking_quality_reward_func(prompts, completions, katago_all=katago_all)
    r3 = outcome_regret_reward_func(prompts, completions, katago_all=katago_all, katago_best=katago_best)

    labels = ["1. 完美回答 (A1空位)", "2. 撞子犯规 (A2已有子)", "3. 格式错误 (无MOVE)"]
    
    for i in range(3):
        print(f"[{labels[i]}]")
        print(f"  格式与二维空间感知 (Layer 1): {r1[i]:.2f}")
        print(f"  言行一致与思考质量 (Layer 3/4): {r2[i]:.2f}")
        print(f"  绝对胜率与后悔值   (Layer 2): {r3[i]:.2f}")
        print(f"  👉 总分: {r1[i] + r2[i] + r3[i]:.2f}\n")