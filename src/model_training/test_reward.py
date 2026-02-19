import re

# 复制你 train_grpo.py (服务器版) 里的正则函数
VALID_COLS = set("ABCDEFGHJ") 
VALID_ROWS = set(str(i) for i in range(1, 10))

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str):
    # 测试你的正则能否搞定各种奇怪的输出
    match = re.search(r"MOVE:\s*([a-zA-Z][0-9]+|pass)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move): return move
    return None

# 模拟 Qwen 可能的输出
test_cases = [
    "<think>Hmm...</think> MOVE: C4 EXPLAIN: Good.",  # 完美
    "MOVE: c4",                                       # 小写 (你的正则支持吗？)
    "I think playing at MOVE: T19 is good.",          # 非法坐标 (T19 超出 9x9)
    "The answer is Move: pass.",                      # 混合写法
    "No move here."                                   # 错误
]

print("=== 测试正则提取能力 ===")
for t in test_cases:
    print(f"Input: {t} -> Extracted: {extract_move(t)}")