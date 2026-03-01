import os  # 新增这一行，防止第 135 行的 os.path.exists 报错崩溃
from email.mime import text
import random
from random import random
import re
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ================= 1. 配置区域 (Configuration) =================
# 路径配置
DATASET_PATH = "data/training_data_augmented_shuffled.jsonl" 
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "runs/Qwen2.5-7B-GRPO-Go-Pro-v1"

# 9x9 围棋坐标系 (不包含 'I')
VALID_COLS = set("ABCDEFGHJ") 
VALID_ROWS = set(str(i) for i in range(1, 10))

# System Prompt 强化：要求它像高手一样思考
# System Prompt 强化：严禁画图，纯文本思考
SYSTEM_PROMPT = (
    "You are an expert 9x9 Go (Weiqi) Player. You must determine the best tactical next move.\n"
    "The board is a 9x9 grid. Columns are A through J (skipping I), and rows are 1 through 9.\n"
    "In the provided board state, '.' represents an empty intersection, 'X' is Black, and 'O' is White.\n\n"
    "CRITICAL SPATIAL RULES:\n"
    "1. You MUST NOT place a move on a coordinate that already contains an 'X' or 'O'.\n"
    "2. You can ONLY play on a coordinate that is currently a '.'.\n\n"
    "OUTPUT FORMAT:\n"
    "You must strictly follow this exact structure:\n"
    "<think>\n"
    "Step 1: Analyze the global board state, evaluating territory, influence, and any urgent tactical situations.\n"  # 优化：更宏观的局势评估
    "Step 2: Propose 2 to 3 candidate moves and briefly compare their pros and cons.\n" # 优化：明确要求对比优劣，体现决策过程
    "Step 3: VERIFY that your final intended move is currently an empty '.' on the board. If it is occupied, pick a different move.\n"
    "</think>\n"
    "MOVE: [Coordinate] (e.g., MOVE: C4 or MOVE: PASS)\n"
    "EXPLAIN: [1-2 short sentences explaining the strategy.]"
)

# ================= 2. 核心辅助函数 (Utilities) =================
import re
from typing import Optional, List, Tuple

def is_valid_9x9_coord(coord: str) -> bool:
    """严格检查是否为 9x9 合法坐标"""
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str) -> Optional[str]:
    """提取 MOVE，增加对 'pass' 的支持，并进行 9x9 校验"""
    match = re.search(r"MOVE:\s*([A-HJ-Z][1-9]|PASS)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None

def extract_think_content(text: str) -> str:
    """提取 <think> 内部的内容用于分析"""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def check_spatial_legality(prompt_text: str, move_str: str) -> float:
    """
    【新增】二维空间碰撞检测裁判。
    解析 prompt 中的 ASCII 棋盘，检查落子是否在空位('.')上。
    返回应扣除的惩罚分数。
    """
    if move_str == "PASS":
        return 0.0 # Pass 在空间上总是合法的

    col_char, row_char = move_str[0], move_str[1]
    
    # 1. 提取出纯棋盘部分 (寻找形如 " 9 . . X O . . . . ." 的行)
    lines = prompt_text.split('\n')
    board_lines = []
    for line in lines:
        if re.match(r"^\s*[1-9]\s+([.XO]\s+)*[.XO]", line):
            board_lines.append(line)

    if len(board_lines) != 9:
         return -0.1 # 解析意外，保守扣除一点分
         
    # 2. 坐标映射 (A-J 映射到 0-8，9-1 映射到 0-8)
    row_idx = 9 - int(row_char)
    col_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}
    col_idx = col_mapping.get(col_char)
    
    if col_idx is None:
        return -0.5

    # 3. 碰撞检测
    row_elements = board_lines[row_idx].strip().split()
    if len(row_elements) >= 10: # 索引 0 是行号，后面紧跟 9 个交叉点状态
        target_spot = row_elements[col_idx + 1]
        if target_spot != '.':
            return -0.8 # 核心惩罚：下在已有棋子上！

    return 0.0 # 合法落子（点在 '.' 上），不奖不罚


# ================= 3. 奖励函数群 (The Reward Engineering) =================

# --- A. 格式与二维空间感知奖励 (Layer 1) ---
def format_and_legality_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    第一层：严格规范输出格式，并教导模型认清 9x9 的二维空间。
    这是模型拿分的底线门槛。
    """
    rewards = []
    for prompt, content in zip(prompts, completions):
        score = 0.0
        
        # 基础格式奖励
        if "<think>" in content and "</think>" in content:
            score += 0.2
            
        move_str = extract_move(content)
        
        if not move_str:
             score -= 1.0 # 最高级惩罚：连合法坐标都没输出
        else:
            # 二维空间碰撞检测 (如果下在已有棋子上，吃 -0.8 的大亏)
            collision_penalty = check_spatial_legality(prompt, move_str)
            score += collision_penalty
            
        rewards.append(score)
    return rewards

# --- B. 言行一致性与思考质量奖励 (Layer 3 & 4) ---
def thinking_quality_reward_func(prompts, completions, katago_all, **kwargs) -> List[float]:
    """
    第三层与第四层：言行一致性校验 + 条件性字数奖励。
    防止模型生成“正确的废话”或者“想到A却下B”的精神分裂。
    """
    rewards = []
    for prompt, content, k_data in zip(prompts, completions, katago_all):
        if not isinstance(k_data, dict):
            k_data = {}
            
        score = 0.0
        move_str = extract_move(content)
        think_text = extract_think_content(content)
        
        # 门槛 1：如果动作本身就不合法（包含格式错和撞子），思考再多也是 0 分
        if not move_str or check_spatial_legality(prompt, move_str) < 0:
            rewards.append(0.0)
            continue
            
        # 门槛 2：言行一致性检查 (最终下的棋，必须在思考链里推导过)
        if move_str == "PASS":
            mentioned_moves = [m.upper() for m in re.findall(r'\bpass\b', think_text, re.IGNORECASE)]
        else:
            # 提取 <think> 中所有形如 A1, C4 的坐标
            mentioned_moves = [m.upper() for m in re.findall(r'\b([A-HJ-Z][1-9])\b', think_text, re.IGNORECASE)]
        
        is_consistent = move_str in mentioned_moves
        
        if not is_consistent:
            score -= 0.5 # 逻辑断裂惩罚
        else:
            score += 0.2 # 言行一致奖励
            
            # 只有在言行一致的前提下，才评估思考的“量” (防止凑字数)
            length = len(think_text)
            if length == 0:
                score -= 0.5
            elif 50 <= length <= 800:
                score += 0.1 # 黄金思考长度
            elif length > 800:
                score -= 0.1 # 啰嗦惩罚
                
            # 加分项：如果不仅思考了，下得还在 KataGo 候选里，说明是真的想明白了
            if move_str in k_data:
                score += 0.2
                
        rewards.append(score)
    return rewards

# 初始化日志文件路径
LOG_FILE = "grpo_training_rollouts.jsonl"

def logging_reward_func(prompts, completions, katago_all, katago_best, **kwargs):
    """
    专门用于抽样记录日志的“卧底”奖励函数。
    它不影响模型得分（永远返回 0.0），只负责悄悄把生成的文本和局势写进本地文件。
    """
    # 纯粹为了满足 TRL 接口要求， 返回全是 0.0 的列表
    zero_rewards = [0.0] * len(completions)
    
    # 为了避免日志爆炸（每步生成量很大），我们每批数据只随机抽取 1 条进行记录
    if random.random() < 1.0: # 如果嫌多，可以改成 0.1 (即 10% 的批次才记录)
        # 随机挑一个幸运儿
        sample_idx = random.randint(0, len(completions) - 1)
        
        prompt_text = prompts[sample_idx]
        content = completions[sample_idx]
        k_data = katago_all[sample_idx] if isinstance(katago_all[sample_idx], dict) else {}
        k_best = katago_best[sample_idx]
        
        # 提取模型最终给出的坐标
        move = extract_move(content)
        
        # 复用我们之前写的打分逻辑，计算一下它这把能得多少分（方便日后复盘）
        layer1_score = 0.2 if "<think>" in content and "</think>" in content else 0.0
        if not move: 
            layer1_score -= 1.0
        else:
            layer1_score += check_spatial_legality(prompt_text, move)
            
        # 组装要写入的日志数据
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "board_state_prompt": prompt_text, # 记录它当时看到了什么
            "model_generation": content,       # 记录它的完整思考和回答
            "extracted_move": move,            # 记录提取出的落子
            "katago_ground_truth": {
                "best_winrate": k_best,
                "candidates": k_data
            },
            "reward_breakdown_preview": {
                "layer1_format_and_space": layer1_score,
                # 这里为了简洁只演示计算了第一层的分，实际训练中它会获得各个函数的总和
            }
        }
        
        # 追加写入 JSONL 文件
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            pass # 忽略文件写入错误，防止中断昂贵的训练
            
    return zero_rewards

# --- C. 绝对胜率与后悔值奖励 (Layer 2 - Core) ---
def outcome_regret_reward_func(prompts, completions, katago_all, katago_best, **kwargs) -> List[float]:
    """
    第二层：解决劣势局过度奖励的悖论。
    公式：Reward = Chosen_Winrate - 1.5 * (Best_Winrate - Chosen_Winrate)
    """
    rewards = []

    for content, k_data, best_wr in zip(completions, katago_all, katago_best):
        if not isinstance(k_data, dict):
            k_data = {}

        move = extract_move(content)
        
        # 门槛：非法动作不在此处重复惩罚，直接给 0
        if not move:
            rewards.append(0.0)
            continue
            
        best_wr_val = float(best_wr) if best_wr is not None else 0.5

        # 情况 1: 模型落子在 KataGo 的高质量候选点中
        if move in k_data and k_data[move] is not None:
            try:
                chosen_wr = float(k_data[move])
                regret = best_wr_val - chosen_wr
                
                # 绝对胜率做基底，错失最优解做惩罚 (系数设为 1.5)
                reward_score = chosen_wr - (1.5 * regret)
                rewards.append(max(-1.0, min(1.0, reward_score)))
            except:
                rewards.append(-0.2)
                
        # 情况 2: 模型下了一步合法棋，但不在 Top 候选推荐里
        else:
            if k_data:
                # 寻找候选池中最差的胜率作为 baseline
                worst_candidate_wr = min(float(v) for v in k_data.values() if v is not None)
                # 因为它比候选池里最差的还差，胜率基底再扣 0.1
                fallback_wr = max(0.0, worst_candidate_wr - 0.1)
            else:
                fallback_wr = 0.0
                
            regret = best_wr_val - fallback_wr
            # 盲点额外固定惩罚 -0.2
            reward_score = fallback_wr - (1.5 * regret) - 0.2
            rewards.append(max(-1.0, min(1.0, reward_score)))
            
    return rewards

# ================= 4. 主程序 (Main) =================

def main():
    print(f"🔄 Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: file: {DATASET_PATH} Does not exist！ Please check the path.")
        return

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 预处理：应用 Chat Template
    def preprocess_function(examples):
        formatted_prompts = []
        for user_p in examples["user_prompt"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p}
            ]
            # add_generation_prompt=True 确保 prompt 停在 <|im_start|>assistant
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # 🟢 新增这行：强行帮模型写下 <think> 的开头，逼迫它推理
            # prompt_text += "<think>\n"
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}
    
    # 处理所有数据
    dataset = dataset.map(preprocess_function, batched=True)

    # LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    # GRPO 配置 (针对 RTX 6000 Ada 优化)
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6, # 稍微降低 LR，因为我们有强 Reward 信号
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True, # 必须开启 BF16 来充分利用 Ada 的性能
        
        # 🛠️ 修改 2: 解决整除报错 & 利用大显存
        # 你的服务器有 48GB，我们可以把 batch_size 提高到 4
        # 4 (Batch) 能被 4 (Generations) 整除，完美解决报错
        per_device_train_batch_size=8, 
        num_generations=8,            
        
        gradient_accumulation_steps=2, # 等效 Batch = 2 * 8 = 16
        
        max_completion_length=1024, # 给足够的空间思考 (512 可能有点紧)
        # 🛠️ 修改 3: 删除 max_prompt_length (防止报错)
        # max_prompt_length=2048, 
        
        save_steps=100,
        max_steps=1000, # 增加步数，因为数据质量高且增强过
        
        # 🛠️ 修改 4: 确保 report_to 是字符串 (防止 None 报错)
        #report_to="tensorboard",
        report_to="wandb",

        # ⚡ 开启 vLLM (服务器专用)
        use_vllm=True, 
        vllm_gpu_memory_utilization=0.5, 
    )
    
    # 初始化 Trainer
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            format_and_legality_reward_func, # 权重 1: 格式
            thinking_quality_reward_func,# 权重 1: 思考质量
            outcome_regret_reward_func,  # 权重 1: 核心胜率 Regret
            logging_reward_func, # 🟢 新增：卧底日志记录员
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("🚀 Starting GRPO training on RTX 6000 Ada...")
    print(f"   - vLLM Enabled: {training_args.use_vllm}")
    print(f"   - Batch Size: {training_args.per_device_train_batch_size}")
    print(f"   - Group Size: {training_args.num_generations}")
    
    trainer.train()
    
    print(f"✅ Training finished. Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()