from email.mime import text
import re
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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
    "You are a professional 9x9 Go player. "
    "Your goal is to find the best move in the current position.\n"
    "1. FIRST, think silently about the board status, liberties, and territory in <think> tags.\n"
    "2. THEN, output your move strictly in 'MOVE: XY' format (e.g., MOVE: C4 or MOVE: pass).\n"
    "3. FINALLY, provide a short explanation starting with 'EXPLAIN:'\n"
    "\n"
    "IMPORTANT RULES:\n"
    "- DO NOT DRAW THE BOARD VISUALLY.\n"  
    "- DO NOT output ASCII art.\n"        
    "- Focus ONLY on text analysis inside <think> tags.\n"
    "\n"
    "Example Output:\n"
    "<think>Black has a weak group in the corner. I should attack at C3.</think>\n" 
    "MOVE: C3\n"
    "EXPLAIN: Attacking the corner group."
)

# ================= 2. 核心辅助函数 (Utilities) =================

def is_valid_9x9_coord(coord: str) -> bool:
    """严格检查是否为 9x9 合法坐标"""
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str) -> Optional[str]:
    """提取 MOVE，增加对 'pass' 的支持，并进行 9x9 校验"""
    # 匹配 MOVE: C4, MOVE: pass, Move: A1 等
    match = re.search(r"MOVE:.*?([A-HJ-T][1-9][0-9]?|pass)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None

def extract_think_content(text: str) -> str:
    """提取 <think> 内部的内容用于分析"""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ================= 3. 奖励函数群 (The Reward Engineering) =================

# --- A. 格式奖励 (基础) ---
def format_reward_func(completions, **kwargs) -> List[float]:
    """
    检查结构完整性。
    奖励设计：
    - 完整结构 (+0.5)
    - 缺少特定标签 (-0.5 per missing tag)
    """
    rewards = []
    print(f"\n[DEBUG PREVIEW] Model Output:\n{completions[0][:500]}\n-------------------")
    for content in completions:
        score = 0.0
        # 只要写了 <think> 就给一点甜头，鼓励它思考
        if "<think>" in content: score += 0.2
        if "</think>" in content: score += 0.2
        # 只要写了 MOVE: 就给甜头
        if "MOVE:" in content: score += 0.3 
        rewards.append(score)
    return rewards

# --- B. 思考过程奖励 (进阶) ---
def thinking_quality_reward_func(completions, **kwargs) -> List[float]:
    """
    奖励思考的'量' (Lenient Length Reward)。
    防止模型偷懒(直接输出答案)或产生无限循环幻觉。
    """
    rewards = []
    for content in completions:
        think_text = extract_think_content(content)
        length = len(think_text)
        
        if length == 0:
            rewards.append(-0.5) # 没思考，扣分
        elif length < 50:
            rewards.append(0.0)  # 思考太短，不给分
        elif 50 <= length <= 500:
            rewards.append(0.5)  # 黄金思考长度，奖励
        else:
            rewards.append(0.0)  # 太长了可能在啰嗦，不给分
            
    return rewards

# --- C. 胜率后悔值奖励 (核心 - Regret Based) ---
def outcome_regret_reward_func(prompts, completions, katago_all, **kwargs) -> List[float]:
    """
    核心逻辑：Regret Minimization
    Reward = 1.0 - (Best_Winrate - Chosen_Move_Winrate)
    """
    rewards = []
    for content, k_data_str in zip(completions, katago_all):
        # 解析 KataGo 数据 (JSON string -> dict)
        if isinstance(k_data_str, str):
            try: k_data = json.loads(k_data_str)
            except: k_data = {}
        else:
            k_data = k_data_str if k_data_str else {}

        move = extract_move(content)
        
        # 情况 1: 格式错误或非法坐标
        if move is None:
            rewards.append(-1.0)
            continue
            
        # 计算当前局面的最佳胜率 (Best Winrate)
        # 注意：如果 k_data 为空（极少见），假设最佳胜率是 0.5 (模糊)
        all_winrates = [float(v) for v in k_data.values() if v is not None]        
        best_wr = max(all_winrates) if all_winrates else 0.5        


        if move in k_data and k_data[move] is not None:
            try:
                chosen_wr = float(k_data[move])
                # Regret = 最佳 - 当前
                regret = best_wr - chosen_wr
                # Reward: 0 regret -> 1.0 score.  0.2 regret -> 0.8 score.
                rewards.append(1.0 - regret)
            except:
                rewards.append(-0.2) # 转换失败作为惩罚
        else:
            # 情况 2: 模型下了一步合法棋，但不在 Top-N 推荐里 (Bad Move)
            # 我们假设这步棋的胜率接近于 0 (或者比最差的推荐还要差)
            # 为了不过度惩罚（避免梯度爆炸），我们给它一个基于最大Regret的分数
            # 假设这步棋胜率为 0.0            
            chosen_wr = 0.0
            regret = best_wr - chosen_wr
            # 我们给它一个额外的 -0.2 惩罚，告诉它“你甚至没进候选列表”
            # 但依然比“格式错误(-1.0)”要好，鼓励它输出坐标
            rewards.append((1.0 - regret) - 0.2)
            
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
            prompt_text += "<think>\n"
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
        per_device_train_batch_size=4, 
        num_generations=4,            
        
        gradient_accumulation_steps=4, # 等效 Batch = 4 * 4 = 16
        
        max_completion_length=1024, # 给足够的空间思考 (512 可能有点紧)
        # 🛠️ 修改 3: 删除 max_prompt_length (防止报错)
        # max_prompt_length=2048, 
        
        save_steps=100,
        max_steps=1000, # 增加步数，因为数据质量高且增强过
        
        # 🛠️ 修改 4: 确保 report_to 是字符串 (防止 None 报错)
        report_to="tensorboard",
        
        # ⚡ 开启 vLLM (服务器专用)
        use_vllm=True, 
        vllm_gpu_memory_utilization=0.5, 
    )
    
    # 初始化 Trainer
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            format_reward_func,          # 权重 1: 格式
            thinking_quality_reward_func,# 权重 1: 思考质量
            outcome_regret_reward_func,  # 权重 1: 核心胜率 Regret
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