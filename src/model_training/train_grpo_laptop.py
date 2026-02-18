import re
import json
import torch
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# ================= 1. 笔记本专用配置 =================
# ⚠️ 注意：请确保这个文件存在，或者改为你实际生成的文件名
# 根据你上传的文件，shuffle_data.py 输出的是 'training_data_augmented_shuffled.jsonl'
DATASET_PATH = "data/training_data_augmented_shuffled.jsonl" 

# 使用 GPT-2 (124M参数)，随便一个笔记本都能跑
MODEL_NAME = "gpt2"
OUTPUT_DIR = "runs/gpt2-grpo-laptop-test"

# 9x9 围棋坐标系
VALID_COLS = set("ABCDEFGHJ") 
VALID_ROWS = set(str(i) for i in range(1, 10))

# System Prompt
SYSTEM_PROMPT = (
    "You are a Go player. Find the best move.\n"
    "Format: <think>...</think> MOVE: XY EXPLAIN: ..."
)

# ================= 2. 辅助函数 (保持不变) =================

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str) -> Optional[str]:
    match = re.search(r"MOVE:\s*([a-zA-Z][0-9]+|pass)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None

def extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ================= 3. 奖励函数 (逻辑不变，降低惩罚) =================

def format_reward_func(completions, **kwargs) -> List[float]:
    rewards = []
    for content in completions:
        score = 0.0
        # GPT2 很笨，我们放宽要求，只要有 MOVE: 就算成功一半
        if "MOVE:" in content:
            score += 0.5
        if "<think>" in content:
            score += 0.2
        rewards.append(score)
    return rewards

def outcome_regret_reward_func(prompts, completions, katago_all, **kwargs) -> List[float]:
    rewards = []
    for content, k_data_str in zip(completions, katago_all):
        # 兼容处理：如果是字符串则解析，如果是字典则直接用
        if isinstance(k_data_str, str):
            try: k_data = json.loads(k_data_str)
            except: k_data = {}
        elif isinstance(k_data_str, dict):
            k_data = k_data_str
        else:
            k_data = {}

        move = extract_move(content)
        
        if move is None:
            rewards.append(-0.5) # 稍微惩罚
            continue
            
        all_winrates = [float(v) for v in k_data.values()]
        best_wr = max(all_winrates) if all_winrates else 0.5
        
        if move in k_data:
            chosen_wr = float(k_data[move])
            regret = best_wr - chosen_wr
            rewards.append(1.0 - regret)
        else:
            # 合法但不在推荐列表里
            rewards.append(0.1) 
            
    return rewards

# ================= 4. 主程序 =================

def main():
    # 检查数据是否存在
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: 数据文件 {DATASET_PATH} 不存在！请检查路径。")
        return

    print(f"🔄 Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # ✂️ 为了快速测试，我们只取前 20 条数据
    print("✂️ Truncating dataset to first 20 examples for testing...")
    dataset = dataset.select(range(min(20, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 🛠️ GPT-2 没有 Chat Template，我们手动加一个简单的
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message.content }}\n{% endfor %}\nAssistant:"

    def preprocess_function(examples):
        formatted_prompts = []
        for user_p in examples["user_prompt"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p}
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}

    dataset = dataset.map(preprocess_function, batched=True)

    # 笔记本配置：低显存，无 LoRA，无 vLLM
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        logging_steps=1,
        bf16=False, fp16=False,       # 笔记本 CPU/低端显卡可能不支持，先关掉
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1, 
        num_generations=2,            # Group Size = 2 (最小生成数，省显存)
        max_completion_length=128,    # GPT2 只有 1024 窗口，这里生成短一点
        max_prompt_length=512,
        save_steps=10,
        max_steps=10,                 # 只跑 10 步验证流程！
        use_vllm=False,               # ❌ 笔记本关掉 vLLM
        report_to=None                # 不上传 wandb
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[format_reward_func, outcome_regret_reward_func],
        args=training_args,
        train_dataset=dataset,
        # peft_config=None,           # ❌ 笔记本跑 GPT-2 不用 LoRA，直接微调
    )

    print("🚀 Starting Laptop GRPO Test (GPT-2)...")
    trainer.train()
    
    print(f"✅ Pipeline verification passed! Model saved to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()