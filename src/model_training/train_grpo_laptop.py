import re
import json
import torch
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from peft import LoraConfig  # <--- 记得加这一行！
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# ================= 1. 笔记本专用配置 =================
# ⚠️ 注意：请确保这个文件存在，或者改为你实际生成的文件名
# 根据你上传的文件，shuffle_data.py 输出的是 'training_data_augmented_shuffled.jsonl'
DATASET_PATH = "data/training_data_augmented_shuffled.jsonl" 

# 使用 GPT-2 (124M参数)，随便一个笔记本都能跑
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "runs/qwen2.5-0.5B-grpo-laptop-test"

# 9x9 围棋坐标系
VALID_COLS = set("ABCDEFGHJ") 
VALID_ROWS = set(str(i) for i in range(1, 10))

# System Prompt
SYSTEM_PROMPT = (
    "You are a professional 9x9 Go player. "
    "Your goal is to find the best move in the current position.\n"
    "1. FIRST, think silently about the board status, liberties, and territory in <think> tags.\n"
    "2. THEN, output your move strictly in 'MOVE: XY' format (e.g., MOVE: C4 or MOVE: pass).\n"
    "3. FINALLY, provide a short explanation starting with 'EXPLAIN:'\n"
    "\n"
    "IMPORTANT RULES:\n"
    "- DO NOT DRAW THE BOARD VISUALLY.\n"  # 🚫 禁止画图
    "- DO NOT output ASCII art.\n"        # 🚫 双重禁止
    "- Focus ONLY on text analysis inside <think> tags.\n"
    "\n"
    "Example Output:\n"
    "<think>Black has a weak group in the corner. I should attack at C3.</think>\n" # 给一个纯文字思考的例子
    "MOVE: C3\n"
    "EXPLAIN: Attacking the corner group."
)

# ================= 2. 辅助函数 (保持不变) =================

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str) -> Optional[str]:
    match = re.search(r"MOVE:.*?([A-HJ-T][1-9][0-9]?|pass)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None


def extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ================= 3. 奖励函数 (逻辑不变，降低惩罚) =================

# 修改 train_grpo_laptop.py 中的 format_reward_func

def format_reward_func(completions, **kwargs) -> List[float]:
    rewards = []
    # 打印第一个样本的内容来看看 (DEBUG 用)
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

def outcome_regret_reward_func(prompts, completions, katago_all, **kwargs) -> List[float]:
    rewards = []
    for content, k_data_str in zip(completions, katago_all):
        # 1. 解析 KataGo 数据
        if isinstance(k_data_str, str):
            try: k_data = json.loads(k_data_str)
            except: k_data = {}
        elif isinstance(k_data_str, dict):
            k_data = k_data_str
        else:
            k_data = {}

        move = extract_move(content)
        
        # 情况 A: 没输出合法坐标 -> 惩罚
        if move is None:
            rewards.append(-0.5) 
            continue
            
        # 2. 🟢 修复点：计算最佳胜率时，过滤掉 None (空值)
        # 只有当 v 不是 None 时，才转成 float
        all_winrates = [float(v) for v in k_data.values() if v is not None]
        
        # 如果过滤完是空的，就默认最佳胜率是 0.5
        best_wr = max(all_winrates) if all_winrates else 0.5
        
        # 3. 🟢 修复点：确保当前选的这步棋也有有效胜率
        if move in k_data and k_data[move] is not None:
            try:
                chosen_wr = float(k_data[move])
                regret = best_wr - chosen_wr
                rewards.append(1.0 - regret)
            except:
                # 万一转 float 还是失败，就当没找到
                rewards.append(-0.1)
        else:
            # 合法但不在推荐列表里 (或者该步棋数据是 None)
            rewards.append(-0.1) 
            
    return rewards

# ================= 4. 主程序 =================

def main():
    # 检查数据是否存在
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: 数据文件 {DATASET_PATH} 不存在！请检查路径。")
        return

    print(f"🔄 Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # ✂️ 为了快速测试，我们只取前 100 条数据
    print("✂️ Truncating dataset to first 100 examples for testing...")
    dataset = dataset.select(range(min(100, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Qwen 的 pad_token 通常就是 eos_token，或者它会自动处理，这里保持这样通常没问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token    
    # 🛠️ GPT-2 没有 Chat Template，我们手动加一个简单的
    #if tokenizer.chat_template is None:
    #    tokenizer.chat_template = "{% for message in messages %}{{ message.content }}\n{% endfor %}\nAssistant:"

    def preprocess_function(examples):
        formatted_prompts = []
        for user_p in examples["user_prompt"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p},
                # 🔴 关键修改：在这里不要直接结束，而是预填 assistant 的开头
                # 注意：trl 的 chat template 处理这个可能稍微麻烦点
                # 我们换一种简单粗暴的方法：直接把 <think> 加在 prompt 的最后
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 🔥 强制添加 <think> 引导模型
            prompt_text += "<think>\n" 
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}

    dataset = dataset.map(preprocess_function, batched=True)

    # ... (dataset map 处理完之后)

    # 🟢 新增: LoRA 配置 (让笔记本也能轻松跑，且模拟服务器环境)
    peft_config = LoraConfig(
        r=8,                    # 笔记本上用小一点的 rank 就可以了
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # 只微调核心部分，省显存
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    
    # ... (下面接着写 training_args)


    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,           # Qwen 对学习率比较敏感，改小一点点
        logging_steps=1,
        bf16=False, fp16=False,       # 笔记本通常跑 float32 最稳 (除非你是 30/40系显卡，可开 fp16=True)
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, # 🟢 增加累积步数，模拟更大的 batch size
        num_generations=4,            # 🟢 尝试生成 4 个 (Group Size)，看显存够不够，不够就改回 2
        max_completion_length=256,    # 🟢 稍微给多一点空间让它思考 (GPT2 是 128)
        # max_prompt_length=512,
        save_steps=10,
        max_steps=10,                 
        use_vllm=False,               
        report_to="none"                
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[format_reward_func, outcome_regret_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,      # 🟢 记得把 peft_config 传进来！
    )

    print("🚀 Starting Laptop GRPO Test (GPT-2)...")
    trainer.train()
    
    print(f"✅ Pipeline verification passed! Model saved to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()