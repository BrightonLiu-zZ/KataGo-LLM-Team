import os
import random
import re
import json
import torch
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ================= 1. 配置区域 =================
DATASET_PATH = "data/training_data_augmented_shuffled.jsonl" 
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "runs/local-test-output"

VALID_COLS = set("ABCDEFGHJ") 
VALID_ROWS = set(str(i) for i in range(1, 10))

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
    "Step 1: Analyze the global board state, evaluating territory, influence, and any urgent tactical situations.\n"  
    "Step 2: Propose 2 to 3 candidate moves and briefly compare their pros and cons.\n" 
    "Step 3: VERIFY that your final intended move is currently an empty '.' on the board. If it is occupied, pick a different move.\n"
    "</think>\n"
    "MOVE: [Coordinate] (e.g., MOVE: C4 or MOVE: PASS)\n"
    "EXPLAIN: [1-2 short sentences explaining the strategy.]"
)

# ================= 2. 核心辅助函数 =================
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
    col_char, row_char = move_str[0], move_str[1]
    lines = prompt_text.split('\n')
    board_lines = [line for line in lines if re.match(r"^\s*[1-9]\s+([.XO]\s+)*[.XO]", line)]
    if len(board_lines) != 9: return -0.1
    row_idx = 9 - int(row_char)
    col_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}
    col_idx = col_mapping.get(col_char)
    if col_idx is None: return -0.5
    row_elements = board_lines[row_idx].strip().split()
    if len(row_elements) >= 10:
        if row_elements[col_idx + 1] != '.': return -0.8
    return 0.0

# ================= 3. 奖励函数群 =================
def format_and_legality_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, content in zip(prompts, completions):
        score = 0.2 if "<think>" in content and "</think>" in content else 0.0
        move_str = extract_move(content)
        if not move_str: score -= 1.0
        else: score += check_spatial_legality(prompt, move_str)
        rewards.append(score)
    return rewards

def thinking_quality_reward_func(prompts, completions, katago_all, **kwargs):
    rewards = []
    for prompt, content, k_data in zip(prompts, completions, katago_all):
        if not isinstance(k_data, dict): k_data = {}
        score = 0.0
        move_str = extract_move(content)
        think_text = extract_think_content(content)
        if not move_str or check_spatial_legality(prompt, move_str) < 0:
            rewards.append(0.0)
            continue
        mentioned = [m.upper() for m in re.findall(r'\b([A-HJ-Z][1-9]|PASS)\b', think_text, re.IGNORECASE)]
        if move_str not in mentioned: score -= 0.5
        else:
            score += 0.2
            l = len(think_text)
            if l == 0: score -= 0.5
            elif 50 <= l <= 800: score += 0.1
            elif l > 800: score -= 0.1
            if move_str in k_data: score += 0.2
        rewards.append(score)
    return rewards

LOG_FILE = "grpo_local_test_rollouts.jsonl"
def logging_reward_func(prompts, completions, katago_all, katago_best, **kwargs):
    zero_rewards = [0.0] * len(completions)
    if random.random() < 1.0:
        sample_idx = random.randint(0, len(completions) - 1)
        prompt_text = prompts[sample_idx]
        content = completions[sample_idx]
        k_data = katago_all[sample_idx] if isinstance(katago_all[sample_idx], dict) else {}
        k_best = katago_best[sample_idx]
        move = extract_move(content)
        layer1_score = 0.2 if "<think>" in content and "</think>" in content else 0.0
        if not move: layer1_score -= 1.0
        else: layer1_score += check_spatial_legality(prompt_text, move)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "board_state_prompt": prompt_text,
            "model_generation": content,
            "extracted_move": move,
            "katago_ground_truth": {"best_winrate": k_best, "candidates": k_data},
            "reward_breakdown_preview": {"layer1_format_and_space": layer1_score}
        }
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except: pass
    return zero_rewards

def outcome_regret_reward_func(prompts, completions, katago_all, katago_best, **kwargs):
    rewards = []
    for content, k_data, best_wr in zip(completions, katago_all, katago_best):
        if not isinstance(k_data, dict): k_data = {}
        move = extract_move(content)
        if not move:
            rewards.append(0.0)
            continue
        best_wr_val = float(best_wr) if best_wr is not None else 0.5
        if move in k_data and k_data[move] is not None:
            try:
                chosen_wr = float(k_data[move])
                regret = best_wr_val - chosen_wr
                reward_score = chosen_wr - (1.5 * regret)
                rewards.append(max(-1.0, min(1.0, reward_score)))
            except: rewards.append(-0.2)
        else:
            if k_data:
                worst = min(float(v) for v in k_data.values() if v is not None)
                fallback = max(0.0, worst - 0.1)
            else: fallback = 0.0
            regret = best_wr_val - fallback
            reward_score = fallback - (1.5 * regret) - 0.2
            rewards.append(max(-1.0, min(1.0, reward_score)))
    return rewards

# ================= 4. 主程序 =================
def main():
    print(f"🔄 Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: file: {DATASET_PATH} Does not exist！")
        return

    # 本地测试只取前 50 条数据加速准备过程
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train[:50]")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
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

    # 🟢 核心改动 1：配置 4-bit 量化加载 (专为 8GB 显存打造)
    print("🔋 Loading model in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    # 🟢 核心改动 2：本地极简参数配置
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6, 
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,             # 每步都打日志，方便监控
        bf16=True,                   # RTX 4060 支持 BF16
        
        per_device_train_batch_size=2, # 极限压缩 Batch
        num_generations=2,             # 最少需要 2 个来计算优势值
        generation_batch_size=2,       # 新增这一行！显式告诉框架生成批次也是 2
        gradient_accumulation_steps=1,
        max_completion_length=256,     # 防止生成太长导致 OOM
        
        save_steps=5,
        max_steps=5,                   # 跑 5 步就自动停！只是为了跑通流程
        
        report_to="none",              # 关闭 wandb，避免本地测试卡在输入 token 环节

        use_vllm=False,                # 禁用 vLLM
        # vllm_gpu_memory_utilization=0.5 # 禁用该选项
    )
    
    trainer = GRPOTrainer(
        model=model, # 传入量化加载的 model 实例，而不是字符串
        reward_funcs=[
            format_and_legality_reward_func,
            thinking_quality_reward_func,
            outcome_regret_reward_func,
            logging_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("🚀 Starting LOCAL GRPO logic test on RTX 4060...")
    trainer.train()
    
    print("✅ Local test passed! The logic graph and reward functions are working correctly.")

if __name__ == "__main__":
    main()