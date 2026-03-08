import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. 核心路径配置 ---
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./runs/Qwen2.5-7B-GRPO-Go-Pro-v2/checkpoint-900" 

print("🔋 正在以全精度 (BF16) 加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🧠 正在挂载第 900 步的强化学习外挂 (LoRA 权重)...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# --- 2. 准备系统提示词 ---
SYSTEM_PROMPT = """You are an expert 9x9 Go (Weiqi) Player. You must determine the best tactical next move.
The board is a 9x9 grid. Columns are A through J (skipping I), and rows are 1 through 9.
In the provided board state, '.' represents an empty intersection, 'X' is Black, and 'O' is White.

CRITICAL SPATIAL RULES:
1. You MUST NOT place a move on a coordinate that already contains an 'X' or 'O'.
2. You can ONLY play on a coordinate that is currently a '.'.

OUTPUT FORMAT:
You must strictly follow this exact structure:
<think>
Step 1: Analyze the global board state, evaluating territory, influence, and any urgent tactical situations.
Step 2: Propose 2 to 3 candidate moves and briefly compare their pros and cons.
Step 3: VERIFY that your final intended move is currently an empty '.' on the board. If it is occupied, pick a different move.
</think>
MOVE: [Coordinate] (e.g., MOVE: C4 or MOVE: PASS)
EXPLAIN: [1-2 short sentences explaining the strategy.]"""

# --- 3. 交互式对话循环 ---
print("\n" + "="*50)
print("✅ 模型加载完毕！进入交互模式。")
print("👉 请在下方粘贴你的局势提示词（支持多行）。")
print("👉 粘贴完成后，在新的一行输入 'GO' 并回车，让模型开始思考。")
print("👉 输入 'QUIT' 退出程序。")
print("="*50 + "\n")

while True:
    print("\n📝 请粘贴棋盘 (输入 GO 提交, QUIT 退出):")
    
    user_prompt_lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
            
        if line.strip().upper() == 'QUIT':
            print("👋 退出测试。")
            exit()
        elif line.strip().upper() == 'GO':
            break
        else:
            user_prompt_lines.append(line)
            
    user_prompt = "\n".join(user_prompt_lines).strip()
    
    if not user_prompt:
        print("⚠️ 提示词不能为空，请重新输入。")
        continue

    # 组装消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print("\n🚀 模型正在思考中...")
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "✨ 思考结果 ✨".center(40, "="))
    print(response)
    print("="*46)