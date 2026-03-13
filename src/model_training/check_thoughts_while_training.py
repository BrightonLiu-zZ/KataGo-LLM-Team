import json
import os

LOG_FILE = "grpo_training_rollouts.jsonl"

def display_latest_thoughts(num_entries=3):
    if not os.path.exists(LOG_FILE):
        print(f"❌ 找不到日志文件: {LOG_FILE}。可能是模型还没跑完第一个采样批次。")
        return

    # 读取所有行并获取最后 N 行
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    recent_lines = lines[-num_entries:]
    
    print(f"\n{'='*20} 🧠 模型最新思考状态展示 (最近 {len(recent_lines)} 条) {'='*20}\n")
    
    for i, line in enumerate(recent_lines, 1):
        try:
            data = json.loads(line)
            
            # 提取信息
            timestamp = data.get("timestamp", "Unknown Time")
            generation = data.get("model_generation", "No generation")
            move = data.get("extracted_move", "None")
            k_truth = data.get("katago_ground_truth", {})
            best_wr = k_truth.get("best_winrate", "N/A")
            
            # 尝试获取模型下这步棋的真实胜率
            chosen_wr = k_truth.get("candidates", {}).get(move, "不在 KataGo 候选池中")

            # 打印排版
            print(f"🔸 【记录 {i}】 时间: {timestamp}")
            print("-" * 60)
            print("🤔 【模型的原生输出 (包含思考与决策)】:")
            print(generation.strip())
            print("-" * 60)
            print(f"🎯 最终提取落子: {move}")
            print(f"🤖 KataGo 评价: 本局最优胜率是 {best_wr} | 模型这步棋的胜率是 {chosen_wr}")
            print("=" * 70 + "\n")
            
        except json.JSONDecodeError:
            print(f"⚠️ 第 {i} 条日志解析失败，可能是写入中断。")

if __name__ == "__main__":
    display_latest_thoughts(3)
