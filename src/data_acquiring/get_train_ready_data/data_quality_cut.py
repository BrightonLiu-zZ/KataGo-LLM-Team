import json
import os
import statistics
import random
from collections import defaultdict

# 配置参数
INPUT_FILE = r'C:\git_repo\KataGo-LLM-Team\data\training_ready_data_shuffled.jsonl'
OUTPUT_FILE = r'C:\git_repo\KataGo-LLM-Team\data\training_data_quality_filtered.jsonl'

TARGET_COUNT = 312  # 最终目标数量 (312 * 16 ≈ 5000)

# 基础阈值 (和你之前的一样)
COMPETITIVE_MIN = 0.20
COMPETITIVE_MAX = 0.80
CRITICAL_REGRET = 0.15

# 高级过滤器阈值
MOVE_MIN = 8   # 过滤掉开局定式
MOVE_MAX = 60  # 过滤掉垃圾填子时间

def parse_move_number(original_id):
    """从 original_id (例如 '..._pos34') 中提取手数"""
    try:
        if '_pos' in original_id:
            return int(original_id.split('_pos')[-1])
        return -1
    except:
        return -1

def calculate_sharpness(katago_all):
    """
    计算'尖锐度' (Tactical Sharpness)
    Sharpness = Top1胜率 - (Top2...Top5的平均胜率)
    值越大，说明最佳一手与次优手的差距越大，局面越需要精确计算。
    """
    if not katago_all or len(katago_all) < 2:
        return 0.0
    
    winrates = sorted(list(katago_all.values()), reverse=True)
    best_wr = winrates[0]
    
    # 取前 5 手 (如果不足 5 手则取全部) 的次优手
    other_moves = winrates[1:min(5, len(winrates))]
    avg_other_wr = statistics.mean(other_moves)
    
    return best_wr - avg_other_wr

def curate_dataset(input_path, output_path, target_count):
    print(f"开始高级清洗: {input_path} -> 目标: {target_count} 条 ...\n")
    
    candidates = []
    
    # --- 1. 第一轮扫描：基础过滤 + 特征提取 ---
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip(): continue
            record = json.loads(line)
            
            katago_all = record.get("katago_all", {})
            if not katago_all: continue
            
            # 计算基础指标
            winrates = sorted(list(katago_all.values()), reverse=True)
            best_wr = winrates[0]
            regret = (winrates[0] - winrates[1]) if len(winrates) > 1 else 0
            
            # 基础过滤: Competitive OR Critical
            is_competitive = (COMPETITIVE_MIN <= best_wr <= COMPETITIVE_MAX)
            is_critical = (regret > CRITICAL_REGRET)
            
            if not (is_competitive or is_critical):
                continue

            # 高级特征: 手数 (Move Number)
            move_num = parse_move_number(record.get("original_id", ""))
            
            # 高级过滤 1: 剔除过早或过晚的局面
            if move_num < MOVE_MIN or move_num > MOVE_MAX:
                continue
                
            # 计算高级指标: 尖锐度 (Sharpness)
            sharpness = calculate_sharpness(katago_all)
            
            # 将合格候选者加入列表，保留所有元数据以便分桶
            candidates.append({
                "record": record,
                "best_wr": best_wr,
                "move_num": move_num,
                "sharpness": sharpness
            })

    print(f"第一轮基础过滤后剩余: {len(candidates)} 条。开始分层采样...")
    
    # --- 2. 第二轮：分层采样 (Stratified Sampling) ---
    # 我们将数据分为 3x3 = 9 个桶 (Bucket)
    # 维度 A: 游戏阶段 (Early, Mid, Late)
    # 维度 B: 局势 (Black Adv, Even, White Adv)
    
    buckets = defaultdict(list)
    
    for item in candidates:
        # 维度 A: 游戏阶段
        if item['move_num'] <= 20: phase = 'Early'
        elif item['move_num'] <= 40: phase = 'Mid'
        else: phase = 'Late'
        
        # 维度 B: 局势 (注意: Katago胜率通常是黑棋视角? 假设 >0.5 黑优)
        # 这里为了简单，我们按绝对值分: 优势方、均势、劣势方
        # 但为了训练 Policy，我们需要多样性。
        if item['best_wr'] > 0.6: balance = 'BlackAdv'
        elif item['best_wr'] < 0.4: balance = 'WhiteAdv'
        else: balance = 'Even'
        
        bucket_key = f"{phase}_{balance}"
        buckets[bucket_key].append(item)
    
    # 计算每个桶应该取多少条数据 (平均分配)
    # 9个桶，目标312条 -> 每个桶约 35 条
    num_buckets = len(buckets)
    per_bucket_target = target_count // num_buckets
    
    final_selection = []
    
    print(f"分桶情况 ({num_buckets} 个非空桶):")
    for key, items in buckets.items():
        # 在每个桶内，按 'Sharpness' (尖锐度) 降序排序
        # 优先保留那些“只有一手正解”的各种局面的数据
        items.sort(key=lambda x: x['sharpness'], reverse=True)
        
        # 取 Top-K
        selected = items[:per_bucket_target]
        final_selection.extend([x['record'] for x in selected])
        
        print(f"  - Bucket [{key:<15}]: 总数 {len(items):<4} -> 选取 {len(selected)}")
        
    # 如果还没凑够 (因为有的桶可能数据不够)，从剩余数据中再补齐 Sharpness 最高的
    if len(final_selection) < target_count:
        needed = target_count - len(final_selection)
        print(f"提示: 桶内数据不足，补充 {needed} 条全剧最高 Sharpness 数据...")
        
        # 收集所有未被选中的
        selected_ids = set(id(x) for x in final_selection) # 简单去重
        remaining = [x for x in candidates if id(x['record']) not in selected_ids]
        remaining.sort(key=lambda x: x['sharpness'], reverse=True)
        
        final_selection.extend([x['record'] for x in remaining[:needed]])

    # --- 3. 写入结果 ---
    with open(output_path, 'w', encoding='utf-8') as fout:
        for record in final_selection:
            fout.write(json.dumps(record) + '\n')
            
    print("\n" + "="*30)
    print(f"优中选优完成!")
    print(f"最终输出: {len(final_selection)} 条高价值数据")
    print(f"保存路径: {output_path}")
    print("="*30)

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
    else:
        curate_dataset(INPUT_FILE, OUTPUT_FILE, TARGET_COUNT)