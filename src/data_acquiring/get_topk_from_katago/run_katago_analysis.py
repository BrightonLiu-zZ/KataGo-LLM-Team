import asyncio
import json
import time
import sys
import os # <--- 新增 os

# ================= 配置区 =================
# 1. 试运行限制：设为 100 用于快速测试跑分
MAX_LIMIT = 100

# 2. 核心路径 (已修改为相对路径，适配 Linux 服务器)
# 注意：确保 Linux 服务器上的 KataGo 可执行文件有运行权限 (chmod +x)
KATAGO_EXE = "./KataGo_engine/katago"  # 去掉了 .exe
CONFIG_FILE = "src/data_acquiring/get_topk_from_katago/analysis.cfg"
MODEL_FILE = "KataGo_engine/KataGo18b9x9.gz"

# 3. 输入输出
INPUT_FILE = "data/json_output.jsonl"
OUTPUT_FILE = "data/json_output_with_topk_TEST.jsonl" # 加上 TEST 后缀防误覆盖原数据
# ===========================================

# ================= 🚨 路径排查小助手 🚨 =================
print("=== 正在执行路径体检 ===")
paths_to_check = {
    "KataGo 引擎 (KATAGO_EXE)": KATAGO_EXE,
    "配置文件 (CONFIG_FILE)": CONFIG_FILE,
    "权重模型 (MODEL_FILE)": MODEL_FILE,
    "输入数据 (INPUT_FILE)": INPUT_FILE
}

all_passed = True
for name, path in paths_to_check.items():
    abs_path = os.path.abspath(path) # 获取服务器上的绝对路径
    exists = os.path.exists(abs_path)
    if exists:
        print(f"✅ {name}: 存在 ({abs_path})")
    else:
        print(f"❌ {name}: 找不到! Python 正在寻找的绝对路径是 -> {abs_path}")
        all_passed = False

if not all_passed:
    print("\n🚨 体检未通过！请根据上面的 ❌ 检查服务器上对应的文件是否存在。")
    sys.exit(1) # 直接退出，不往下跑了
print("=== 体检通过，准备启动 KataGo ===\n")
# ========================================================

def map_policy_to_coords(policy_array):
    """将 KataGo 的 1D Policy 数组映射为 2D 坐标字典"""
    coords_dict = {}
    cols = "ABCDEFGHJ"
    for i in range(81):
        x = i % 9
        y = 8 - (i // 9)
        coords_dict[f"{cols[x]}{y+1}"] = policy_array[i]
    coords_dict["PASS"] = policy_array[81]
    return coords_dict

async def run_analysis():
    print(f"=== KataGo 异步全盘跑分测试 (选项 B) ===")
    print(f"目标数量: {MAX_LIMIT} 局 | Temp: 2.5 | Visits: 5000")
    
    try:
        process = await asyncio.create_subprocess_exec(
            KATAGO_EXE, "analysis", "-config", CONFIG_FILE, "-model", MODEL_FILE,
            stdin=asyncio.subprocess.PIPE, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        print(f"错误: 找不到 {KATAGO_EXE}，请检查路径或执行权限。")
        return

    print("引擎已启动，开始火力全开...")
    start_time = time.time()
    count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            if count >= MAX_LIMIT: break
                
            original_data = json.loads(line)
            
            # === 构造选项 B 的魔法请求 ===
            query = {
                "id": original_data.get("id"),
                "rules": original_data.get("rules", "chinese"),
                "komi": original_data.get("komi", 6.5),
                "boardXSize": 9, "boardYSize": 9,
                "initialStones": original_data.get("initialStones", []),
                "moves": original_data.get("moves", []),
                "analyzeTurns": original_data.get("analyzeTurns", [len(original_data.get("moves", []))]),
                
                # 👇 选项 B 的核心参数
                "reportPolicy": True,
                "rootPolicyTemperature": 2.5,
                "maxVisits": 5000
            }
            
            process.stdin.write((json.dumps(query) + "\n").encode('utf-8'))
            await process.stdin.drain()
            
            response_line = await process.stdout.readline()
            if not response_line: break
            response = json.loads(response_line.decode('utf-8'))
            
            # === 解析全盘数据 ===
            katago_evals = {}
            if "policy" in response:
                policy_dict = map_policy_to_coords(response["policy"])
                for coord, pol in policy_dict.items():
                    katago_evals[coord] = {"policy": pol, "winrate": None, "scoreLead": None}
            
            if "moveInfos" in response:
                original_data["root_winrate"] = response["rootInfo"]["winrate"]
                original_data["root_scoreLead"] = response["rootInfo"]["scoreLead"]
                for info in response["moveInfos"]:
                    mv = info.get("move")
                    if mv in katago_evals:
                        katago_evals[mv]["winrate"] = info.get("winrate")
                        katago_evals[mv]["scoreLead"] = info.get("scoreLead")

            original_data["katago_evals"] = katago_evals
            
            # 清理旧字段（如果存在）
            original_data.pop("katago_analysis", None)
            
            fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
            
            count += 1
            if count % 10 == 0:
                elapsed = time.time() - start_time
                speed = count / elapsed if elapsed > 0 else 0
                print(f"进度: {count}/{MAX_LIMIT} | 速度: {speed:.1f} 图/秒", end='\r')

    process.stdin.close()
    print(f"\n\n=== 测试完成 ===")
    total_time = time.time() - start_time
    print(f"总耗时: {total_time:.2f} 秒 | 平均速度: {count / total_time:.2f} 图/秒")
    print(f"输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(run_analysis())