import os
import asyncio
import json
import time
import sys

# ================= 配置区 =================
# 1. 试运行限制：设为 100 用于快速测试跑分
MAX_LIMIT = float('inf')

# 2. 核心路径 (已修改为相对路径，适配 Linux 服务器)
# 注意：确保 Linux 服务器上的 KataGo 可执行文件有运行权限 (chmod +x)
KATAGO_EXE = "./KataGo_engine/katago"  # 去掉了 .exe
CONFIG_FILE = "src/data_acquiring/get_topk_from_katago/analysis.cfg"
MODEL_FILE = "KataGo_engine/KataGo18b9x9.gz"

# 3. 输入输出
INPUT_FILE = "data/json_output.jsonl"
#OUTPUT_FILE = "data/json_output_with_topk_TEST.jsonl" # 加上 TEST 后缀防误覆盖原数据
OUTPUT_FILE = "data/json_output_with_topk_official.jsonl"

# 5. Resume knob: set START_LINE to the 1-based line number to begin reading from INPUT_FILE.
#    Lines before START_LINE are skipped; output is always APPENDED (never overwritten).
#    Example: if the job was interrupted after 757 output lines, set START_LINE = 758.
#    Default (1) = start from the beginning.
START_LINE = 2249

# 4. KataGo crash recovery
MAX_RESTARTS = 9999   # max times to restart KataGo if it crashes (total across all rows)
MAX_ROW_CRASHES = 5  # Fix 2: skip a row after it causes this many consecutive KataGo crashes
RESTART_INTERVAL = 1000  # Fix 4: proactively restart KataGo every N successfully processed rows
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
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    if exists:
        print(f"✅ {name}: 存在 ({abs_path})")
    else:
        print(f"❌ {name}: 找不到! Python 正在寻找的绝对路径是 -> {abs_path}")
        all_passed = False

if not all_passed:
    print("\n🚨 体检未通过！请根据上面的 ❌ 检查服务器上对应的文件是否存在。")
    sys.exit(1)
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


def load_already_processed_ids(output_file):
    """Read output file and return set of IDs already successfully processed."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(obj["id"])
            except json.JSONDecodeError:
                pass
    return done


# Fix 3: generator that streams rows from INPUT_FILE lazily, one at a time.
def stream_input_rows(filepath, start_line, max_limit):
    """Yield parsed JSON rows from filepath starting at start_line, up to max_limit rows."""
    yielded = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if lineno < start_line:
                continue
            if not line.strip():
                continue
            if yielded >= max_limit:
                break
            yield json.loads(line)
            yielded += 1


def count_input_rows(filepath, start_line, max_limit):
    """Fast line count (no JSON parsing) for progress display."""
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if lineno < start_line:
                continue
            if line.strip():
                total += 1
                if total >= max_limit:
                    break
    return total


async def drain_stderr(process, stderr_lines):
    """Background task: continuously read stderr so it never blocks KataGo."""
    try:
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            decoded = line.decode('utf-8', errors='replace').rstrip()
            stderr_lines.append(decoded)
    except Exception:
        pass


async def start_katago():
    process = await asyncio.create_subprocess_exec(
        KATAGO_EXE, "analysis", "-config", CONFIG_FILE, "-model", MODEL_FILE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    return process


async def restart_katago(process, stderr_lines, stderr_task):
    """Gracefully shut down current KataGo and start a fresh one."""
    stderr_task.cancel()
    await asyncio.gather(stderr_task, return_exceptions=True)
    try:
        process.stdin.close()
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass
    await process.wait()
    new_process = await start_katago()
    new_stderr_lines = []
    new_stderr_task = asyncio.ensure_future(drain_stderr(new_process, new_stderr_lines))
    await asyncio.sleep(2)
    return new_process, new_stderr_lines, new_stderr_task


async def query_one(process, query, debug_print=False):
    """
    Send one query to KataGo and return the parsed response dict,
    or None if the process died or sent an error.
    Returns (response_or_None, process_alive: bool)
    """
    query_id = query["id"]
    try:
        process.stdin.write((json.dumps(query) + "\n").encode('utf-8'))
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"\n[ERROR] stdin write failed for id={query_id}: {e}")
        return None, False

    response = None
    while True:
        # Check if process already died before blocking on readline
        if process.returncode is not None:
            print(f"\n[ERROR] KataGo exited (code {process.returncode}) while waiting for id={query_id}")
            return None, False

        try:
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=120)
        except asyncio.TimeoutError:
            print(f"\n[ERROR] Timeout waiting for KataGo response for id={query_id}")
            return None, False
        except (ConnectionResetError, OSError) as e:
            print(f"\n[ERROR] stdout read failed for id={query_id}: {e}")
            return None, False

        if not response_line:
            print(f"\n[ERROR] KataGo stdout closed unexpectedly for id={query_id}")
            return None, False

        raw = response_line.decode('utf-8').strip()
        if debug_print:
            print(f"\n[DEBUG] raw KataGo output: {raw[:500]}")

        try:
            candidate = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"\n[DEBUG] JSON parse error: {e} | line: {raw[:200]}")
            continue

        if "error" in candidate:
            print(f"\n[DEBUG] KataGo error for id={query_id}: {candidate['error']}")
            return None, True  # KataGo reported an error but is still alive

        if candidate.get("isDuringSearch", False) or "warning" in candidate:
            continue

        response = candidate
        break

    return response, True


def parse_response(original_data, response):
    """Build the enriched output dict from a KataGo response."""
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
    original_data.pop("katago_analysis", None)
    return original_data


async def run_analysis():
    print(f"=== KataGo 异步全盘跑分测试 (选项 B) ===")
    print(f"目标数量: {MAX_LIMIT} 局 | Temp: 2.5 | Visits: 2000")

    if START_LINE > 1:
        print(f"[RESUME] START_LINE={START_LINE}: skipping first {START_LINE - 1} input lines, appending to {OUTPUT_FILE}.")

    # Fix 3: count rows without loading them (for progress display only)
    print("[INFO] Counting input rows (no JSON parsing)...")
    total_input = count_input_rows(INPUT_FILE, START_LINE, MAX_LIMIT)
    print(f"[INFO] {total_input} rows to process (input lines {START_LINE}+)")

    start_time = time.time()
    count = 0        # successfully written rows this session
    restarts = 0     # total KataGo restarts (crash + proactive)
    per_row_crashes = 0  # Fix 2: consecutive crashes on the current row

    process = await start_katago()
    stderr_lines = []
    stderr_task = asyncio.ensure_future(drain_stderr(process, stderr_lines))
    print(f"[INFO] KataGo started (PID {process.pid})")
    await asyncio.sleep(2)

    # Fix 3: stream rows one at a time — no giant list in memory
    row_stream = stream_input_rows(INPUT_FILE, START_LINE, MAX_LIMIT)
    current_data = next(row_stream, None)

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        while current_data is not None:

            # Fix 4: proactive restart every RESTART_INTERVAL to flush GPU/shm state
            if count > 0 and count % RESTART_INTERVAL == 0:
                print(f"\n[INFO] Proactive restart after {count} rows (Fix 4)...")
                process, stderr_lines, stderr_task = await restart_katago(process, stderr_lines, stderr_task)
                print(f"[INFO] KataGo restarted proactively (PID {process.pid})")

            query = {
                "id": current_data.get("id"),
                "rules": current_data.get("rules", "chinese"),
                "komi": current_data.get("komi", 6.5),
                "boardXSize": 9, "boardYSize": 9,
                "initialStones": current_data.get("initialStones", []),
                "moves": current_data.get("moves", []),
                "analyzeTurns": current_data.get("analyzeTurns", [len(current_data.get("moves", []))]),
                "includePolicy": True,
                "rootPolicyTemperature": 2.5,
                "maxVisits": 2000
            }

            debug = (count < 3)
            response, alive = await query_one(process, query, debug_print=debug)

            if not alive:
                # KataGo process died — print stderr tail
                if stderr_lines:
                    print("\n[KataGo STDERR tail]:")
                    for l in stderr_lines[-30:]:
                        print("  STDERR:", l)

                restarts += 1
                per_row_crashes += 1

                if restarts > MAX_RESTARTS:
                    print(f"[FATAL] KataGo crashed {restarts} times total, giving up.")
                    break

                # Fix 2: this row keeps killing KataGo — write empty evals and skip it
                if per_row_crashes >= MAX_ROW_CRASHES:
                    print(f"\n[SKIP] id={current_data.get('id')} caused {per_row_crashes} consecutive crashes, skipping.")
                    current_data["katago_evals"] = {}
                    fout.write(json.dumps(current_data, ensure_ascii=False) + "\n")
                    fout.flush()
                    count += 1
                    per_row_crashes = 0
                    current_data = next(row_stream, None)

                print(f"\n[WARN] KataGo died — restarting (attempt {restarts}/{MAX_RESTARTS})...")
                process, stderr_lines, stderr_task = await restart_katago(process, stderr_lines, stderr_task)
                print(f"[INFO] KataGo restarted (PID {process.pid})")
                continue  # retry current_data (or the new one if we just skipped)

            # KataGo alive
            per_row_crashes = 0
            if response is None:
                # KataGo returned an error for this query — write empty evals and skip
                current_data["katago_evals"] = {}
                fout.write(json.dumps(current_data, ensure_ascii=False) + "\n")
            else:
                enriched = parse_response(current_data, response)
                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            fout.flush()

            count += 1
            current_data = next(row_stream, None)

            if count % 10 == 0:
                elapsed = time.time() - start_time
                speed = count / elapsed if elapsed > 0 else 0
                print(f"进度: {START_LINE - 1 + count}/{START_LINE - 1 + total_input} | 速度: {speed:.1f} 图/秒", end='\r')

    # Cleanup
    stderr_task.cancel()
    await asyncio.gather(stderr_task, return_exceptions=True)
    try:
        process.stdin.close()
    except Exception:
        pass
    await process.wait()

    if stderr_lines:
        print("\n[KataGo STDERR tail (final)]:")
        for l in stderr_lines[-20:]:
            print("  STDERR:", l)

    print(f"\n\n=== 完成 ===")
    total_time = time.time() - start_time
    if total_time > 0 and count > 0:
        print(f"总耗时: {total_time:.2f} 秒 | 本次处理: {count} | 平均速度: {count / total_time:.2f} 图/秒")
    print(f"输出文件: {OUTPUT_FILE} (本次新增 {count} 条，从第 {START_LINE} 行起)")


if __name__ == "__main__":
    asyncio.run(run_analysis())
