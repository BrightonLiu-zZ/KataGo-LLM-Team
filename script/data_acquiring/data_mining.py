import json
from pathlib import Path

PATH_WITH_TOPK = Path(r"D:\katago_old\lizzie\json_output_with_top_k.jsonl")  # path to KataGo output
PATH_WITH_REQS = Path(r"D:\katago_old\lizzie\json_output.jsonl") # your orginal .json file that is sent to KataGO for analysis（includes initialStones/moves）
PATH_OUTPUT    = Path(r"D:\katago_old\lizzie\cleaned_katago_output.jsonl") # output path

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_id_to_req(path_jsonl):
    # read json_output.jsonl, build {id: {"initialStones":..., "moves":...}} map
    id2req = {}
    with path_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
                _id = js.get("id")
                if not _id:
                    continue
                id2req[_id] = {
                    "initialStones": js.get("initialStones", []),
                    "moves": js.get("moves", [])
                }
            except Exception:
                pass
    return id2req

def simplify_one(js_topk, req_info):
    mid = js_topk.get("moveInfos") or []
    if not mid:
        return None

    # sorted by winrate in descending order
    mid = sorted(
        mid,
        key=lambda x: (x.get("winrate", -1.0), x.get("visits", 0)),
        reverse=True
    )

    out = {
        "局面编号": js_topk.get("id"),
        "让子摆放位置信息": req_info.get("initialStones", []) if req_info else [],
        "黑白双方落子位置序列": req_info.get("moves", []) if req_info else [],
        "KataGo模型对于这个局面提供的一些的落点以及其胜率与目差信息": []
    }

    # preserve move/winrate/scoreLead
    for idx, mi in enumerate(mid, start=1):
        move = mi.get("move")
        wr   = to_float(mi.get("winrate"))
        sl   = to_float(mi.get("scoreLead"))
        out["KataGo模型对于这个局面提供的一些的落点以及其胜率与目差信息"].append({
            f"落点{idx}": move,
            "在这里落子后, 你的胜率会是": wr,
            "在这里落子后, 你将领先于对手的目数": sl
        })

    return out

def main():
    # read id -> (initialStones,moves)
    id2req = load_id_to_req(PATH_WITH_REQS)

    # read the file with moveInfos
    n_in, n_out = 0, 0
    with PATH_WITH_TOPK.open("r", encoding="utf-8") as fin, \
         PATH_OUTPUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                js = json.loads(line)
                rid = js.get("id")
                req_info = id2req.get(rid)
                rec = simplify_one(js, req_info)
                if rec is None:
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
            except Exception:
                pass

    print(f"read {n_in} lines，wrote out {n_out} lines -> {PATH_OUTPUT}")

if __name__ == "__main__":
    main()
