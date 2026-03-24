# GTP Proxy — User Guide

Guides for connecting the GTP proxy to Lizzie for live gameplay against the GRPO-trained Go AI.

Three proxy variants are available:

- **v4 (Qwen3-8B fine-tuned)** — current; full board-state tracking, `REASONING:` + `MOVE:` format, reasoning output to GTP Console
- **v4 base (Qwen3-8B un-fine-tuned)** — same proxy logic, points at base model for comparison
- **v1 (Qwen2.5-7B)** — legacy; move-history only, `<answer>` format

---

## v4 Proxy (Qwen3-8B Fine-tuned) — Recommended

### What is this?

`gtp_proxy_v4.exe` bridges Lizzie and LM Studio. It maintains a full 9x9 board with Go capture logic, renders an ASCII board matching the v4 training format, and sends it to the LLM on each move.

```
┌────────┐   GTP commands    ┌────────────────────────┐   POST /v1/chat/completions   ┌────────────────────────────────────┐
│ Lizzie │ ───────────────►  │ gtp_proxy_v4.exe       │ ─────────────────────────►   │ LM Studio (localhost:1234)          │
│  (GUI) │ ◄───────────────  │  (board + capture +    │ ◄─────────────────────────   │ qwen3-8b-go-v4-Q4_K_M.gguf         │
│        │   move (stdout)   │   reasoning on stderr) │     JSON response             │                                    │
│        │   reasoning       │                        │                               │                                    │
└────────┘   (GTP Console)   └────────────────────────┘                               └────────────────────────────────────┘
```

- The proxy tracks full board state including stone captures (flood-fill liberty counting).
- Each `genmove` call sends the LLM the ASCII board, move history, valid empty coordinates, and player to move — identical to the v4 training prompt format.
- The LLM responds with `REASONING: ... MOVE: D4`; the proxy extracts the coordinate via regex.
- The full reasoning text (with `<think>` tags stripped) is written to **stderr**, which Lizzie displays in the **GTP Console**.
- If the model returns an invalid move or fails to respond, the proxy falls back to `pass`.

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **LM Studio** | Download from [lmstudio.ai](https://lmstudio.ai/) and install |
| **GGUF model** | `qwen3-8b-go-v4-Q4_K_M.gguf` — produced by `export_gguf_model_v4.sh` |
| **gtp_proxy_v4.exe** | Built from `gtp_proxy_v4.cpp` (see below) |
| **Go GUI** | e.g. [Lizzie](https://github.com/featurecat/lizzie/releases) |
| **GPU** | NVIDIA GPU with 8GB+ VRAM |

### Setup

#### Step 1 — Quantize the model

From the repo root (inside Docker for root access):

```bash
bash src/deployment/export_gguf_model_v4.sh
```

This merges the LoRA checkpoint with Qwen3-8B, converts to GGUF, and quantizes to Q4_K_M. Output: `qwen3-8b-go-v4-Q4_K_M.gguf` (~5GB).

Transfer the GGUF file to your Windows machine.

#### Step 2 — Load the model into LM Studio

1. Open **LM Studio**.
2. Add `qwen3-8b-go-v4-Q4_K_M.gguf` to your models.
3. Switch to the **Server** tab.
4. Select the model and click **Start Server**.
5. Confirm: `Server listening on http://localhost:1234`.

**Important (LM Studio v0.4.7+):** Models downloaded via LM Studio's search use identifiers like `qwen/qwen3-8b@q4_k_m` (not raw filenames). If the proxy reports a model identifier error, check the error message for the correct format and update `LM_STUDIO_MODEL` in the source.

#### Step 3 — Build the proxy

On Windows, open **Developer Command Prompt for Visual Studio** (search for it in Start menu — regular PowerShell will not find `cl.exe`):

```bat
cd src\interception
cl.exe /std:c++17 /EHsc /utf-8 /W4 /O2 gtp_proxy_v4.cpp /Fe:gtp_proxy_v4.exe /link ws2_32.lib
```

**Alternative (MinGW g++):**
```bat
g++ -std=c++17 -O2 -Wall -o gtp_proxy_v4.exe gtp_proxy_v4.cpp -lws2_32
```

This produces `gtp_proxy_v4.exe`.

#### Step 4 — Configure Lizzie

1. Open Lizzie, go to **File -> Preferences**.
2. In the **Engine** field, paste the absolute path to `gtp_proxy_v4.exe`:
   ```
   C:\tools\gtp_proxy_v4.exe
   ```
3. Leave the arguments field empty.
4. Click **OK** / **Save**, then restart the engine.

Lizzie should report the engine as **`Qwen3-Go-v4 v2.0`**.

### Reasoning output in GTP Console

The proxy writes the model's reasoning to **stderr** before returning the move. Lizzie's GTP Console displays stderr output, so you will see lines like:

```
[Qwen3] REASONING: Black has a strong group on the left side. Playing at E5 would
strengthen the center influence while threatening to connect.
MOVE: E5
```

The `<think>...</think>` blocks (from Qwen3's native thinking mode) are automatically stripped — only the user-facing `REASONING: ... MOVE: ...` text is shown.

### Supported GTP Commands

| Command | Description |
|---------|-------------|
| `name` | Returns `Qwen3-Go-v4` |
| `version` | Returns `2.0` |
| `protocol_version` | Returns `2` |
| `list_commands` | Lists all supported commands |
| `boardsize <N>` | Accepts 9 only; resets the board |
| `clear_board` | Resets board and move history |
| `komi <value>` | Accepted and ignored |
| `known_command <cmd>` | Returns `true` or `false` |
| `play <color> <coord>` | Places a stone (with capture handling) |
| `genmove <color>` | Queries LM Studio and returns the AI's move |
| `quit` | Exits the proxy |

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Engine fails to start in Lizzie | Wrong path to exe | Use absolute path; verify the file exists |
| `cl.exe` not found | Using regular PowerShell | Open **Developer Command Prompt for Visual Studio** instead |
| Lizzie reports "engine error" after `genmove` | LM Studio not running | Start LM Studio server on port 1234 first |
| Engine always plays `pass` | LM Studio thinking mode consuming all tokens | Ensure `max_tokens` is `4096` in proxy source; append `/no_think` to system prompt (already set in v4 proxy) |
| `Invalid model identifier` error in LM Studio logs | LM Studio v0.4.7+ uses `publisher/model@quant` format | Check error message for correct identifier; update `LM_STUDIO_MODEL` in source |
| Model file loads as 0 bytes in LM Studio | Corrupted/incomplete file transfer | Re-transfer the GGUF; verify file size is ~5GB |
| Very slow move generation (> 30s) | GPU offloading disabled | In LM Studio, enable GPU offloading (all layers) |
| No reasoning in GTP Console | Old proxy version without stderr output | Rebuild from latest `gtp_proxy_v4.cpp` |

### Customizing port or model name

Edit these constants in `gtp_proxy_v4.cpp` and rebuild:

```cpp
static const char* LM_STUDIO_MODEL = "qwen3-8b-go-v4-Q4_K_M.gguf";
static const char* LM_STUDIO_HOST  = "localhost";
static const int   LM_STUDIO_PORT  = 1234;
```

**Note:** For models downloaded via LM Studio's search (v0.4.7+), use the `publisher/model@quant` format instead of the filename, e.g. `"qwen/qwen3-8b@q4_k_m"`.

### Alternative inference backends

Instead of LM Studio, you can use:

1. **llama.cpp server** — built alongside `llama-quantize` during the export pipeline:
   ```bash
   ./llama.cpp/build/bin/llama-server -m qwen3-8b-go-v4-Q4_K_M.gguf --port 1234
   ```
   No code changes needed — same `/v1/chat/completions` API.

2. **Ollama** — install from [ollama.com](https://ollama.com/), create a `Modelfile`, then `ollama serve`. Exposes the same API on port 11434; change `LM_STUDIO_PORT` to `11434` in the source.

---

## v4 Base Proxy (Qwen3-8B Un-fine-tuned) — For Comparison

`gtp_proxy_v4_base.exe` is identical to the v4 proxy above except it points at the **un-fine-tuned base Qwen3-8B** model. This lets you compare the base model's Go play against the fine-tuned version in Lizzie.

### Setup

1. **Download the base model GGUF:** Get `Qwen3-8B-Q4_K_M.gguf` from [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) on HuggingFace (~5GB). Or search for `Qwen3-8B` in LM Studio and download the Q4_K_M variant.

2. **Build the base proxy** (Developer Command Prompt):
   ```bat
   cd src\interception
   cl.exe /std:c++17 /EHsc /utf-8 /W4 /O2 gtp_proxy_v4_base.cpp /Fe:gtp_proxy_v4_base.exe /link ws2_32.lib
   ```

3. **Load the base model** in LM Studio and start the server.

4. **Configure Lizzie** to point at `gtp_proxy_v4_base.exe`.

**Note:** LM Studio can only load one model at a time on 8GB VRAM. To switch between fine-tuned and base models, swap the loaded model in LM Studio and change the Lizzie engine accordingly.

**Model identifier:** The base proxy uses `"qwen/qwen3-8b@q4_k_m"` as the model identifier (matching LM Studio v0.4.7+ format). If this doesn't work, check the error message in LM Studio's Developer Logs for the correct identifier and update `LM_STUDIO_MODEL` in `gtp_proxy_v4_base.cpp`.

---

## v1 Proxy (Qwen2.5-7B) — Legacy

The original proxy (`gtp_proxy.exe`) works with `qwen-7b-go-Q4_K_M.gguf`. It sends only move history (no board visualization) and expects `<answer>...</answer>` tags in the response.

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **GGUF model** | `qwen-7b-go-Q4_K_M.gguf` — download from [Hugging Face](https://huggingface.co/brightonliuzZ/qwen2.5-7b-go-GGUF/tree/main) |
| **gtp_proxy.exe** | Built from `gtp_proxy.cpp` using `build.bat` |

### Building

```bat
cd src\interception
build.bat
```

### Configuring Lizzie

Same as v4, but point to `gtp_proxy.exe` and load `qwen-7b-go-Q4_K_M.gguf` in LM Studio.

Engine reports as **`Qwen-Edge-Engine v1.0`**.
