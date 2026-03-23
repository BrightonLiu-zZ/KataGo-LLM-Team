# GTP Proxy — User Guide

Guides for connecting the GTP proxy to Lizzie for live gameplay against the GRPO-trained Go AI.

Two proxy versions are available:

- **v4 (Qwen3-8B)** — current; full board-state tracking, `REASONING:` + `MOVE:` format
- **v1 (Qwen2.5-7B)** — legacy; move-history only, `<answer>` format

---

## v4 Proxy (Qwen3-8B) — Recommended

### What is this?

`gtp_proxy_v4.exe` bridges Lizzie and LM Studio. It maintains a full 9x9 board with Go capture logic, renders an ASCII board matching the v4 training format, and sends it to the LLM on each move.

```
┌────────┐   GTP commands    ┌────────────────────┐   POST /v1/chat/completions   ┌────────────────────────────────────┐
│ Lizzie │ ───────────────►  │ gtp_proxy_v4.exe   │ ─────────────────────────►   │ LM Studio (localhost:1234)          │
│  (GUI) │ ◄───────────────  │  (board + capture) │ ◄─────────────────────────   │ qwen3-8b-go-v4-Q4_K_M.gguf         │
└────────┘   GTP responses   └────────────────────┘     JSON response             └────────────────────────────────────┘
```

- The proxy tracks full board state including stone captures.
- Each `genmove` call sends the LLM the ASCII board, move history, valid empty coordinates, and player to move — identical to the v4 training prompt format.
- The LLM responds with `REASONING: ... MOVE: D4`; the proxy extracts the coordinate via regex.
- If the model returns an invalid move or fails to respond, the proxy falls back to `pass`.

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **LM Studio** | Download from [lmstudio.ai](https://lmstudio.ai/) and install |
| **GGUF model** | `qwen3-8b-go-v4-Q4_K_M.gguf` — produced by `export_gguf_model_v4.sh` |
| **gtp_proxy_v4.exe** | Built from `gtp_proxy_v4.cpp` using `build_v4.bat` |
| **Go GUI** | e.g. [Lizzie](https://github.com/featurecat/lizzie/releases) |
| **GPU** | NVIDIA GPU with 8GB+ VRAM |

### Setup

#### Step 1 — Quantize the model

From the repo root (inside Docker for root access):

```bash
bash src/deployment/export_gguf_model_v4.sh
```

This merges the LoRA checkpoint with Qwen3-8B, converts to GGUF, and quantizes to Q4_K_M. Output: `qwen3-8b-go-v4-Q4_K_M.gguf`.

#### Step 2 — Load the model into LM Studio

1. Open **LM Studio**.
2. Click **"My Models"** and add `qwen3-8b-go-v4-Q4_K_M.gguf`.
3. Switch to the **"Local Server"** tab.
4. Select `qwen3-8b-go-v4-Q4_K_M.gguf` from the model dropdown.
5. Click **"Start Server"** and confirm:
   ```
   Server listening on http://localhost:1234
   ```

#### Step 3 — Build the proxy

On Windows with MSVC Build Tools installed:

```bat
cd src\interception
build_v4.bat
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
| Lizzie reports "engine error" after `genmove` | LM Studio not running | Start LM Studio server on port 1234 first |
| Engine always plays `pass` | Wrong model loaded in LM Studio | Ensure `qwen3-8b-go-v4-Q4_K_M.gguf` is loaded |
| Very slow move generation (> 30s) | GPU offloading disabled | In LM Studio, enable GPU offloading (all layers) |

### Customizing port or model name

Edit these constants in `gtp_proxy_v4.cpp` and rebuild:

```cpp
static const char* LM_STUDIO_MODEL = "qwen3-8b-go-v4-Q4_K_M.gguf";
static const char* LM_STUDIO_HOST  = "localhost";
static const int   LM_STUDIO_PORT  = 1234;
```

### Alternative inference backends

Instead of LM Studio, you can use:

1. **llama.cpp server** — built alongside `llama-quantize` during the export pipeline:
   ```bash
   ./llama.cpp/build/bin/llama-server -m qwen3-8b-go-v4-Q4_K_M.gguf --port 1234
   ```
   No code changes needed — same `/v1/chat/completions` API.

2. **Ollama** — install from [ollama.com](https://ollama.com/), create a `Modelfile`, then `ollama serve`. Exposes the same API on port 11434; change `LM_STUDIO_PORT` to `11434` in the source.

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

