# GTP Proxy — User Guide

A step-by-step guide to connecting `gtp_proxy.exe` to any standard Go GUI (e.g. Lizzie) for live gameplay against the GRPO-aligned Qwen-7B Go AI.

---

## What is this?

`gtp_proxy.exe` is a lightweight, standalone C++ binary that bridges a Go GUI and your locally-hosted LLM. It speaks the **Go Text Protocol (GTP)** on one side (towards the GUI) and makes **REST API calls** to an LM Studio server on the other side to generate moves.

```
┌────────┐   GTP commands    ┌───────────────┐   POST /v1/chat/completions   ┌───────────────────────────┐
│ Lizzie │ ───────────────►  │ gtp_proxy.exe │ ─────────────────────────►   │ LM Studio (localhost:1234) │
│  (GUI) │ ◄───────────────  │  (this proxy) │ ◄─────────────────────────   │ qwen-7b-go-Q4_K_M.gguf    │
└────────┘   GTP responses   └───────────────┘     JSON response             └───────────────────────────┘
```

- The GUI does **not** know there is a proxy — it communicates over standard GTP exactly as with any traditional engine.
- The proxy is **stateless per connection**: it maintains the current game's move history in memory and reconstructs the board context for each LLM call.
- The LLM responds inside `<answer>…</answer>` tags; the proxy extracts the coordinate and returns it as a GTP move response.

---

## Prerequisites

Before starting, ensure the following are ready:

| Requirement | Details |
|-------------|---------|
| **LM Studio** | Download from [lmstudio.ai](https://lmstudio.ai/) and install |
| **GGUF model** | `qwen-7b-go-Q4_K_M.gguf` — download from [Hugging Face](https://huggingface.co/brightonliuzZ/qwen2.5-7b-go-GGUF/tree/main) |
| **gtp_proxy.exe** | Download `gtp_proxy_windows.zip` from [GitHub Releases](https://github.com/BrightonLiu-zZ/KataGo-LLM-Team/releases) and extract |
| **Go GUI** | e.g. [Lizzie](https://github.com/featurecat/lizzie/releases) |
| **GPU** | NVIDIA GPU with 8GB+ VRAM (RTX 4060 or equivalent) |

---

## Setup

### Step 1 — Load the Model into LM Studio

1. Open **LM Studio**.
2. Click **"My Models"** and add `qwen-7b-go-Q4_K_M.gguf`.
3. Switch to the **"Local Server"** tab (server icon in the left sidebar).
4. Select `qwen-7b-go-Q4_K_M.gguf` from the model dropdown.
5. Click **"Start Server"** and confirm the log shows:
   ```
   Server listening on http://localhost:1234
   ```

> **Important:** The server must remain running throughout your game session. The proxy connects to `localhost:1234` on every `genmove` call.

---

### Step 2 — Place `gtp_proxy.exe`

Extract the downloaded zip and place `gtp_proxy.exe` anywhere accessible on your machine. Note the **absolute path** — you will need it in the next step. Example:

```
C:\tools\gtp_proxy.exe
```

---

### Step 3 — Configure Lizzie

1. Open Lizzie, go to **File → Preferences** (or the engine settings gear icon).
2. In the **Engine** field, paste the **absolute path** to `gtp_proxy.exe`:
   ```
   C:\tools\gtp_proxy.exe
   ```
3. **Leave the arguments field completely empty.** Do not add `gtp`, `-model`, or any other flags. The proxy is self-contained and requires no arguments.
4. Click **OK** / **Save**, then restart the engine within Lizzie.

You should see Lizzie report the engine as **`Qwen-Edge-Engine v1.0`** in the engine info panel.

---

## Supported GTP Commands

The proxy implements the full set of GTP commands required for live play:

| Command | Description |
|---------|-------------|
| `name` | Returns `Qwen-Edge-Engine` |
| `version` | Returns `1.0` |
| `protocol_version` | Returns `2` |
| `list_commands` | Lists all supported commands |
| `boardsize <N>` | Sets board size (model is optimised for 9×9) |
| `clear_board` | Resets game state |
| `play <color> <coord>` | Records an opponent move (e.g. `play B D4`) |
| `genmove <color>` | Queries LM Studio and returns the AI's chosen move |
| `quit` | Exits the proxy |

> On `genmove`, the proxy sends the full game history and current player to LM Studio, waits up to **120 seconds** for a response, extracts the coordinate from `<answer>…</answer>` tags, records the move internally, and replies to the GUI. If the LLM fails to respond or does not format correctly, the proxy returns `pass` as a safe fallback.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Engine fails to start in Lizzie | Wrong path to `gtp_proxy.exe` | Use absolute path; verify the file exists |
| Lizzie reports "engine error" after `genmove` | LM Studio server not running | Start the LM Studio server on port 1234 before launching Lizzie |
| Engine always plays `pass` | LLM response format mismatch | Ensure `qwen-7b-go-Q4_K_M.gguf` is loaded in LM Studio (not a different model) |
| Very slow move generation (> 30s) | GPU not being used for inference | In LM Studio, confirm GPU offloading is enabled (all layers to GPU) |
| Port 1234 already in use | Another service on the same port | Change LM Studio's server port and rebuild the proxy with the updated port, or free port 1234 |

---

## Building from Source (Optional)

If you need to rebuild the proxy (e.g. to change the LM Studio port), use the included CMake project:

```bash
# Windows — using MSVC (Developer Command Prompt)
cd src/interception
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

Or use the convenience script:

```powershell
# From src/interception/
.\build.ps1
```

The binary will be output to `src/interception/build/Release/gtp_proxy.exe`.

To change the LM Studio port or model name, edit the following lines in `gtp_proxy.cpp` before building:

```cpp
httplib::Client cli("localhost", 1234);          // ← change port here
{"model", "qwen-7b-go-Q4_K_M.gguf"},            // ← change model name here
```

