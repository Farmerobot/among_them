# Among Them

A text-based social deduction game where AI agents play a game similar to "Among Us". AI agents use Large Language Models (LLMs) to make decisions, reason about the game state, and interact with other players.

## Quick Start

### Prerequisites

- Python 3.8+ with Poetry
- Ollama (for local inference) or OpenRouter API key

### Installation

1. Install dependencies:
```bash
poetry install
```

2. Set up Ollama (if using local inference):
```bash
# Install Ollama from https://ollama.ai
# Pull a reasoning model
ollama pull deepseek-r1:14b
```

3. Configure environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
```

### Running a Game

```bash
python .\scripts\manual_llm_game.py
```

The game will start with AI-controlled players making decisions. You can interrupt at any time with `Ctrl+C` to take manual control or exit.

## Configuration

All configuration is done through environment variables in the `.env` file.

### Required Variables

#### `LLM_BACKEND`
**Values:** `ollama`, `mlx`, or `openrouter`  
**Default:** `ollama`

Selects which LLM backend to use for AI agents:
- `ollama` - Local inference using Ollama (recommended for most users)
- `mlx` - Local inference using MLX (macOS with Apple Silicon only)
- `openrouter` - Cloud inference via OpenRouter API

**Example:**
```bash
LLM_BACKEND=ollama
```

### Optional Variables

#### `MODEL_NAME`
**Default:** Backend-specific (see below)

Specifies the model to use. If not set, uses backend-specific defaults:
- **Ollama default:** `deepseek-r1:14b`
- **MLX default:** `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit`
- **OpenRouter default:** `deepseek/deepseek-r1:free`

You can use any model compatible with your backend. For Ollama, you can also use GGUF models from Hugging Face:

**Examples:**
```bash
# Use Ollama with a GGUF model from Hugging Face
MODEL_NAME=hf.co/Farmerobot/deepseek-r1-among-them-gguf

# Use a different Ollama model
MODEL_NAME=deepseek-r1:1.5b

# Use OpenRouter with a specific model
MODEL_NAME=deepseek/deepseek-chat
```

#### `OPENROUTER_API_KEY`
**Required when:** `LLM_BACKEND=openrouter`

Your OpenRouter API key for cloud inference.

**Example:**
```bash
OPENROUTER_API_KEY=sk-or-v1-abc123...
```

Get your API key at: https://openrouter.ai/keys

#### `OPENAI_API_KEY`
**Required for:** `trace_analyzer.py` script only (optional workflow)

Your OpenAI API key, used only if you want to evaluate dataset quality with GPT-4o-mini.

**Example:**
```bash
OPENAI_API_KEY=sk-abc123...
```

#### `STATE_FILE`
**Default:** `data/game_state.json`

Path where the game state is saved. Useful if you want to continue interrupted games.

**Example:**
```bash
STATE_FILE=data/my_game.json
```

### Complete Configuration Example

```bash
# Use Ollama with a custom GGUF model
LLM_BACKEND=ollama
MODEL_NAME=hf.co/Farmerobot/deepseek-r1-among-them-gguf

# Or use OpenRouter
# LLM_BACKEND=openrouter
# MODEL_NAME=deepseek/deepseek-r1:free
# OPENROUTER_API_KEY=sk-or-v1-abc123...

# Optional: Custom state file location
STATE_FILE=data/game_state.json
```

## Creating Training Datasets

The project includes tools to generate fine-tuning datasets in Alpaca format from completed games.

### Workflow Overview

1. **Run games** → Saves game traces as JSON files in `data/`
2. **Generate full dataset** → Extracts all conversations
3. **Evaluate traces** (optional) → Score trace quality with GPT-4o-mini
4. **Generate sampled dataset** (optional) → Filter high-quality traces only

### Step 1: Run Games

Run multiple games to generate training data:

```bash
# Run a single game
python .\scripts\manual_llm_game.py

# Or run multiple games in batch
python .\scripts\batch_game_runner.py
```

Games are saved as JSON files in the `data/` directory (e.g., `data/game_state_15.json`).

### Step 2: Generate Full (Unsampled) Dataset

Extract all conversations from game traces into Alpaca format:

```bash
python .\scripts\create_sft_dataset.py
```

**What it does:**
- Processes all JSON files in `data/` folder
- Extracts multi-turn conversations for each player
- Generates token statistics and visualizations
- Creates train/eval splits (80%/20%)

**Output files:**
- `data/alpaca/among_them_train.json` - Training set
- `data/alpaca/among_them_eval.json` - Evaluation set
- `data/alpaca/dataset_info.json` - Dataset metadata
- `generated/token_usage_distribution.png` - Token distribution plots
- `generated/alpaca_dataset_stats.md` - Detailed statistics

**Configuration (in script):**
```python
use_actual_tokenizer = True  # Use real tokenizer for accurate counts
tokenizer_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_OUTPUT_TOKEN_RESPONSE_CUTOFF = 3000  # Filter very long responses
```

### Step 3: Evaluate Traces (Optional)

Use GPT-4o-mini to score trace quality on a scale of 0-3:

```bash
# First, set your OpenAI API key in .env
echo "OPENAI_API_KEY=sk-abc123..." >> .env

# Run the trace analyzer
python .\scripts\trace_analyzer.py
```

**What it does:**
- Evaluates each trace for quality (0=hallucination, 3=outstanding)
- Creates `generated/trace_analysis.txt` with scores

**Scoring criteria:**
- **0** - Hallucination or invalid output
- **1** - Poor quality or inconsistent
- **2** - Good quality (minimum for sampled dataset)
- **3** - Excellent quality

**Note:** This requires creating the CSV dataset first (outdated workflow). You may need to adapt the script for the new multi-turn conversation format.

### Step 4: Generate Sampled (High-Quality) Dataset

Filter the dataset to include only high-quality traces (score ≥ 2):

```bash
python .\scripts\create_sampled_sft_dataset.py
```

**What it does:**
- Reads `generated/trace_analysis.txt` scores
- Filters traces with score ≥ `TRACE_QUALITY_THRESHOLD` (default: 2)
- Creates sampled train/eval splits

**Output files:**
- `data/alpaca_sampled/among_them_train.json` - High-quality training set
- `data/alpaca_sampled/among_them_eval.json` - High-quality evaluation set
- `data/alpaca_sampled/dataset_info.json` - Dataset metadata
- `generated/alpaca_sampled_dataset_stats.md` - Statistics with quality distribution

**Configuration (in script):**
```python
TRACE_QUALITY_THRESHOLD = 2  # Minimum score to include
```

### Dataset Format

Generated datasets use the Alpaca/ShareGPT multi-turn conversation format:

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "content": "<game context and prompt>"
      },
      {
        "role": "assistant",
        "content": "<think>reasoning...</think>action"
      },
      {
        "role": "user",
        "content": "<incremental observations>"
      },
      {
        "role": "assistant",
        "content": "<think>reasoning...</think>action"
      }
    ]
  }
]
```

## Game Features

- **Spatial Navigation**: Players move between interconnected rooms
- **Role-Based Actions**: Different actions for crewmates vs impostors
- **Discussion & Voting**: Social deduction through conversation and voting
- **Chain-of-Thought Reasoning**: AI agents show their reasoning in `<think>` blocks
- **Multiple LLM Backends**: Local (Ollama, MLX) or cloud (OpenRouter) inference
- **State Persistence**: Games can be saved and resumed

## Scripts Overview

- `manual_llm_game.py` - Play a game with AI agents (main entry point)
- `batch_game_runner.py` - Run multiple games automatically
- `create_sft_dataset.py` - Generate full Alpaca dataset from game traces
- `create_sampled_sft_dataset.py` - Generate high-quality filtered dataset
- `trace_analyzer.py` - Evaluate trace quality with GPT-4o-mini
- `streamlit.py` - Web UI for game visualization

## Project Structure

```
among_them/
├── src/among_them/          # Core game engine
│   ├── config.py            # Configuration and LLM backend setup
│   ├── game_engine.py       # Main game loop and logic
│   ├── models/              # Game entities (Player, Action, etc.)
│   └── utils/               # Helper functions (prompts, LLM utils)
├── scripts/                 # Utility scripts
├── data/                    # Game traces and datasets
│   ├── *.json              # Saved game files
│   ├── alpaca/             # Full dataset
│   └── alpaca_sampled/     # High-quality filtered dataset
└── generated/              # Generated statistics and plots
```