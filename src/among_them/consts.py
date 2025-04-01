NUM_TASKS = 4
NUM_CHATS = 2
NUM_ACTIONS_WITHOUT_REPORT = 7
IMPOSTOR_COOLDOWN = 0
STATE_FILE = "data/game_state.json"

TOKEN_COSTS = {
    # OPENAI
    "openai/gpt-4o": {  # 128k context
        "input_tokens": 2.5,
        "cache_read": 1.25,
        "output_tokens": 10,
    },
    "openai/gpt-4o-mini": {  # 128k context
        "input_tokens": 0.15,
        "cache_read": 0.075,
        "output_tokens": 0.6,
    },
    # CLAUDE
    "anthropic/claude-3.5-sonnet": {  # 200k context
        "input_tokens": 3,
        "cache_read": 0,
        "output_tokens": 15,
    },
    "anthropic/claude-3-5-haiku": {  # 200k context
        "input_tokens": 1,
        "cache_read": 0,
        "output_tokens": 5,
    },
    # GOOGLE
    "google/gemini-flash-1.5": {  # 1M context
        "input_tokens": 0.075,
        "cache_read": 0,
        "output_tokens": 0.3,
    },
    "google/gemini-pro-1.5": {  # 2M context
        "input_tokens": 1.25,
        "cache_read": 0,
        "output_tokens": 5,
    },
    # META
    "meta-llama/llama-3.2-1b-instruct": {  # 131k context
        "input_tokens": 0.01,
        "cache_read": 0,
        "output_tokens": 0.02,
    },
    "meta-llama/llama-3.2-3b-instruct": {  # 131k context
        "input_tokens": 0.03,
        "cache_read": 0,
        "output_tokens": 0.05,
    },
    "meta-llama/llama-3.1-8b-instruct": {  # 16k-131k context
        "input_tokens": 0.05,
        "cache_read": 0,
        "output_tokens": 0.05,
    },
    "meta-llama/llama-3.1-70b-instruct": {  # 33k-131k context
        "input_tokens": 0.34,
        "cache_read": 0,
        "output_tokens": 0.39,
    },
    "meta-llama/llama-3.1-405b-instruct": {  # 33k-131k context
        "input_tokens": 1.79,
        "cache_read": 0,
        "output_tokens": 1.79,
    },
    # MICROSOFT
    "microsoft/wizardlm-2-7b": {  # 32k context
        "input_tokens": 0.055,
        "cache_read": 0,
        "output_tokens": 0.055,
    },
    "microsoft/wizardlm-2-8x22b": {  # 66k context
        "input_tokens": 0.5,
        "cache_read": 0,
        "output_tokens": 0.5,
    },
    # ROLEPLAY
    # 1. mythomax (too small context) 4k
    # 2. llama 8b
    # 3. llama 70b
    # 4. sonnet 3.5
    # 5. mistral-nemo
    # 6. wizard 8x22b
    # 7. wizard 2-7b
    # 8. nous hermes 405b
    # ?. qwen eva
    "mistralai/mistral-nemo": {  # 128k context
        "input_tokens": 0.15,
        "cache_read": 0,
        "output_tokens": 0.15,
    },
    "nousresearch/hermes-3-llama-3.1-405b": {
        "input_tokens": 1.79,
        "cache_read": 0,
        "output_tokens": 2.49,
    },
    "eva-unit-01/eva-qwen-2.5-32b": {  # 32k context
        "input_tokens": 0.5,
        "cache_read": 0,
        "output_tokens": 0.5,
    },
}
