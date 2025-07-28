# -*- coding: utf-8 -*-
import sys
sys.path.append(".")

from scripts.manual_llm_game import main
from among_them.game_config import GameConfig

# Create the game config
game_config = GameConfig(
    num_tasks=5,
    num_players=7,
    num_impostors=2,
    map_size=1,
    num_task_phase_actions_per_player=10,
    num_discuss_phase_actions_per_player=2,
    impostor_cooldown=1
)

# Run the game with the specific state file path
main(game_config, "src/among_them/data/game_state_temp_18.json")
