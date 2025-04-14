"""
Game History Reference:
Players: {'David': 'Impostor', 'Alice': 'Crewmate', 'Bob': 'Crewmate', 'Charlie': 'Crewmate'}
Game loaded from state file with 22 history entries
0 GamePhase.GAME_START(0) [The game started(1), Location.CAFETERIA] next: David, Alice, Charlie, Bob||David, Alice, Charlie, Bob saw it
1 GamePhase.TASK(39) [Charlie waited(0), Location.CAFETERIA] next: David, Alice, Bob||David, Alice, Bob saw it
2 GamePhase.TASK(38) [David moved to location Medbay(0), Location.MEDBAY] next: Alice, Bob||Bob, Charlie, Alice saw it
3 GamePhase.TASK(37) [Bob completed task: Start the coffee maker in the cafeteria(0), Location.CAFETERIA] next: Alice||Alice, Charlie saw it
4 GamePhase.TASK(36) [Alice completed task: Empty the cafeteria trash(0), Location.CAFETERIA] next: ||Charlie, Bob saw it
5 GamePhase.TASK(35) [Charlie waited(0), Location.CAFETERIA] next: David, Alice, Bob||Alice, Bob saw it
6 GamePhase.TASK(34) [David moved to location Cafeteria(0), Location.CAFETERIA] next: Alice, Bob||Bob, Charlie, Alice saw it
7 GamePhase.TASK(33) [Alice waited(0), Location.CAFETERIA] next: Bob||David, Charlie, Bob saw it
8 GamePhase.TASK(32) [Bob completed task: Empty the cafeteria trash(0), Location.CAFETERIA] next: ||David, Alice, Charlie saw it
9 GamePhase.TASK(31) [Alice waited(0), Location.CAFETERIA] next: David, Charlie, Bob||David, Charlie, Bob saw it
10 GamePhase.TASK(30) [Bob waited(0), Location.CAFETERIA] next: David, Charlie||David, Alice, Charlie saw it
11 GamePhase.TASK(29) [Charlie waited(0), Location.CAFETERIA] next: David||David, Alice, Bob saw it
12 GamePhase.TASK(28) [David killed Alice- Alice(0), Location.CAFETERIA] next: ||Alice, Charlie, Bob saw it
13 GamePhase.TASK(27) [Bob reported dead body of Alice to everyone and started discussion(0), Location.CAFETERIA] next: David, Charlie||David, Charlie, Bob saw it
14 GamePhase.DISCUSS(2) [[Charlie]: helolo(0), Location.CAFETERIA] next: Bob, David||Bob, David, Charlie saw it
15 GamePhase.DISCUSS(1) [[David]: hi(0), Location.CAFETERIA] next: Bob||Bob, David, Charlie saw it
16 GamePhase.DISCUSS(0) [[Bob]: bry(0), Location.CAFETERIA] next: ||Bob, David, Charlie saw it
17 GamePhase.VOTING(2) [David voted for nobody(0), Location.CAFETERIA] next: Charlie, Bob||David, Charlie, Bob saw it
18 GamePhase.VOTING(1) [Bob voted for David(0), Location.CAFETERIA] next: Charlie||David, Charlie, Bob saw it
19 GamePhase.VOTING(0) [Charlie voted for David(0), Location.CAFETERIA] next: ||David, Charlie, Bob saw it
20 GamePhase.VOTE_RESULTS(0) [David was voted out.- David(1), Location.CAFETERIA] next: ||David, Charlie, Bob saw it
21 GamePhase.GAME_END(0) [The game ended (No impostors left)(1), Location.CAFETERIA] next: ||David, Alice, Charlie, Bob saw it
"""

import json
import os
from typing import List, Tuple

from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.player import Player
from among_them.game_jsonencoder import game_object_hook


def load_test_game() -> Tuple[List[History], List[Player], GameConfig]:
    """
    Load the predefined game state from test_game.json.
    
    Returns:
        A tuple containing (history, players, game_config)
    """
    test_file_path = os.path.join(os.path.dirname(__file__), "test_game.json")
    with open(test_file_path, 'r') as f:
        json_str = f.read()
        loaded_data = json.loads(json_str, object_hook=game_object_hook)
        
        if len(loaded_data) == 3:
            history, players, game_config = loaded_data
            return history, players, game_config
        elif len(loaded_data) == 2:
            history, players = loaded_data
            return history, players, GameConfig()
        else:
            raise ValueError("Invalid test game file format")
