"""
Game History Reference:
Players: {'David': 'Impostor', 'Alice': 'Crewmate', 'Bob': 'Crewmate', 'Charlie': 'Crewmate'}
Game loaded from state file with 24 history entries (using new phase structure)
Phases: TASKS, DISCUSS, VOTING

0: [C:4,D:1,A:4,B:4] [TASKS 40] 💬 @CAFETERIA System said: The game started cd=1 next: Charlie, David, Alice, Bob seen by: Charlie, David, Alice, Bob
1: [C:4,D:1,A:4,B:4] [TASKS 39] · @CAFETERIA Charlie waited. next: Alice, David, Bob seen by: Bob, Alice, David
2: [C:4,D:1,A:4,B:4] [TASKS 38] → @MEDBAY David moved to Medbay. next: Alice, Bob seen by: Charlie, Bob, Alice
3: [C:4,D:1,A:4,B:3] [TASKS 37] ✓ @CAFETERIA Bob completed the Start the coffee maker in the cafeteria task. next: Alice seen by: Alice, Charlie
4: [C:4,D:1,A:3,B:3] [TASKS 36] ✓ @CAFETERIA Alice completed the Empty the cafeteria trash task. seen by: Bob, Charlie
5: [C:4,D:1,A:3,B:3] [TASKS 35] · @CAFETERIA Charlie waited. next: David, Bob, Alice seen by: Bob, Alice
6: [C:4,D:1,A:3,B:3] [TASKS 34] → @CAFETERIA David moved to Cafeteria. next: Bob, Alice seen by: Charlie, Bob, Alice
7: [C:4,D:1,A:3,B:3] [TASKS 33] · @CAFETERIA Alice waited. next: Bob seen by: Bob, David, Charlie
8: [C:4,D:1,A:3,B:2] [TASKS 32] ✓ @CAFETERIA Bob completed the Empty the cafeteria trash task. seen by: Alice, David, Charlie
9: [C:4,D:1,A:3,B:2] [TASKS 31] · @CAFETERIA Alice waited. next: Bob, David, Charlie seen by: Bob, David, Charlie
10: [C:4,D:1,A:3,B:2] [TASKS 30] · @CAFETERIA Bob waited. next: David, Charlie seen by: Alice, David, Charlie
11: [C:4,D:1,A:3,B:2] [TASKS 29] · @CAFETERIA Charlie waited. next: David seen by: Bob, Alice, David
12: [C:4,D:1,B:2] [TASKS 28] × Alice @CAFETERIA Alice was killed. cd=1 seen by: Bob, Alice, Charlie
13: [C:4,D:1,B:2] [TASKS 27] ⚑ Alice @CAFETERIA A dead body was reported by Bob. Discussion started. next: Charlie, David seen by: Charlie, David, Bob
14: [C:4,D:1,B:2] [DISCUSS 3] 💬 @CAFETERIA System said: It is discussion phase now. Discuss who to eject from the game. cd=1 next: Charlie, David, Bob seen by: Charlie, David, Bob
15: [C:4,D:1,B:2] [DISCUSS 2] 💬 @CAFETERIA Charlie said: helolo next: David, Bob seen by: Charlie, David, Bob
16: [C:4,D:1,B:2] [DISCUSS 1] 💬 @CAFETERIA David said: hi next: Bob seen by: Charlie, David, Bob
17: [C:4,D:1,B:2] [DISCUSS 0] 💬 @CAFETERIA Bob said: bry seen by: Charlie, David, Bob
18: [C:4,D:1,B:2] [VOTING 3] 💬 @CAFETERIA System said: Discussion ended. Voting phase started. Vote who to eject from the game. cd=1 next: Charlie, David, Bob seen by: Charlie, David, Bob
19: [C:4,D:1,B:2] [VOTING 2] ✔ @CAFETERIA David voted for nobody. next: Charlie, Bob seen by: Bob, Charlie
20: [C:4,D:1,B:2] [VOTING 1] ✔ @CAFETERIA Bob voted for David. next: Charlie seen by: David, Charlie
21: [C:4,B:2] [VOTING 0] ✔ @CAFETERIA Charlie voted for David. seen by: Bob, David
22: [C:4,B:2] [VOTING 0] × David @CAFETERIA David was killed. cd=1 next: Charlie, Bob seen by: Charlie, Bob
23: [C:4,B:2] [TASKS 20] 💬 @CAFETERIA System said: Everyone is in the cafeteria and start from there. It is task phase now. cd=1 next: Charlie, Bob seen by: Charlie, Bob
24: [C:4,B:2] [TASKS 0] 💬 @CAFETERIA System said: The game ended (No impostors left) cd=1 next: Charlie, Bob seen by: Charlie, Bob
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
