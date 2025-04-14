import unittest
from among_them.utils.player_utils import (
    get_next_random_player,
    get_last_player_action,
    get_dead_players,
    get_players_in_room
)

from tests.test_utils import load_test_game


class TestPlayerUtils(unittest.TestCase):
    def setUp(self):
        # Load the predefined game state
        self.history, self.players, self.game_config = load_test_game()
        
    def test_get_last_player_action(self):
        # Test getting the last action for a player
        # Using turn 9 (Alice waited in CAFETERIA)
        mid_game_idx = 9
        
        # Find Alice player
        alice = next((p for p in self.players if p.name == "Alice"), None)
        self.assertIsNotNone(alice, "Alice player not found")
        
        # Get last action for Alice
        last_action = get_last_player_action(
            history=self.history[:mid_game_idx+1],
            player=alice
        )
        
        # Verify last action belongs to Alice
        self.assertEqual(last_action.action_taken.player_name, "Alice", 
                        "Last action should belong to Alice")
    
    def test_get_dead_players(self):
        # Test getting dead players
        # Using turn 13 (after David killed Alice)
        after_kill_idx = 13
        
        # Get dead players
        dead_players = get_dead_players(self.history[:after_kill_idx+1])
        
        # Verify dead players dictionary
        self.assertIsInstance(dead_players, dict, "Dead players should be a dictionary")
        
        # Alice should be dead after David killed her
        self.assertIn("Alice", dead_players, "Alice should be in the dead players dictionary")
        self.assertEqual(dead_players["Alice"], "Cafeteria", 
                        "Alice should be dead in the Cafeteria")
        
        # Verify David is not dead after being ejected - he is automatically a ghost
        after_ejection_idx = 20
        dead_players = get_dead_players(self.history[:after_ejection_idx+1])
        self.assertNotIn("David", dead_players, "David should not be in the dead players dictionary")
    
    def test_get_players_in_room(self):
        # Test getting players in a room
        # Using turn 8 (Bob completed task in CAFETERIA)
        room_action_idx = 8
        
        # Find Bob player
        bob = next((p for p in self.players if p.name == "Bob"), None)
        self.assertIsNotNone(bob, "Bob player not found")
        
        # Get players in the same room as Bob
        players_in_room = get_players_in_room(
            history=self.history[:room_action_idx+1],
            players=self.players,
            player=bob
        )
        
        # Verify players in room is a list
        self.assertIsInstance(players_in_room, list, "Players in room should be a list")
        
        # Verify Bob is not in the list (as the function excludes the player parameter)
        self.assertNotIn(bob, players_in_room, "The player parameter should not be in the returned list")
        
        # Check if David, Alice, and Charlie are in the same room as Bob (Cafeteria)
        player_names_in_room = [p.name for p in players_in_room]
        self.assertIn("David", player_names_in_room, "David should be in the Cafeteria with Bob")
        self.assertIn("Alice", player_names_in_room, "Alice should be in the Cafeteria with Bob")
        self.assertIn("Charlie", player_names_in_room, "Charlie should be in the Cafeteria with Bob")
    
    def test_get_next_random_player(self):
        # Test getting the next random player
        # Using turn 5 (Charlie waited in CAFETERIA)
        after_action_idx = 5
        
        # Get next player and remaining players
        next_player, remaining_players = get_next_random_player(
            history=self.history[:after_action_idx+1],
            players=self.players
        )
        
        # Verify next player is a valid player
        self.assertIn(next_player, self.players, "Next player should be a valid player")
        
        # Verify next player is alive
        self.assertIn(next_player.name, self.history[after_action_idx].alive_player_names, 
                     "Next player should be alive")
        
        # Verify remaining players list doesn't contain the next player
        self.assertNotIn(next_player.name, remaining_players, 
                        "Remaining players should not contain the next player")
        
        # Verify all remaining players are alive
        for player_name in remaining_players:
            self.assertIn(player_name, self.history[after_action_idx].alive_player_names, 
                         f"Remaining player {player_name} should be alive")


if __name__ == "__main__":
    unittest.main()
