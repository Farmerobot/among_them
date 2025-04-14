import unittest

from among_them.models.action_type import ActionType
from among_them.models.end_game import EndGameReason
from among_them.models.phase import GamePhase
from among_them.utils.history_utils import (
    initialize_history,
    end_game_history,
    create_vote_history_entry,
    get_action_history_str
)

from tests.test_utils import load_test_game


class TestHistoryUtils(unittest.TestCase):
    def setUp(self):
        # Load the predefined game state
        self.history, self.players, self.game_config = load_test_game()
        
    def test_initialize_history(self):
        # Test initializing a new game history
        
        # Initialize new history
        new_history = initialize_history(self.players, self.game_config)
        
        # Verify new history is a list with one item
        self.assertIsInstance(new_history, list, "New history should be a list")
        self.assertEqual(len(new_history), 1, "New history should have one item")
        
        # Verify the first history item has the correct phase
        self.assertEqual(new_history[0].phase, GamePhase.GAME_START, 
                        "First history item should have GAME_START phase")
        
        # Verify all players are alive
        self.assertEqual(len(new_history[0].alive_player_names), len(self.players), 
                        "All players should be alive in the initial history")
        
        # Verify tasks are assigned to all players
        self.assertEqual(len(new_history[0].tasks_left_to_do), len(self.players), 
                        "Tasks should be assigned to all players")
    
    def test_end_game_history(self):
        # Test creating an end game history entry
        
        # Create end game history entry
        end_reason = EndGameReason.NO_IMPOSTORS_LEFT
        end_history = end_game_history(
            history=self.history,
            players=self.players,
            game_config=self.game_config,
            reason=end_reason
        )
        
        # Verify end history has the correct phase
        self.assertEqual(end_history.phase, GamePhase.GAME_END, 
                        "End history should have GAME_END phase")
        
        # Verify end history has the correct action type
        self.assertEqual(end_history.action_taken.type, ActionType.WAIT, 
                        "End history action should be WAIT")
        
        # Verify end history spectator message contains the reason
        self.assertIn(str(end_reason), end_history.action_taken.spectator, 
                     "End history spectator message should contain the end reason")
    
    def test_create_vote_history_entry(self):
        # Test creating a vote history entry
        # Using alive players from turn 19 (after Charlie voted for David)
        vote_phase_idx = 19
        
        # Get alive players
        alive_players = [p for p in self.players 
                         if p.name in self.history[vote_phase_idx].alive_player_names]
        
        # Create vote history entry
        ejected_player = "David"  # David was voted out in the actual game
        action_result = f"{ejected_player} was voted out."
        action_type = ActionType.KILL
        
        vote_history = create_vote_history_entry(
            history=self.history,
            alive_players=alive_players,
            ejected_player=ejected_player,
            action_result=action_result,
            action_type=action_type,
            game_config=self.game_config
        )
        
        # Verify vote history has the correct phase
        self.assertEqual(vote_history.phase, GamePhase.VOTE_RESULTS, 
                        "Vote history should have VOTE_RESULTS phase")
        
        # Verify vote history has the correct action type
        self.assertEqual(vote_history.action_taken.type, action_type, 
                        f"Vote history action should be {action_type}")
        
        # Verify vote history has the correct ejected player
        self.assertEqual(vote_history.action_taken.target_player_name, ejected_player, 
                        f"Vote history ejected player should be {ejected_player}")
        
        # Verify vote history has the correct action result
        self.assertEqual(vote_history.action_taken.spectator, action_result, 
                        f"Vote history action result should be '{action_result}'")
        
        # Verify ejected player is not in the alive players list
        self.assertNotIn(ejected_player, vote_history.alive_player_names, 
                        f"Ejected player {ejected_player} should not be in alive players list")
    
    def test_get_action_history_str(self):
        # Test getting action history string for a player
        # Using turn 10 (Alice waited in CAFETERIA)
        action_idx = 10
        
        # Find Alice player
        alice = next((p for p in self.players if p.name == "Alice"), None)
        self.assertIsNotNone(alice, "Alice player not found")
        
        # Get action history string
        history_str = get_action_history_str(
            history=self.history[:action_idx+1],
            players=self.players,
            player=alice,
            game_config=self.game_config
        )
        
        # Verify history string is not empty
        self.assertIsInstance(history_str, str, "History string should be a string")
        self.assertGreater(len(history_str), 0, "History string should not be empty")
        
        # Verify history string contains player info section
        self.assertIn("<player_info>", history_str, "History string should contain player info section")
        self.assertIn(f"<player_name>{alice.name}</player_name>", history_str, 
                     f"History string should contain player name {alice.name}")
        
        # Verify history string contains game history section
        self.assertIn("<game_history>", history_str, "History string should contain game history section")
        
        # Verify history string contains current state section
        self.assertIn("<current_state>", history_str, "History string should contain current state section")
        
        # Verify history string contains location information
        self.assertIn("<location>Cafeteria</location>", history_str, 
                     "History string should indicate Alice is in the Cafeteria")


if __name__ == "__main__":
    unittest.main()
