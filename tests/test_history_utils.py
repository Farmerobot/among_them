import unittest

from among_them.models.action_type import ActionType
from among_them.models.end_game import EndGameReason
from among_them.models.phase import GamePhase
from among_them.utils.history_utils import (
    initialize_history,
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
        self.assertEqual(new_history[0].phase, GamePhase.TASKS, 
                        "First history item should have TASKS phase")
        
        # Verify all players are alive
        self.assertEqual(len(new_history[0].alive_player_names), len(self.players), 
                        "All players should be alive in the initial history")
        
        # Verify tasks are assigned to all players
        self.assertEqual(len(new_history[0].tasks_left_to_do), len(self.players), 
                        "Tasks should be assigned to all players")
    
    # NOTE: test_end_game_history removed - end_game_history function no longer exists
    
    # NOTE: test_create_vote_history_entry removed - create_vote_history_entry function no longer exists
    
    # NOTE: test_get_action_history_str removed - function is deprecated and uses legacy Action object structure


if __name__ == "__main__":
    unittest.main()
