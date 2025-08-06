import unittest

from among_them.models.end_game import EndGameReason
from among_them.utils.end_utils import get_end_game_reason

from tests.test_utils import load_test_game


class TestEndUtils(unittest.TestCase):
    def setUp(self):
        # Load the predefined game state
        self.history, self.players, self.game_config = load_test_game()
        
    def test_get_end_game_reason_none(self):
        # Test during mid-game when there should be no end game reason
        # Using turn 9 (Alice waited in CAFETERIA)
        mid_game_idx = 9
        
        # Get end game reason
        reason = get_end_game_reason(
            history=self.history[:mid_game_idx+1],
            players=self.players
        )
        
        # Verify no end game reason during mid-game
        self.assertIsNone(reason, "There should be no end game reason during mid-game")
        
    def test_get_end_game_reason_at_end(self):
        # Test at the end of the game when there should be an end game reason
        # Using turn 21 (end of game)
        end_game_idx = 21  # The game ended (No impostors left)
        
        # Get end game reason
        reason = get_end_game_reason(
            history=self.history[:end_game_idx+1],
            players=self.players
        )
        
        # Verify there is an end game reason at the end
        self.assertIsNotNone(reason, "There should be an end game reason at the end of the game")
        self.assertIn(reason, [
            EndGameReason.NO_IMPOSTORS_LEFT,
            EndGameReason.TOO_SMALL_NUMBER_OF_CREWMATES_LEFT,
            EndGameReason.ALL_TASKS_DONE,
            EndGameReason.NO_ACTIONS_LEFT
        ], f"End game reason should be one of the valid reasons, got {reason}")
        
    def test_get_end_game_reason_after_kill(self):
        # Test after a kill that might lead to game end
        # Using turn 22 (after David was voted out)
        after_kill_idx = 22  # David was voted out
        
        # Get end game reason
        reason = get_end_game_reason(
            history=self.history[:after_kill_idx+1],
            players=self.players
        )
        
        # Check if the vote led to a game end condition
        # In this case, it should be NO_IMPOSTORS_LEFT since David was the impostor
        self.assertEqual(reason, EndGameReason.NO_IMPOSTORS_LEFT, 
                        "After voting out the impostor, end game reason should be NO_IMPOSTORS_LEFT")


if __name__ == "__main__":
    unittest.main()
