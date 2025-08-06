import unittest

from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.utils.phase_utils import (
    get_last_voting_action_idx,
    count_votes,
    determine_ejection_result
)

from tests.test_utils import load_test_game


class TestPhaseUtils(unittest.TestCase):
    def setUp(self):
        # Load the predefined game state
        self.history, self.players, self.game_config = load_test_game()
        
    def test_get_last_voting_action_idx(self):
        # Test finding the last voting action index
        # Using the full game history which should contain voting phases
        
        # Get the last voting action index
        last_voting_idx = get_last_voting_action_idx(self.history)
        
        # Verify the index points to a VOTING phase
        self.assertEqual(
            self.history[last_voting_idx-1].phase, 
            GamePhase.VOTING,
            "Last voting index should point to a history item after VOTING phase"
        )

        task_phase_idx = 8 # Bob does task
        last_voting_idx = get_last_voting_action_idx(self.history[:task_phase_idx+1])
        
        # Verify the index points to 0 if no voting has occurred
        self.assertEqual(last_voting_idx, 0, "Last voting index should be 0 if no voting has occurred")

    # NOTE: test_get_phase_and_when_it_ends_task_phase removed - get_phase_and_when_it_ends function no longer exists
    
    # NOTE: test_get_phase_and_when_it_ends_after_report removed - get_phase_and_when_it_ends function no longer exists
    
    # NOTE: test_get_phase_and_when_it_ends_after_voting_results removed - get_phase_and_when_it_ends function no longer exists
    
    # NOTE: test_handle_phase_change removed - handle_phase_change function no longer exists
    
    def test_count_votes(self):
        # Test vote counting
        # Using history up to turn 21 (after Charlie voted for David)
        vote_phase_end_idx = 21  # Charlie voted for David
        
        # Get vote counts and votes
        vote_counts, votes = count_votes(self.history[:vote_phase_end_idx+1])
        
        # Verify vote counts and votes dictionaries are not empty if voting occurred
        self.assertTrue(any(h.action_taken.type == ActionType.VOTE for h in self.history[:vote_phase_end_idx+1]), "Voting occurred")
        self.assertGreater(len(vote_counts), 0, "Vote counts should not be empty after voting")
        self.assertGreater(len(votes), 0, "Votes dictionary should not be empty after voting")
        
        # Check that vote counts sum matches the number of votes cast
        total_votes = sum(vote_counts.values())
        self.assertEqual(total_votes, len(votes), "Total vote count should match number of votes cast")
        
        # Verify David received 2 votes (from Bob and Charlie)
        self.assertEqual(vote_counts.get("David", 0), 2, "David should have received 2 votes")
        # Verify nobody received 1 vote (from David)
        self.assertEqual(vote_counts.get("nobody", 0), 1, "nobody should have received 1 vote")
    
    def test_determine_ejection_result(self):
        # Test ejection result determination
        # Using history up to turn 21 (after Charlie voted for David)
        vote_phase_end_idx = 21  # Charlie voted for David
        
        # Get vote counts
        vote_counts, _ = count_votes(self.history[:vote_phase_end_idx+1])
        
        self.assertTrue(vote_counts, "Vote counts should not be empty after voting")
        
        # Determine ejection result
        ejected_player, action_type = determine_ejection_result(vote_counts)
        
        # Verify ejected player and action type
        self.assertIsNotNone(ejected_player, "Ejected player should not be None")
        self.assertEqual(ejected_player, "David", "David should be ejected")
        self.assertEqual(action_type, ActionType.KILL, "Action type should be KILL when someone is ejected")


if __name__ == "__main__":
    unittest.main()
