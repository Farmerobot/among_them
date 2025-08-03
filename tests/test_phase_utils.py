import unittest

from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.utils.phase_utils import (
    get_last_voting_action_idx,
    get_phase_and_when_it_ends,
    handle_phase_change,
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
        
        # Verify the index points to a VOTE_RESULTS phase
        self.assertEqual(
            self.history[last_voting_idx-1].phase, 
            GamePhase.VOTE_RESULTS,
            "Last voting index should point to a history item after VOTE_RESULTS phase"
        )

        task_phase_idx = 8 # Bob does task
        last_voting_idx = get_last_voting_action_idx(self.history[:task_phase_idx+1])
        
        # Verify the index points to 0 if no voting has occurred
        self.assertEqual(last_voting_idx, 0, "Last voting index should be 0 if no voting has occurred")

    def test_get_phase_and_when_it_ends_task_phase(self):
        # Test phase determination during TASK phase
        # Using turn 6 (David moved to location Cafeteria)
        task_phase_idx = 6  # David moved to location Cafeteria
        
        # Get phase and actions until it ends
        phase, actions_left = get_phase_and_when_it_ends(
            history=self.history[:task_phase_idx+1],
            game_config=self.game_config,
        )
        
        # Verify phase is TASK and actions left is correct
        self.assertEqual(phase, GamePhase.TASKS, "Phase should be TASK")
        self.assertGreaterEqual(actions_left, 0, "Actions left should be non-negative")
    
    def test_get_phase_and_when_it_ends_after_report(self):
        # Test phase determination after a REPORT action
        # Using turn 13 (Bob reported dead body of Alice)
        report_phase_idx = 13  # Bob reported dead body of Alice
        
        # Check if this is actually a REPORT action
        self.assertTrue(self.history[report_phase_idx].action_taken.type == ActionType.REPORT)
        
        # Get phase and actions until it ends
        phase, actions_left = get_phase_and_when_it_ends(
            history=self.history[:report_phase_idx+1],
            game_config=self.game_config,
        )
        
        # Verify phase is DISCUSS after a report
        self.assertEqual(phase, GamePhase.DISCUSS, "Phase should be DISCUSS after a REPORT")
        self.assertGreater(actions_left, 0, "Actions left should be positive after a REPORT")

    def test_get_phase_and_when_it_ends_after_voting_results(self):
        # Test transition from VOTE_RESULTS to GAME_END since there are no impostors left
        # Using turn 20 (David was voted out)
        vote_results_idx = 20
        next_phase, actions_left = get_phase_and_when_it_ends(
            history=self.history[:vote_results_idx+1],
            game_config=self.game_config,
        )
        self.assertEqual(next_phase, GamePhase.GAME_END, "After VOTE_RESULTS phase should come GAME_END phase since there are no impostors left")
        self.assertEqual(actions_left, 0, "No actions left after GAME_END phase")
    
    def test_handle_phase_change(self):
        # Test phase transitions
        # Test transition from GAME_START to TASK
        # Using turn 0 (initial game state)
        last_game_start_idx = 0
        next_phase, actions_left = handle_phase_change(
            history=self.history[:last_game_start_idx+1],
            previous_phase=self.history[last_game_start_idx].phase,
            game_config=self.game_config
        )
        self.assertEqual(next_phase, GamePhase.TASKS, "After GAME_START phase should come TASK phase")
        self.assertEqual(actions_left, len(self.history[last_game_start_idx].alive_player_names)*self.game_config.num_task_phase_actions_per_player-1, "TASK phase should have all alive players left")

        # Test transition from DISCUSS to VOTE
        # Using turn 16 (Bob said "bry")
        last_discussion_idx = 16
        next_phase, actions_left = handle_phase_change(
            history=self.history[:last_discussion_idx+1],
            previous_phase=self.history[last_discussion_idx].phase,
            game_config=self.game_config
        )
        self.assertEqual(next_phase, GamePhase.VOTING, "After DISCUSS phase should come VOTE phase")
        self.assertEqual(actions_left, len(self.history[last_discussion_idx].alive_player_names)-1, "VOTE phase should have all alive players left")
        
        # Test transition from VOTE to VOTE_RESULTS
        # Using turn 19 (Charlie voted for David)
        last_vote_idx = 19
        next_phase, actions_left = handle_phase_change(
            history=self.history[:last_vote_idx+1],
            previous_phase=self.history[last_vote_idx].phase,
            game_config=self.game_config
        )
        self.assertEqual(next_phase, GamePhase.VOTE_RESULTS, "After VOTE phase should come VOTE_RESULTS phase")
        self.assertEqual(actions_left, 0, "VOTE_RESULTS phase should have no actions left")
    
    def test_count_votes(self):
        # Test vote counting
        # Using history up to turn 19 (after Charlie voted for David)
        vote_phase_end_idx = 19  # Charlie voted for David
        
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
        # Using history up to turn 19 (after Charlie voted for David)
        vote_phase_end_idx = 19  # Charlie voted for David
        
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
