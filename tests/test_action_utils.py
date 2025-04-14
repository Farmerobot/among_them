import unittest
from typing import List

from among_them.models.action import Action, ActionType
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.location import Location
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions

from tests.test_utils import load_test_game


class TestActionUtils(unittest.TestCase):
    def setUp(self):
        # Load the predefined game state
        self.history, self.players, self.game_config = load_test_game()
        
    def test_get_task_phase_actions_david_cooldown(self):
        """Test David's impostor cooldown behavior throughout the game."""
        # Find David player (the impostor)
        david = next((p for p in self.players if p.name == "David"), None)
        self.assertIsNotNone(david, "David player not found")
        self.assertEqual(david.role, PlayerRole.IMPOSTOR, "David should be an impostor")
        
        # Test 1: At game start (turn 0), David has cooldown of 1 and cannot kill
        start_idx = 0  # Game start
        actions = get_task_phase_actions(
            player=david,
            history=self.history[:start_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify no KILL actions are available due to cooldown
        self.assertFalse(any(a.type == ActionType.KILL for a in actions),
                        "David should not have KILL actions available at game start due to cooldown")
        
        # Test 2: After moving to Medbay (turn 2), David still cannot kill because no one is there
        medbay_idx = 2  # David moved to Medbay
        actions = get_task_phase_actions(
            player=david,
            history=self.history[:medbay_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify no KILL actions are available because no one is in Medbay
        self.assertFalse(any(a.type == ActionType.KILL for a in actions),
                        "David should not have KILL actions available in Medbay with no other players")
        
        # Test 3: After returning to Cafeteria (turn 6), David can kill because cooldown is 0 and others are there
        cafeteria_return_idx = 6  # David moved back to Cafeteria
        actions = get_task_phase_actions(
            player=david,
            history=self.history[:cafeteria_return_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify KILL actions are available
        kill_actions = [a for a in actions if a.type == ActionType.KILL]
        self.assertTrue(len(kill_actions) > 0, 
                       "David should have KILL actions available after returning to Cafeteria")
        
        # Verify all players in Cafeteria can be killed
        killable_players = [a.target_player_name for a in kill_actions]
        self.assertIn("Alice", killable_players, "Alice should be a killable target")
        self.assertIn("Bob", killable_players, "Bob should be a killable target")
        self.assertIn("Charlie", killable_players, "Charlie should be a killable target")
        
    def test_get_task_phase_actions_player_tasks(self):
        """Test task availability for different players in the Cafeteria."""
        # Test Bob's tasks in Cafeteria (he has 2 tasks there)
        bob = next((p for p in self.players if p.name == "Bob"), None)
        self.assertIsNotNone(bob, "Bob player not found")
        
        # Check Bob's initial tasks (turn 0)
        start_idx = 0
        bob_actions = get_task_phase_actions(
            player=bob,
            history=self.history[:start_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify Bob has 2 tasks in Cafeteria
        bob_task_actions = [a for a in bob_actions if a.type == ActionType.TASK]
        self.assertEqual(len(bob_task_actions), 2, "Bob should have 2 tasks in Cafeteria")
        
        # Check after Bob completes first task (turn 3)
        after_first_task_idx = 3
        bob_actions = get_task_phase_actions(
            player=bob,
            history=self.history[:after_first_task_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify Bob now has 1 task left in Cafeteria
        bob_task_actions = [a for a in bob_actions if a.type == ActionType.TASK]
        self.assertEqual(len(bob_task_actions), 1, "Bob should have 1 task left in Cafeteria after completing first task")
        
        # Test Alice's task in Cafeteria (she has 1 task there)
        alice = next((p for p in self.players if p.name == "Alice"), None)
        self.assertIsNotNone(alice, "Alice player not found")
        
        # Check Alice's initial tasks (turn 0)
        alice_actions = get_task_phase_actions(
            player=alice,
            history=self.history[:start_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify Alice has 1 task in Cafeteria
        alice_task_actions = [a for a in alice_actions if a.type == ActionType.TASK]
        self.assertEqual(len(alice_task_actions), 1, "Alice should have 1 task in Cafeteria")
        
        # Test Charlie's tasks in Cafeteria (he has no tasks there)
        charlie = next((p for p in self.players if p.name == "Charlie"), None)
        self.assertIsNotNone(charlie, "Charlie player not found")
        
        # Check Charlie's initial tasks (turn 0)
        charlie_actions = get_task_phase_actions(
            player=charlie,
            history=self.history[:start_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify Charlie has no tasks in Cafeteria
        charlie_task_actions = [a for a in charlie_actions if a.type == ActionType.TASK]
        self.assertEqual(len(charlie_task_actions), 0, "Charlie should have no tasks in Cafeteria")
    
    def test_get_task_phase_actions_after_kill(self):
        """Test Bob's actions after David kills Alice, including REPORT action."""
        # Find Bob player
        bob = next((p for p in self.players if p.name == "Bob"), None)
        self.assertIsNotNone(bob, "Bob player not found")
        
        # After David kills Alice (turn 12)
        after_kill_idx = 12
        
        # Get actions for Bob after the kill
        bob_actions = get_task_phase_actions(
            player=bob,
            history=self.history[:after_kill_idx+1],
            players=self.players,
            game_config=self.game_config
        )
        
        # Verify Bob has REPORT action available
        report_actions = [a for a in bob_actions if a.type == ActionType.REPORT]
        self.assertTrue(len(report_actions) > 0, "Bob should have REPORT action available after Alice is killed")
        
        # Verify the report action is for Alice
        self.assertEqual(report_actions[0].target_player_name, "Alice", 
                        "Bob's REPORT action should target Alice")
    
    def test_get_vote_actions(self):
        # Test vote actions during vote phase
        # Using turn 16 (Discussion phase ends)
        vote_phase_idx = 16  # Discussion phase ends
        
        # Find David player
        david = next((p for p in self.players if p.name == "David"), None)
        self.assertIsNotNone(david, "David player not found")
        
        # Get vote actions for David
        vote_actions = get_vote_actions(
            history=self.history[:vote_phase_idx+1],
            players=self.players,
            player=david
        )
        
        # Verify vote actions
        self.assertTrue(any(a.type == ActionType.VOTE for a in vote_actions), 
                       "Vote actions should include VOTE action type")
        
        # Check that "nobody" is an option
        self.assertTrue(any(a.type == ActionType.VOTE and a.target_player_name == "nobody" for a in vote_actions),
                       "Vote actions should include option to vote for nobody")
        
        # Check that all alive players except self are vote options
        alive_players = self.history[vote_phase_idx].alive_player_names
        for player_name in alive_players:
            if player_name != david.name:
                self.assertTrue(
                    any(a.type == ActionType.VOTE and a.target_player_name == player_name for a in vote_actions),
                    f"Vote actions should include option to vote for {player_name}"
                )
        
        # Check that dead players are not vote options
        all_player_names = [p.name for p in self.players]
        dead_players = [name for name in all_player_names if name not in alive_players]
        for player_name in dead_players:
            self.assertFalse(
                any(a.type == ActionType.VOTE and a.target_player_name == player_name for a in vote_actions),
                f"Vote actions should not include option to vote for dead player {player_name}"
            )


if __name__ == "__main__":
    unittest.main()
