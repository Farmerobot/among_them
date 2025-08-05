import json
import unittest
from among_them.models.action import Action
from among_them.models.action_type import ActionType
from among_them.models.location import Location
from among_them.game_jsonencoder import GameJSONEncoder, game_object_hook

class TestJSONSerialization(unittest.TestCase):
    def test_action_serialization_excludes_perspective_fields(self):
        """Test that Action serialization excludes perspective fields"""
        action = Action(
            type=ActionType.MOVE,
            player_name="Alice",
            target_location=Location.CAFETERIA
        )
        
        # Serialize the action
        serialized = json.dumps(action, cls=GameJSONEncoder)
        data = json.loads(serialized)
        
        # Check that perspective fields are not in the serialized data
        self.assertNotIn('agent_perspective', data)
        self.assertNotIn('observer_perspective', data)
        self.assertNotIn('global_perspective', data)
        self.assertNotIn('command_perspective', data)
        
        # Check that other fields are present
        self.assertIn('type', data)
        self.assertIn('player_name', data)
        self.assertIn('target_location', data)
    
    def test_action_serialization_excludes_empty_fields(self):
        """Test that Action serialization excludes empty fields"""
        action = Action(
            type=ActionType.MOVE,
            player_name="Alice",
            target_location=Location.CAFETERIA,
            target_player_name=None,  # This should be excluded
            target_message="Test message"  # This should be included
        )
        
        # Serialize the action
        serialized = json.dumps(action, cls=GameJSONEncoder)
        data = json.loads(serialized)
        
        # Check that empty/null fields are not in the serialized data
        self.assertNotIn('target_player_name', data)
        
        # Check that non-empty fields are present
        self.assertIn('type', data)
        self.assertIn('player_name', data)
        self.assertIn('target_location', data)
        self.assertIn('target_message', data)
        self.assertEqual(data['target_message'], 'Test message')
    
    def test_action_deserialization(self):
        """Test that Action can be properly deserialized"""
        action = Action(
            type=ActionType.MOVE,
            player_name="Alice",
            target_location=Location.CAFETERIA,
            target_message="Test message"
        )
        
        # Serialize and deserialize the action
        serialized = json.dumps(action, cls=GameJSONEncoder)
        deserialized = json.loads(serialized, object_hook=game_object_hook)
        
        # Check that the deserialized action has the correct type
        self.assertEqual(deserialized.__class__.__name__, 'Action')
        self.assertEqual(deserialized.type, ActionType.MOVE)
        self.assertEqual(deserialized.player_name, "Alice")
        self.assertEqual(deserialized.target_location, Location.CAFETERIA)
        self.assertEqual(deserialized.target_message, "Test message")

if __name__ == '__main__':
    unittest.main()
