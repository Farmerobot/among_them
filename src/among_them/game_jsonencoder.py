import json
from enum import Enum
from typing import Any, Dict

from among_them.models.action_type import ActionType
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player_role import PlayerRole


class GameJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle serialization of game objects."""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": obj.value}
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__.copy()
            obj_dict["__class__"] = obj.__class__.__name__
            obj_dict["__module__"] = obj.__class__.__module__
            return obj_dict
        return super().default(obj)


def game_object_hook(obj_dict: Dict[str, Any]) -> Any:
    """Custom JSON decoder hook to handle deserialization of game objects."""
    if "__enum__" in obj_dict:
        # Handle enums
        for enum_cls in [Location, PlayerRole, GamePhase, ActionType]:
            try:
                return enum_cls(obj_dict["__enum__"])
            except (ValueError, TypeError):
                continue
        return obj_dict
    
    # Handle enum fields directly in objects
    if "__class__" in obj_dict and "__module__" in obj_dict:
        # Reconstruct the class
        module_name = obj_dict.pop("__module__")
        class_name = obj_dict.pop("__class__")
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls.__new__(cls)
            for key, value in obj_dict.items():
                setattr(instance, key, value)
            return instance
        except (ImportError, AttributeError) as e:
            print(f"Error reconstructing object: {e}")
            return obj_dict
    
    return obj_dict