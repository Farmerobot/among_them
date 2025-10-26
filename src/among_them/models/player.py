from typing import Optional
from among_them.models.player_role import PlayerRole
from among_them.config import MODEL_NAME


class Player:
    """
    A player in the game. This class holds player-specific state and is responsible
    for generating LLM prompts for the player's turn.
    """

    def __init__(
        self, 
        name: str, 
        role: PlayerRole = PlayerRole.CREWMATE,
        manual_human_control: bool = False,
        llm_model_name: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.manual_human_control = manual_human_control
        
        # Set model name from config or use provided override
        if llm_model_name is None:
            self.llm_model_name = MODEL_NAME
        else:
            self.llm_model_name = llm_model_name
            
        # print(f"Using LLM backend: {LLM_BACKEND.value} with model: {self.llm_model_name}")
