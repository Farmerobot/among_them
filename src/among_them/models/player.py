from among_them.config import OLLAMA_LLM_MODEL_NAME
from among_them.models.player_role import PlayerRole


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
        llm_model_name: str = OLLAMA_LLM_MODEL_NAME
    ):
        self.name = name
        self.role = role
        self.manual_human_control = manual_human_control
        self.llm_model_name = llm_model_name

