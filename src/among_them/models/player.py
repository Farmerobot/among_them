from typing import Optional
from among_them.models.player_role import PlayerRole
from among_them.config import LLMBackend, LLM_BACKEND, OLLAMA_LLM_MODEL_NAME, OPENROUTER_MODEL_NAME, HUGGINGFACE_MODEL_NAME


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
        
        # Set model name based on backend if not provided
        if llm_model_name is None:
            if LLM_BACKEND == LLMBackend.OLLAMA:
                self.llm_model_name = OLLAMA_LLM_MODEL_NAME
            elif LLM_BACKEND == LLMBackend.OPENROUTER:
                self.llm_model_name = OPENROUTER_MODEL_NAME
            elif LLM_BACKEND == LLMBackend.HUGGINGFACE:
                self.llm_model_name = HUGGINGFACE_MODEL_NAME
            else:  # MLX
                self.llm_model_name = "mlx-model"  # MLX uses loaded model, not name
        else:
            self.llm_model_name = llm_model_name
            
        # print(f"Using LLM backend: {LLM_BACKEND.value} with model: {self.llm_model_name}")
