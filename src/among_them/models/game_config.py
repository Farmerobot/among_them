from dataclasses import dataclass

@dataclass
class GameConfig:
    """Holds the configuration parameters for a game instance."""
    num_tasks: int = 4
    num_players: int = 5
    num_impostors: int = 1
    map_size: int = 0 # Map size: 0 = small, 1 = medium, 2 = large
    num_task_phase_actions_per_player: int = 7 # resets after report
    num_discuss_phase_actions_per_player: int = 2 # resets after vote
    impostor_cooldown: int = 0 # resets after kill


