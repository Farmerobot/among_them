from enum import Enum

class EndGameReason(Enum):
    NO_ACTIONS_LEFT = "No actions left"
    NO_IMPOSTORS_LEFT = "No impostors left"
    ALL_TASKS_DONE = "All tasks done"
    TOO_SMALL_NUMBER_OF_CREWMATES_LEFT = "Too small number of crewmates left"
        
    def __repr__(self):
        return self.value
    
    def __str__(self):
        return self.value
        
