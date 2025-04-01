# Among Them

A text-based social deduction game where AI agents play a game similar to "Among Us". In this game, players explore an environment while trying to complete objectives, with some players secretly assigned as impostors trying to sabotage the crew's efforts.

## Game Overview

"Among Them" is a social deduction game where:

- Players explore a spaceship-like environment with multiple interconnected rooms
- Crewmates work to complete tasks while impostors try to eliminate them
- When a dead body is found or reported, players discuss and vote on who to eject
- Game ends when either all impostors are ejected, crewmates complete all tasks, or impostors outnumber crewmates

## Key Game Elements

### Roles
- **Crewmates**: Complete tasks to win the game, report bodies, and vote out impostors
- **Impostors**: Eliminate crewmates, blend in with the crew, and avoid detection

### Game Phases
1. **Task Phase**: Players move between rooms and perform various actions
2. **Discussion Phase**: Following a dead body report, players discuss to identify impostors
3. **Voting Phase**: Players vote to eject someone from the game

### Player Actions
- **Move**: Travel between connected rooms
- **Task**: Complete an assigned task in a specific location
- **Kill**: (Impostor only) Eliminate a crewmate
- **Report**: Report a dead body to start a discussion
- **Speak**: Communicate during discussion phase
- **Vote**: Vote to eject a player (or skip voting)
- **Pretend**: (Impostor only) Pretend to complete a task
- **Wait**: Do nothing

## State Tracking and Game Progression

### History-Based State Management

The game uses a history-based approach to state management, with the `History` class being central to the entire system. Each game action creates a new immutable history entry that captures the complete state at that point.

#### History Structure

Each `History` object contains:
- **Player state**: Who acted, who will act next, player locations
- **Game phase**: Current phase and actions until phase ends
- **Action details**: Type of action, target, result text
- **Spectator information**: Which players witnessed the action
- **LLM data**: Chain of thought, response text, token usage
- **Task state**: Tasks remaining for each player
- **Location**: Where the action occurred
- **Impostor cooldown**: Impostor cooldown

#### State Transitions

State transitions are primarily handled in `phase_utils.py`:
1. `get_phase_and_when_it_ends()`: Determines the next phase based on current history
2. `handle_phase_change()`: Manages automatic phase transitions
3. `count_votes()`: Tallies votes during voting phase
4. `determine_ejection_result()`: Processes voting outcomes

### Game Step Execution (`perform_step()`)

The `perform_step()` method in `GameEngine` is the core function that advances the game. For each step:

1. Get alive players and current phase
2. Check for game end conditions
3. Select the next player to act
4. Determine available actions based on:
   - Player role
   - Current location
   - Current phase
   - Cooldown status
5. Generate history string for the LLM prompt
6. Get action choice from player (via LLM or human input)
7. Process action consequences
8. Create new history entry with updated state
9. Save game state
10. Return whether the game has ended

This approach ensures immutability of past events while maintaining a complete record of game progression.

### Player and Location Tracking

Player tracking is managed through several utility functions in `player_utils.py`:
- `get_alive_players()`: Filters players who haven't been killed
- `get_last_player_action()`: Finds most recent action for a player
- `get_dead_players()`: Maps dead players to their death locations
- `get_players_in_room()`: Identifies all players in a specific location

Location tracking is handled implicitly through the `location` field in each history entry, with movement actions updating this field.

## Technical Architecture

### Core Components

#### Game Engine (`game_engine.py`)
The central component that manages game logic, player actions, state transitions, and win conditions. Key responsibilities:
- `__init__()`: Initializes the game with players and assigns roles
- `perform_step()`: Processes game steps where players perform actions
- `save_state()` and `load_state()`: Manages game persistence
- `check_players_set_impostors()`: Handles role assignment

The engine deliberately maintains no mutable state outside of the history list, ensuring all game logic can be derived from history entries.

#### AI Agents (`agents/unified_agent.py`)
AI-controlled player behavior using Large Language Models (LLMs):
- `act()`: Core function that generates prompts and processes responses
- `normalize_and_check_action_valid()`: Validates LLM outputs against available actions

The agent is designed with error recovery, attempting to self-correct when invalid responses are given.

#### Player Types
The game supports both AI and human players:
- **AIPlayer**: LLM-powered agents that make decisions based on game state
- **HumanPlayer**: Console interface for human players to interact with the game

#### Action Generation (`action_utils.py`)
Functions that determine available actions based on game state:
- `get_task_phase_actions()`: Generates available actions during task phase based on player role, location, and cooldown
- `get_vote_actions()`: Creates voting options during voting phase

#### Models
- `Action`: Represents a player action with text representations for both the player and spectators
- `Player`: Base class with AIPlayer and HumanPlayer implementations
- `Location`: Defines the game map with rooms and connections via the `DOORS` dictionary
- `Phase`: Enumerates game phases (Task, Discuss, Vote)
- `History`: The core data structure that tracks game state
- `Task`: Implements ShortTask and LongTask with location-specific completion logic

### Design Choices

#### 1. Immutable History-Based State
The game uses an append-only history list rather than mutable state objects. This design:
- Simplifies debugging by preserving complete game history
- Allows game replay and analysis
- Makes state transitions explicit and traceable
- Supports save/load functionality without additional logic

#### 2. Function-Based Action Generation
Instead of hardcoding available actions, the system uses functions to generate actions dynamically:
- Adapts to changing game conditions
- Enforces game rules consistently
- Makes adding new action types easier

#### 3. Separation of Agent and Player Logic
The system clearly separates:
- Player logic (what actions are available)
- Agent logic (how decisions are made)
- Game progression (how actions affect state)

This allows for different agent implementations without changing game rules.

#### 4. Location-Based Task System
Tasks are tied to specific locations:
- Forces players to move around the map
- Creates opportunities for player interaction
- Mirrors the gameplay of the original Among Us inspiration

#### 5. Room-Based Observation Model
Actions are only visible to players in the same room:
- Creates information asymmetry critical for deduction
- Encourages strategic movement
- Makes alibis and witness testimony meaningful

#### 6. JSON Serialization
Game state is serialized using jsonpickle:
- Allows game state to be saved and restored
- Supports archiving completed games
- Makes debugging and analysis easier

## LLM Integration

The game leverages Large Language Models to control AI players:
- Each player is powered by an LLM (default is "deepseek-r1:14b")
- The game provides the LLM with game context, available actions, and history
- The LLM decides what action to take and provides reasoning
- Special prompting techniques are used to enforce valid outputs

### Prompt Structure
The LLM prompt is carefully structured to provide:
1. Player information (role, tasks, allies/enemies)
2. Action history (what the player has seen/done)
3. Current situation (location, phase, other players present)
4. Available actions

### Chain of Thought Reasoning
The LLM agent uses chain-of-thought reasoning enclosed in `<think>` tags to make decisions. This allows the agent to:
- Analyze the game state
- Consider possible strategies
- Evaluate the consequences of actions
- Make decisions based on the current situation

## Running the Game

To run the game:
1. Install dependencies with `poetry install`
2. Start the game with `poetry run main`

The game requires Ollama to be installed and running with the specified model.

## Environment Variables

The project uses a `.env` file for configuration (via `python-dotenv`):

```
# API key for OpenRouter (optional)
OPENROUTER_API_KEY=your_api_key_here

# Whether to run the game locally (default: True)
RUN_LOCALLY=True
```

## Configuration Constants

The `consts.py` file defines important game parameters:
- `NUM_SHORT_TASKS` (4): Number of short tasks assigned to each player
- `NUM_LONG_TASKS` (1): Number of long tasks assigned to each player
- `NUM_CHATS` (5): Number of discussion messages per player
- `NUM_ACTIONS_WITHOUT_REPORT` (5): Maximum actions without reporting. After this the game will end.
- `IMPOSTOR_COOLDOWN` (1): Turns between impostor kill actions
- `STATE_FILE`: Location for saving game state
- `TOKEN_COSTS`: Cost mapping for different LLM models

## Game Progression and Win Conditions

The game progresses through turns where:
1. A random player is selected to act
2. The player is provided with available actions based on their role and location
3. The player's agent (LLM) decides what action to take
4. The action is processed and the game state is updated
5. This continues until a win condition is met

The game can end in several ways (`end_game.py`):
- **Crewmate Victory**: All impostors eliminated or all tasks completed
- **Impostor Victory**: Equal number of impostors and crewmates remain
- **Game End**: No actions left to perform

## Key Features

- **Spatial Navigation**: Players move between interconnected rooms
- **Role-Based Actions**: Different actions available based on player role
- **Discussion & Voting**: Players discuss and vote after bodies are reported
- **State Persistence**: Game state is saved after each action
- **LLM-Powered Agents**: AI players make decisions using language models
- **Local LLM Support**: Uses Ollama to run models locally without requiring cloud API access
- **Persuasion Analysis**: The codebase includes tools for analyzing persuasion techniques used in discussions

This project demonstrates how LLMs can be used to create emergent gameplay in a social deduction setting, with AI agents exhibiting complex reasoning and social interaction.