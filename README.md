# Among Them

This project aims to test and compare the persuasive abilities of large language models (LLMs) in a game-like environment. The system simulates the game "Among Us" where AI agents, powered by various LLMs, interact, make decisions, and attempt to persuade each other. The goal is to understand how different LLMs approach persuasion, deception, and social interaction within a complex, dynamic environment.  The project analyzes both the success of persuasive attempts and the strategies employed by the agents.

![overview.png](images/overview.png)

## Current Features

- GUI for game state visualization and interaction using Streamlit.
- Game state tracking and saving for reproducibility and analysis.
- Integration with multiple LLM models via OpenRouter.
- Detailed logging and debugging tools.
- Cost estimation and analysis of LLM usage.
- Player state and history tracking.
- Map visualization with player locations and game logs.
- Discussion analysis with persuasion technique annotations.
- Selection of preferred LLM models for players.
- Implement functionality to export game data for analysis.
- A web-based panel for setting game parameters and customizing scenarios.
- Running tournaments along with game visualization
- Analysing all games in a tournament and displaying the results and persuasion techniques used by each model

## Project Architecture

The project is structured into several modules:

* **`agents`**: Contains implementations of different AI agents, each using a specific LLM.
* **`models`**: Defines data structures for tasks and game state.
* **`players`**: Defines the players.
* **`data`**: Contains saved game states ready to analyze. files are named in following format: `game_state_<phase>_<action_phase_count>_<discussion_phase_count>_<who_won>_<id>.json`.

![diagram.png](images/diagram.png)

## Table of Contents

- Installation
- Usage
- Configuration
- Running Tests

## Installation

This project requires Python 3.11 or higher.

### 1. Install Poetry

If you don't have Poetry installed, you can install it by following these steps:

- For Unix/macOS:

```bash
  curl -sSL https://install.python-poetry.org | python3 -
```

- For Windows:

  Visit the official [Poetry installation page](https://python-poetry.org/docs/#installation) for Windows-specific instructions.

Once installed, verify Poetry's installation by running:

```bash
  poetry --version
```

### 2. Clone the Repository and Install Dependencies

- Clone the repository

- Install dependencies using Poetry:

```bash
  poetry install
```

### 3. Activate the Virtual Environment

Once the dependencies are installed, activate the virtual environment:

```bash
  poetry shell
```

## Usage

To run the simulation with the GUI:

```bash
  poetry run main
```

## Configuration

The project requires API keys for LLMs such as OpenRouter. Set the following environment variables in your shell:

```bash
  export OPENROUTER_API_KEY=your-api-key
```

Alternatively, you can set these in a `.env` file at the project root.

## Running Tests

To run all tests, use the following command:

```bash
  poetry run pytest
```

## Future Enhancements (Roadmap)

We are planning to add the following features to enhance the project:

- **Voting Results Visualization**: Display voting results using bar charts for clarity.
- **Scenario Selection**: Allow users to choose game states.
- **Human Player Interface**: Allow human participation alongside AI agents.
- **LiteLLM Support**: Integrate LiteLLM.
- **Automated Testing**: Ensure robustness and reliability with automated tests.

These planned features aim to make the project more comprehensive and versatile for researchers and enthusiasts in AI and NLP fields.

This project offers a unique and interactive way to study the persuasive abilities of LLMs in a controlled, game-like environment, providing valuable insights for researchers in the field of artificial intelligence and natural language processing.
