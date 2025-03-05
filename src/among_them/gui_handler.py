import concurrent.futures
import datetime as dt
import json
import os
import random
import shutil
import uuid
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from annotated_text.util import p
import pandas as pd
import plotly.graph_objects as go
from plotly.io import show
import streamlit as st
from annotated_text import annotated_text
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from streamlit.delta_generator import DeltaGenerator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from among_them.annotation import annotate_dialogue
from among_them.config import OPENROUTER_API_KEY
from among_them.game import dummy
from among_them.game.consts import (
    IMPOSTOR_COOLDOWN,
    NUM_CHATS,
    NUM_LONG_TASKS,
    NUM_SHORT_TASKS,
    STATE_FILE,
    TOKEN_COSTS,
)
from among_them.game.game_engine import GameEngine
from among_them.game.game_state import GameState
from among_them.llm_prompts import (
    ADVENTURE_ACTION_SYSTEM_PROMPT,
    ADVENTURE_ACTION_USER_PROMPT,
    ADVENTURE_PLAN_SYSTEM_PROMPT,
    ADVENTURE_PLAN_USER_PROMPT,
    ANNOTATION_SYSTEM_PROMPT,
    DISCUSSION_RESPONSE_SYSTEM_PROMPT,
    DISCUSSION_RESPONSE_USER_PROMPT,
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_USER_PROMPT,
    PERSUASION_TECHNIQUES,
    VOTING_SYSTEM_PROMPT,
    VOTING_USER_PROMPT,
)
from among_them.game.models.engine import ROOM_COORDINATES, GamePhase
from among_them.game.models.history import PlayerState, RoundData
from among_them.game.players.ai import AIPlayer
from among_them.game.players.base_player import Player, PlayerRole
import csv
import numpy as np

class Watchdog(FileSystemEventHandler):
    def __init__(self, hook: Callable):
        self.hook = hook

    def on_modified(self, event: Any):
        self.hook()


def update_dummy_module():
    # Rewrite the dummy.py module. Because this script imports dummy,
    # modifying dummy.py will cause Streamlit to rerun this script.
    # This is to update gui automatically when tournament is run.
    # https://discuss.streamlit.io/t/how-to-monitor-the-filesystem-and-have-streamlit-updated-when-some-files-are-modified/822/9
    dummy_path = dummy.__file__
    with open(dummy_path, "w") as fp:
        fp.write(f'timestamp = "{dt.datetime.now()}"')


@st.cache_resource
def install_monitor():
    watchdog = Watchdog(update_dummy_module)
    observer = Observer()
    observer.schedule(watchdog, "data", recursive=False)
    observer.start()


class GUIHandler(BaseModel):
    def display_gui(self, game_engine: GameEngine):
        if OPENROUTER_API_KEY != "None":
            install_monitor()
        game_overview, tournaments, techniques = st.tabs([
            "Game Overview",
            "Tournaments",
            "Persuasion Techniques",
        ])
        with game_overview:
            if game_engine.state.game_stage == GamePhase.MAIN_MENU:
                self.game_settings()
            else:
                self.sidebar(game_engine=game_engine)
                self.game_overview(game_engine)
        with tournaments:
            self.tournaments(debug=OPENROUTER_API_KEY != "None")
        with techniques:
            self._display_persuasion_techniques()

    def sidebar(self, game_engine: GameEngine):
        with st.sidebar:
            total_cost = round(game_engine.state.get_total_cost()["total_cost"], 3)
            st.write(f"{game_engine.state.game_stage.value}. Total cost: {total_cost}$")
            for i, player in enumerate(game_engine.state.players):
                self._display_short_player_info(
                    player, i == game_engine.state.player_to_act_next, st
                )

    def game_overview(self, game_engine: GameEngine):
        st.title("Among Them")
        should_perform_step = False

        self._handle_tournament_file_selection(game_engine)

        if game_engine.state.DEBUG:
            should_perform_step = st.checkbox("Perform Steps automatically")
            # Create columns for buttons
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            # Buttons in columns
            with col1:
                if st.button("Make Step"):
                    should_perform_step = True
            with col2:
                if st.button("Clear Game State"):
                    self.clear_game_state()
            with col3:
                if st.button("Save State to Tournaments"):
                    if game_engine.check_game_over():
                        self.save_state_to_tournaments(game_engine)
                    else:
                        st.warning(
                            "Game is not over yet! Please finish the game first."
                        )
            with col4:
                if st.button("Force Set and Step Action"):
                    game_engine.state.set_stage(GamePhase.ACTION_PHASE)
                    game_engine.perform_step()
            with col5:
                if st.button("Force Set and Step Discussion"):
                    game_engine.state.set_stage(GamePhase.DISCUSS)
                    game_engine.perform_step()
            with col6:
                if st.button("Force step Voting"):
                    game_engine.go_to_voting()

        col1, col2 = st.columns([2, 1])
        with col1:
            self._display_map(game_engine.state)
        with col2:
            with st.container(height=300):
                st.text("\n".join(game_engine.state.playthrough))
        self._display_player_selection(game_engine.state.players)
        discussion = self._display_discussion_chat(game_engine.state.players)
        # Analyze Chat Button
        if OPENROUTER_API_KEY != "None":
            if st.button("Analyze Chat"):
                # results = chat_analyzer.analyze()
                results = annotate_dialogue(discussion)
                st.session_state.results = results
        if "results" in st.session_state:
            self._display_annotated_text(
                json.loads(st.session_state.results), game_engine.state.players
            )

        if st.session_state.selected_player < len(game_engine.state.players):
            player: Player = game_engine.state.players[st.session_state.selected_player]
            st.subheader(f"Player: {player.name} Chat History:")
            self._display_chat_history(player.history.rounds + [player.state])

        # Cost Visualization
        cost_data = self.get_cost_data(game_engine)
        if game_engine.state.round_number >= 1:
            estimated_cost_data = self.estimate_future_cost(cost_data, 5)
            combined_cost_data = self.combine_data(cost_data, estimated_cost_data)
            # self.plot_cost(combined_cost_data, 5)

        st.text("Cost Breakdown:")
        st.json(game_engine.state.get_total_cost(), expanded=False)
        st.text("Raw Game State:")
        st.json(game_engine.state.to_dict(), expanded=False)

        if should_perform_step:
            try:
                game_engine.perform_step()
            except Exception as e:
                if "LLM did" in str(e):
                    pass
                else:
                    raise e

    def _handle_tournament_file_selection(self, game_engine: Optional[GameEngine]):
        # Get list of tournament files
        tournament_dir = "data/tournament"

        # List all JSON files in the directory
        tournament_files = [
            f for f in os.listdir(tournament_dir) if f.endswith(".json")
        ]
        if tournament_files:
            # Check for OpenRouter API key
            tournament_files = (
                ["None"]
                + (["DEBUG"] if OPENROUTER_API_KEY != "None" else [])
                + tournament_files
            )
            if "previous_selected_file" not in st.session_state:
                st.session_state.previous_selected_file = None

            selected_file = st.selectbox("Select tournament file", tournament_files, index=0 if OPENROUTER_API_KEY!="None" else 1)
            game_state_path = "data/game_state.json"

            if selected_file == "None":
                if os.path.exists(game_state_path):
                    os.remove(game_state_path)
                    st.success("Game state cleared")
                    st.rerun()
                st.session_state.previous_selected_file = None
            elif selected_file == "DEBUG":
                if game_engine is not None:
                    game_engine.state.DEBUG = True
                    st.success("Debug mode enabled")
                st.session_state.previous_selected_file = "DEBUG"
            elif (
                selected_file
                and selected_file != st.session_state.previous_selected_file
            ):
                # Copy selected file to game_state.json
                shutil.copy(
                    os.path.join(tournament_dir, selected_file), game_state_path
                )
                st.success(f"Loaded game state from {selected_file}")
                st.session_state.previous_selected_file = selected_file

                # Try to load annotations if they exist
                annotation_file = os.path.join("data/annotations", selected_file)
                if os.path.exists(annotation_file):
                    with open(annotation_file, "r") as f:
                        st.session_state.results = json.dumps(json.load(f))
                        st.success("Loaded existing annotations")
                st.rerun()

    def tournaments(self, debug: bool = False):
        st.title("Tournaments")
        if debug:
            if st.button("Analyze Tournaments"):
                self.analyze_tournaments()
            if st.button("Analyze Tournaments v2"):
                self.analyze_tournaments_v2()
            if st.button("Analyze Persuasion Wins"):
                self.analyze_persuasion_wins()

        # read data/analysis.json
        df = None
        if os.path.exists("data/analysis.json"):
            with open("data/analysis.json", "r") as f:
                data = json.load(f)
                model_techniques = data["model_techniques"]
                model_player_counts = data["model_player_counts"]
                model_input_tokens = data["model_input_tokens"]
                model_output_tokens = data["model_output_tokens"]
                df = self._display_tournament_persuasion_analysis(
                    model_techniques,
                    model_player_counts,
                    model_input_tokens,
                    model_output_tokens,
                    "Persuasion Techniques",
                )
                
            # Display random examples for each technique
            df.sort_values("Total Techniques", ascending=False, inplace=True)
            if df is not None and os.path.exists("data/combined_annotations.csv"):
                st.subheader("Example Usage of Each Technique")
                
                # Read the examples
                annotations_df = pd.read_csv("data/combined_annotations.csv")
                technique_examples = defaultdict(list)
                
                # Process annotations
                for _, row in annotations_df.iterrows():
                    if pd.notna(row['annotation']):
                        for technique in row['annotation'].split(';'):
                            technique = technique.strip().lower()
                            technique_examples[technique].append({
                                'text': row['text'],
                                'speaker': row['speaker'],
                                'model': row['model'],
                                'role': row['role'],
                                'file': row['source_file'],
                                'annotation': [x.strip().lower() for x in row['annotation'].split(';') if x.strip().lower() in df.index]
                            })
                
                # Create columns for techniques
                cols = st.columns(2)
                for i, technique in enumerate(df.index):
                    if technique.lower() in technique_examples and technique_examples[technique.lower()]:
                        with cols[i % 2]:
                            with st.expander(f"### {i+1}. {technique} ({int(df.loc[technique]['Total Techniques'])} times)", expanded=False):
                                example = random.choice(technique_examples[technique.lower()])
                                # print({
                                #     "text": example['text'],
                                #     "annotation": example['annotation']
                                # })
                                # Show model and role with colored badges
                                role_color = "red" if example['role'] == "impostor" else "green"
                                st.markdown(
                                    f'<span style="background-color: #404040; padding: 3px 8px; border-radius: 4px; margin-right: 5px">{example["model"]}</span>'
                                    f'<span style="background-color: {role_color}; padding: 3px 8px; border-radius: 4px">{example["role"]}</span>',
                                    unsafe_allow_html=True
                                )
                                
                                # Show the example text in a quote block
                                st.markdown(f"> {example['text']}")
                                
                                # Show who said it
                                st.caption(f"*— {example['speaker']}* @ {example['file']}")
                                
                                # Add a button to show another example
                                st.button(f"Show another example", key=f"refresh_{technique}")
                                
        
        # read data/analysis_impostor.json
        if os.path.exists("data/analysis_impostor.json"):
            with open("data/analysis_impostor.json", "r") as f:
                data = json.load(f)
                model_techniques = data["model_techniques"]
                model_player_counts = data["model_player_counts"]
                model_input_tokens = data["model_input_tokens"]
                model_output_tokens = data["model_output_tokens"]
                self._display_tournament_persuasion_analysis(
                    model_techniques,
                    model_player_counts,
                    model_input_tokens,
                    model_output_tokens,
                    "Impostor Techniques"
                )

        # read data/analysis_crewmate.json
        if os.path.exists("data/analysis_crewmate.json"):
            with open("data/analysis_crewmate.json", "r") as f:
                data = json.load(f)
                model_techniques = data["model_techniques"]
                model_player_counts = data["model_player_counts"]
                model_input_tokens = data["model_input_tokens"]
                model_output_tokens = data["model_output_tokens"]
                self._display_tournament_persuasion_analysis(
                    model_techniques,
                    model_player_counts,
                    model_input_tokens,
                    model_output_tokens,
                    "Crewmate Techniques"
                )
                

    def clear_game_state(self):
        """Deletes the game_state.json file to clear the game state."""
        try:
            os.remove("data/game_state.json")
            st.success("Game state cleared successfully!")
            st.rerun()
        except FileNotFoundError:
            st.warning("No game state file found.")

    def analyze_tournaments(self):
        # Directory containing tournament JSON files
        tournament_dir = "data/tournament"

        # List all JSON files in the directory
        tournament_files = [
            f for f in os.listdir(tournament_dir) if f.endswith(".json")
        ]
        # Filter files that have corresponding annotations
        tournament_files = [
            f for f in tournament_files
            if not os.path.exists(os.path.join("data/annotations", f))
        ]

        # Dictionary to accumulate techniques for each model
        model_techniques = defaultdict(lambda: defaultdict(int))
        model_player_counts = defaultdict(int)

        # Dictionaries to store token usage per model
        model_input_tokens = defaultdict(lambda: defaultdict(int))
        model_output_tokens = defaultdict(lambda: defaultdict(int))

        # Iterate over each file and load the game state
        progress_placeholder = st.text("Starting to analyze tournament files...")
        with st.status("Analyzing tournament files...") as status:
            total_files = len(tournament_files)
            progress_text = st.empty()
            files_analyzed = 0

            def analyze_file(file_name: str):
                file_path = os.path.join(tournament_dir, file_name)
                game_engine = GameEngine()
                if game_engine.load_state(file_path):
                    game_state = game_engine.state
                    players = game_state.players

                    discussion_chat = ""
                    # Get the longest discussion chat from all players - ensure the
                    # player was alive until the end
                    for player in players:
                        if player.state.life == PlayerState.ALIVE:
                            discussion_chat = "\n".join(player.get_chat_messages())
                            if not discussion_chat.strip():
                                discussion_chat = "\n".join([
                                    obs[18:]
                                    for obs in player.state.observations
                                    if obs.startswith("chat")
                                ])
                            break

                    if not discussion_chat:
                        print(f"No discussion chat found for file: {file_name}")
                        st.write(f"No discussion chat found for file: {file_name}")
                        return

                    annotation_json = None
                    annotated = annotate_dialogue(discussion_chat).strip()
                    
                    if annotated.startswith("```json"):
                        annotated = annotated.split("```json", 1)[1].split("```", 1)[0].strip()

                    annotated = annotated.replace(',\n]', '\n]')
                    
                    if not annotated:
                        raise ValueError("no annotation")
                        return
                    try:
                        annotation_json = json.loads(annotated)
                    except Exception as e:
                        print(annotated)
                        raise e
                    if not annotation_json:
                        print(f"No annotation found for file: {file_name}")
                        st.write(f"No annotation found for file: {file_name}")
                        raise ValueError("no annotation")
                        return
                    else:
                        # Create annotations directory if it doesn't exist
                        os.makedirs("data/annotations", exist_ok=True)

                        # Save annotation to file
                        annotation_file = os.path.join(
                            "data/annotations", file_name
                        )
                        with open(annotation_file, "w", encoding="utf-8") as f:
                            json.dump(annotation_json, f, indent=2)

                    previous_player = None
                    player_techniques = defaultdict(list)

                    for item in annotation_json:
                        replaced_text = item["text"]
                        current_player = (
                            replaced_text.split("]:")[0].strip("[]")
                            if "]: " in replaced_text
                            else previous_player
                        )

                        if item["annotation"]:
                            player_techniques[current_player].extend(item["annotation"])

                        previous_player = current_player

                    for player in players:
                        model_name = player.llm_model_name
                        model_player_counts[model_name] += 1
                        model_input_tokens[model_name][file_name] = (
                            player.state.token_usage.input_tokens
                        )
                        model_output_tokens[model_name][file_name] = (
                            player.state.token_usage.output_tokens
                        )
                        for technique in player_techniques[player.name]:
                            model_techniques[model_name][technique] += 1

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(analyze_file, file_name): file_name
                    for file_name in tournament_files
                }
                for future in concurrent.futures.as_completed(futures):
                    file_name = futures[future]
                    files_analyzed += 1
                    progress_text.write(
                        f"Analyzing files... ({files_analyzed}/{total_files})"
                    )
                    try:
                        future.result()  # This will raise any exceptions that occurred
                        st.write(f"✅ Successfully analyzed {file_name}")
                    except Exception as e:
                        st.error(f"❌ Error analyzing {file_name}: {str(e)}")
                        executor._threads.clear()
                        raise e
                
            status.update(label="Analysis complete!", state="complete")
            progress_text.empty()

        # Clear the progress message
        progress_placeholder.empty()

        # save dicts to a file
        with open("data/analysis.json", "w") as f:
            json.dump(
                {
                    "model_techniques": model_techniques,
                    "model_player_counts": model_player_counts,
                    "model_input_tokens": model_input_tokens,
                    "model_output_tokens": model_output_tokens,
                },
                f,
            )
            
    def analyze_tournaments_v2(self):
        # Load combined annotations
        if os.path.exists("data/combined_annotations.csv"):
            annotations_df = pd.read_csv("data/combined_annotations.csv")
            impostor_model_techniques = defaultdict(lambda: defaultdict(int))
            crewmate_model_techniques = defaultdict(lambda: defaultdict(int))
            impostor_model_player_counts = defaultdict(int)
            crewmate_model_player_counts = defaultdict(int)

            # Dictionaries to store token usage per model
            impostor_model_input_tokens = defaultdict(lambda: defaultdict(int))
            impostor_model_output_tokens = defaultdict(lambda: defaultdict(int))
            crewmate_model_input_tokens = defaultdict(lambda: defaultdict(int))
            crewmate_model_output_tokens = defaultdict(lambda: defaultdict(int))

            # Dictionary to store technique examples
            technique_examples = defaultdict(list)

            for _, row in annotations_df.iterrows():
                if pd.notna(row['annotation']):
                    for technique in row['annotation'].split(';'):
                        technique = technique.strip().lower()
                        technique_examples[technique].append({
                            'text': row['text'],
                            'speaker': row['speaker'],
                            'model': row['model'],
                            'role': row['role']
                        })

            for _, row in annotations_df.iterrows():
                if pd.notna(row['annotation']):
                    for technique in row['annotation'].split(';'):
                        technique = technique.strip().lower()
                        if row['role'] == 'impostor':
                            impostor_model_techniques[row['model']][technique] += 1
                            impostor_model_player_counts[row['model']] += 1
                        else:
                            crewmate_model_techniques[row['model']][technique] += 1
                            crewmate_model_player_counts[row['model']] += 1

            # save dicts to a file
            with open("data/analysis.json", "w") as f:
                json.dump(
                    {
                        "model_techniques": {k: dict(Counter(impostor_model_techniques.get(k, {})) + Counter(crewmate_model_techniques.get(k, {}))) for k in set(impostor_model_techniques) | set(crewmate_model_techniques)},
                        "model_player_counts": dict(Counter(impostor_model_player_counts) + Counter(crewmate_model_player_counts)),
                        "model_input_tokens": dict(Counter(impostor_model_input_tokens) + Counter(crewmate_model_input_tokens)),
                        "model_output_tokens": dict(Counter(impostor_model_output_tokens) + Counter(crewmate_model_output_tokens)),
                    },
                    f
                )
            
        
            with open("data/analysis_impostor.json", "w") as f:
                json.dump(
                    {
                        "model_techniques": impostor_model_techniques,
                        "model_player_counts": impostor_model_player_counts,
                        "model_input_tokens": impostor_model_input_tokens,
                        "model_output_tokens": impostor_model_output_tokens,
                    },
                    f,
                )
        
            with open("data/analysis_crewmate.json", "w") as f:
                json.dump(
                    {
                        "model_techniques": crewmate_model_techniques,
                        "model_player_counts": crewmate_model_player_counts,
                        "model_input_tokens": crewmate_model_input_tokens,
                        "model_output_tokens": crewmate_model_output_tokens,
                    },
                    f,
                )

    def analyze_persuasion_wins(self):
        # Directory containing tournament JSON files
        tournament_dir = "data/tournament"
        output_file = "data/persuasion_wins_analysis.csv"

        # List all JSON files in the directory
        tournament_files = [
            f for f in os.listdir(tournament_dir) if f.endswith(".json")
        ]

        # Create CSV file and write header
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'role', 'number_of_persuasive_phrases', 'is_win'])

            # Iterate over each file and load the game state
            for file_name in tournament_files:
                file_path = os.path.join(tournament_dir, file_name)
                game_engine = GameEngine()
                if game_engine.load_state(file_path):
                    game_state = game_engine.state
                    players = game_state.players

                    # Get annotations
                    annotation_file = os.path.join("data/annotations", file_name)
                    try:
                        with open(annotation_file, "r", encoding="utf-8") as f:
                            annotation_json = json.load(f)
                    except FileNotFoundError:
                        print(f"No annotation file found for: {file_name}")
                        continue

                    # Count persuasive phrases per player
                    previous_player = None
                    player_techniques = defaultdict(list)

                    for item in annotation_json:
                        replaced_text = item["text"]
                        current_player = (
                            replaced_text.split("]:")[0].strip("[]")
                            if "]: " in replaced_text
                            else previous_player
                        )

                        if item["annotation"]:
                            player_techniques[current_player].extend(item["annotation"])

                        previous_player = current_player

                    # Write data for each player
                    for player in players:
                        role = "impostor" if player.is_impostor else "crewmate"
                        num_techniques = len(player_techniques[player.name])
                        
                        # Determine if player won
                        is_win = False
                        impostor_wins = game_engine.check_impostors_win()
                        if player.is_impostor:
                            is_win = impostor_wins
                        else:
                            is_win = not impostor_wins

                        writer.writerow([file_name, role, num_techniques, is_win])

        print(f"Analysis complete. Results saved to {output_file}")

    def _display_tournament_persuasion_analysis(
        self,
        model_techniques: Dict[str, Dict[str, int]],
        model_player_counts: Dict[str, int],
        model_input_tokens: Dict[str, int],
        model_output_tokens: Dict[str, int],
        title: str = "Persuasion Techniques",
    ):
        st.title(title)
        if not model_techniques:
            st.warning("No data available. Please run the tournament analysis first.")
            return

        # --- Token Usage Chart (Keep this as it is) ---
        # self.plot_token_usage(model_input_tokens, model_output_tokens)

        # --- Techniques Table ---
        st.subheader("Technique Breakdown by Model")
        # Extract valid techniques from PERSUASION_TECHNIQUES string
        valid_techniques = []
        for line in PERSUASION_TECHNIQUES.split("\n"):
            if line.startswith("### ") and "**" in line:
                # Extract technique name between ** **
                technique = line.split("**")[1]
                valid_techniques.append(technique)

        # Filter all_techniques to only include valid ones
        all_techniques = sorted(
            set(
                technique
                for model_data in model_techniques.values()
                for technique in model_data
                # if technique in valid_techniques
            )
        )
        data = []
        data2 = []
        for model_name, techniques in model_techniques.items():
            row = {"Model": model_name}
            row2 = {"Model": model_name}
            row2["Total games"] = len(model_techniques[model_name])
            row2["Total Uses"] = sum(techniques.values())
            row2["Avg. per Game"] = (
                row2["Total Uses"] / row2["Total games"] if row2["Total games"] else 0
            )
            row2["Avg. per Player"] = (
                row2["Total Uses"] / model_player_counts[model_name]
                if model_player_counts[model_name]
                else 0
            )
            for technique in all_techniques:
                row[technique] = techniques.get(
                    technique, 0
                )  # Get count or 0 if not present
    
            data.append(row)
            data2.append(row2)

        df = pd.DataFrame(data)
        df = df.transpose()
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df.index.name = "Persuasion Technique"  # Add this line to name the index
        df2 = pd.DataFrame(data2)
        df2 = df2.transpose()
        df2.columns = df2.iloc[0]
        df2 = df2.iloc[1:]
        
        # add total techniques column
        df["Total Techniques"] = df.sum(axis=1)
        df2["Total"] = df2.sum(axis=1)
        
        # Join with the same technique in lowercase
        df.index = df.index.str.lower()
        df = df.groupby(df.index).sum()
        
        df = df[df["Total Techniques"] >= 10]
        
        st.dataframe(df2)
        st.dataframe(df.sort_values('Total Techniques', ascending=False))


        # Add CSV download button
        st.subheader(f"Download {title} Data")
        csv = df.to_csv(index=True)
        st.download_button(
            label=f"Download {title} Usage CSV",
            data=csv,
            key=f"{list(list(model_techniques.values())[0].values())[0]}",
            file_name=f"{title.lower().replace(' ', '_')}_usage.csv",
            mime="text/csv",
            help=f"Download a CSV file containing {title} usage statistics for each model",  # noqa: E501
        )
        return df

    def plot_token_usage(
        self,
        model_input_tokens: defaultdict[defaultdict[int]],
        model_output_tokens: defaultdict[defaultdict[int]],
    ):
        """Plots input and output token usage per model."""
        models = list(model_input_tokens.keys())
        fig = go.Figure()

        # Define colors for each model. Add more colors if needed.
        colors = [
            "cyan",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "blue",
        ]

        for i, model in enumerate(models):
            input_tokens_model = [
                model_input_tokens[model][filename]
                for filename in model_input_tokens[model].keys()
            ]
            output_tokens_model = [
                model_output_tokens[model][filename]
                for filename in model_output_tokens[model].keys()
            ]
            fig.add_trace(
                go.Scatter(
                    x=input_tokens_model,
                    y=output_tokens_model,
                    mode="markers",
                    name=model,
                    marker=dict(color=colors[i % len(colors)]),
                    hovertemplate="<br>Input: %{x}<br>Output: %{y}",
                )
            )

        fig.update_layout(
            title="Token Usage per Model",
            xaxis_title="Input Tokens",  # Corrected x-axis title
            yaxis_title="Output Tokens",  # Corrected y-axis title
            showlegend=True,  # Show the legend
            legend_title="Models",
        )

        st.plotly_chart(fig)

    def save_state_to_tournaments(self, game_engine: GameEngine):
        """Saves the game state to the tournaments folder."""
        impostor_model = None
        crewmate_model = None
        for player in game_engine.state.players:
            if player.is_impostor:
                impostor_model = player.adventure_agent.llm_model_name
            else:
                crewmate_model = player.adventure_agent.llm_model_name

        # Construct the filename
        impostor_model = impostor_model.split("/")[-1]
        crewmate_model = crewmate_model.split("/")[-1]
        filename = f"{impostor_model}_{crewmate_model}"
        i = 1
        while os.path.exists(f"data/tournament/{filename}_{i}.json"):
            i += 1
        filename = f"{filename}_{i}.json"

        # Copy the game state file to the tournaments folder
        shutil.copyfile("data/game_state.json", f"data/tournament/{filename}")
        st.success(
            f"Game state saved to tournament folder as {filename}. "
            "You can clear the game state now."
        )

    def _display_short_player_info(
        self, player: Player, current: bool, placeholder: DeltaGenerator
    ):
        with placeholder.container(border=True):
            self._display_name_role_status(player, current)
            self._display_tasks_progress(player)
            with st.expander("Info"):
                self._display_location(player)
                self._display_action_taken(player)
                self._display_action_result(player)
                self._display_recent_actions(player)
                self._display_tasks(player)

    def _display_name_role_status(self, player: Player, current: bool):
        status_icon = "✅" if player.state.life == PlayerState.ALIVE else "❌"
        role_icon = "😈" if player.role == PlayerRole.IMPOSTOR else "👤"
        current_icon = "⭐️" if current else ""
        complete_tasks = sum(1 for task in player.state.tasks if "DONE" in str(task))
        if player.role == PlayerRole.IMPOSTOR:
            st.write(
                f"{status_icon} {player.name} - "
                f"({complete_tasks}/{len(player.state.tasks)}) "
                f"{role_icon} ⏳{player.kill_cooldown} {current_icon}"
            )
        else:
            st.write(
                f"{status_icon} {player.name} - "
                f"({complete_tasks}/{len(player.state.tasks)}) "
                f"{role_icon} {current_icon}"
            )

    def _display_tasks_progress(self, player: Player):
        completed_tasks = sum(1 for task in player.state.tasks if "DONE" in str(task))
        total_tasks = len(player.state.tasks)
        st.progress(
            completed_tasks / total_tasks if total_tasks > 0 else 0
        )  # Handle division by zero

    def _display_tasks(self, player: Player):
        completed_tasks = sum(1 for task in player.state.tasks if "DONE" in str(task))
        total_tasks = len(player.state.tasks)
        st.write(f"Tasks: {completed_tasks}/{total_tasks}")
        st.write("Tasks:")
        for task in player.state.tasks:
            st.write(f"- {task}")

    def _display_location(self, player: Player):
        st.write(
            f"Location: {player.state.location.value} {player.state.player_in_room}"
        )

    def _display_action_taken(self, player: Player):
        action = player.state.response
        if action.isdigit():
            st.write(f"Action Taken: {player.state.actions[int(action)]}")
        else:
            st.write(f"Action Taken: {action}")

    def _display_action_result(self, player: Player):
        st.write(f"Action Result: {player.state.action_result}")

    def _display_recent_actions(self, player: Player):
        st.write("Seen Actions:")
        for action in player.state.seen_actions:
            st.write(f"- {action}")

    def _display_map(self, game_state: GameState):
        fig = go.Figure()
        img_width = 836
        img_height = 470
        scale_factor = 0.5

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig.add_trace(
            go.Scatter(
                x=[0, img_width * scale_factor],
                y=[0, img_height * scale_factor],
                mode="markers",
                marker_opacity=0,
            )
        )

        # Configure axes
        fig.update_xaxes(visible=False, range=[0, img_width * scale_factor])
        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source="https://d.techtimes.com/en/full/374414/electrical.png?w=836&f=111ca30545788b099bf5224400a2dbca",
            )
        )

        # Configure other layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        # Add player markers
        def update_player_markers(game_state: GameState):
            fig.data = []  # Clear existing traces
            for i, player in enumerate(game_state.players):
                x, y = ROOM_COORDINATES[player.state.location]
                marker_color = "yellow" if player.role == PlayerRole.CREWMATE else "red"
                marker_size = 15
                marker_symbol = (
                    "circle" if player.role == PlayerRole.CREWMATE else "square"
                )

                # Highlight the player to act next
                if i == game_state.player_to_act_next:
                    marker_size = 25
                    marker_symbol = "star"

                fig.add_trace(
                    go.Scatter(
                        x=[x * 100 + random.randint(-10, 10)],
                        y=[y * 100 + random.randint(-10, 10)],
                        mode="markers",
                        showlegend=False,
                        marker=dict(
                            color=marker_color,
                            size=marker_size,
                            symbol=marker_symbol,
                        ),
                        name=player.name,
                        customdata=[
                            f"<b>{player.name}</b><br>"
                            f"Role: {player.role.value}<br>"
                            f"Status: {player.state.life.value}"
                        ],
                        hovertemplate="%{customdata}",
                    )
                )

        update_player_markers(game_state)

        # Display the map
        map_placeholder = st.empty()
        map_placeholder.plotly_chart(fig, use_container_width=True, key=uuid.uuid4())

    def _display_annotated_text(
        self, annotation_json: List[dict], players: List[Player]
    ):
        args = []
        previous_player = None
        player_techniques = defaultdict(list)

        for item in annotation_json:
            replaced_text = item["text"]
            current_player = (
                replaced_text.split("]:")[0].strip("[]")
                if "]: " in replaced_text
                else previous_player
            )

            if previous_player and previous_player != current_player:
                args.append("\n\n")

            if item["annotation"]:
                combined_annotation = ", ".join(item["annotation"])
                args.append((replaced_text, combined_annotation))
                player_techniques[current_player].extend(item["annotation"])
            else:
                args.append(replaced_text)

            previous_player = current_player

        annotated_text(*args)

        # Determine unique models used by each team
        crewmates = [p for p in players if not p.is_impostor]
        impostors = [p for p in players if p.is_impostor]

        crewmate_models = set(p.llm_model_name for p in crewmates)
        impostor_models = set(p.llm_model_name for p in impostors)

        # Flag to check if only one model per team exists
        single_model_per_team = len(crewmate_models) == 1 and len(impostor_models) == 1

        # Summary function with conditional header and model listing
        def summarize_team(team: list[Player], team_name: str, models: set[str]):
            model_names = ", ".join(models)
            header_text = (
                f"Summary for {team_name} - Model: {model_names}"
                if single_model_per_team
                else f"Summary for {team_name}"
            )
            st.subheader(header_text)
            if not single_model_per_team:
                st.write(f"Models used: {model_names}")

            st.write(f"Number of {team_name.lower()}: {len(team)}")
            st.write("Players:", ", ".join([p.name for p in team]))

            total_techniques = sum(len(player_techniques[p.name]) for p in team)
            avg_techniques = total_techniques / len(team) if team else 0
            st.write(f"Total techniques: {total_techniques}")
            st.write(f"Average techniques per player: {avg_techniques:.2f}")

            # Technique breakdown for the team
            all_team_techniques = [
                tech for p in team for tech in player_techniques[p.name]
            ]
            team_technique_counts = Counter(all_team_techniques)
            st.markdown("**Technique breakdown:**")
            for technique, count in team_technique_counts.items():
                avg_per_player = count / len(team) if team else 0
                st.write(
                    f" - {technique}: {count} times - "
                    f"(avg per player: {avg_per_player:.2f})"
                )

        # Summaries for crewmates and impostors
        summarize_team(crewmates, "Crewmates", crewmate_models)
        summarize_team(impostors, "Impostors", impostor_models)

        # Add model summary if multiple models are used within teams
        if not single_model_per_team:
            st.subheader("Summary by Model")
            models = defaultdict(list)
            for player in players:
                models[player.llm_model_name].append(player)

            for model_name, model_players in models.items():
                st.markdown(f"### Model: {model_name}")
                st.write(f"Number of players: {len(model_players)}")
                st.write("Players:", ", ".join([p.name for p in model_players]))

                total_techniques = sum(
                    len(player_techniques[p.name]) for p in model_players
                )
                avg_techniques = (
                    total_techniques / len(model_players) if model_players else 0
                )
                st.write(f"Total techniques: {total_techniques}")
                st.write(f"Average techniques per player: {avg_techniques:.2f}")

                # Technique breakdown for this model
                all_model_techniques = [
                    tech for p in model_players for tech in player_techniques[p.name]
                ]
                model_technique_counts = Counter(all_model_techniques)
                st.markdown("**Technique breakdown:**")
                for technique, count in model_technique_counts.items():
                    avg_per_player = count / len(model_players) if model_players else 0
                    st.write(
                        f" - {technique}: {count} times "
                        f"(avg per player: {avg_per_player:.2f})"
                    )

    def _display_player_selection(self, players: List[Player]):
        selected_player = st.radio(
            "Select Player to see their discussion and llm messages:",
            [len(players)] + list(range(len(players))),
            horizontal=True,
            key=f"player_selection_{players}",
            format_func=lambda i: players[i].name if i < len(players) else "None",
        )
        st.session_state.selected_player = selected_player

    def _display_discussion_chat(self, players: List[Player]):
        discussion_chat = ""
        if st.session_state.selected_player == len(players):
            for player in players:
                if player.state.life == PlayerState.ALIVE:
                    discussion_chat = "\n".join(player.get_chat_messages())
                    break
        else:
            player = players[st.session_state.selected_player]
            discussion_chat = "\n".join([
                x
                for x in player.get_chat_messages()
                if x.startswith(f"[{player.name}]")
            ])
        st.text_area(label="Discussion log:", value=discussion_chat)
        return discussion_chat

    def get_cost_data(self, game_engine: GameEngine) -> Dict[str, List[float]]:
        """Extracts cost data from player history.

        Returns:
            Cost mapping
        """
        cost_data = {}
        for player in game_engine.state.players:
            costs = [round(r.token_usage.cost, 4) for r in player.history.rounds]
            cost_data[player.name] = costs
        return cost_data

    def estimate_future_cost(
        self,
        player_costs: Dict[str, List[float]],
        rounds_to_forecast: int,
        degree: int = 2,
    ) -> Dict[str, List[float]]:
        """Estimates future cost using polynomial regression for each player.

        Returns:
            Estimated cost
        """
        estimated_cost_data = {}
        for player_name, costs in player_costs.items():
            # Prepare data for linear regression
            X = [[i] for i in range(len(costs))]
            y = costs

            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)

            # Train a separate model for each player
            model = LinearRegression()
            model.fit(X_poly, y)

            # Estimate future costs
            future_rounds = [
                [i] for i in range(len(costs), len(costs) + rounds_to_forecast)
            ]
            future_rounds_poly = poly.transform(future_rounds)
            estimated_costs = [
                round(model.predict([future_round])[0], 4)
                for future_round in future_rounds_poly
            ]
            estimated_cost_data[player_name] = estimated_costs
        return estimated_cost_data

    def combine_data(
        self,
        player_costs: Dict[str, List[float]],
        estimated_player_costs: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        """Combines actual and estimated cost data.

        Returns:
            Chained costs
        """
        combined_player_costs = {}
        for player_name in player_costs:
            combined_player_costs[player_name] = (
                player_costs[player_name] + estimated_player_costs[player_name]
            )
        return combined_player_costs

    def plot_cost(self, player_costs: Dict[str, List[float]], rounds_to_forecast: int):
        """Plots cost data using Plotly."""
        fig = go.Figure()
        history = list(player_costs.values())[0]

        for player_name, costs in player_costs.items():
            # Separate actual and estimated costs
            actual_costs = costs[: len(history) - rounds_to_forecast]
            estimated_costs = costs[len(history) - rounds_to_forecast :]

            # Plot actual costs as solid lines
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(actual_costs) + 1)),
                    y=actual_costs,
                    name=player_name,
                    mode="lines",
                )
            )

            # Plot estimated costs as dashed lines
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(actual_costs), len(costs) + 1)),
                    y=[actual_costs[-1]] + estimated_costs,
                    name=player_name,
                    mode="lines",
                    line=dict(dash="dash"),
                )
            )

        # Calculate total cost
        total_costs = [
            sum(costs[i] for costs in player_costs.values())
            for i in range(len(history))
        ]

        # Separate actual and estimated total costs
        actual_total_costs = total_costs[: len(history) - rounds_to_forecast]
        estimated_total_costs = total_costs[len(history) - rounds_to_forecast :]

        # Plot actual total cost as solid lines
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(actual_total_costs) + 1)),
                y=actual_total_costs,
                name="Total Cost",
                mode="lines",
            )
        )

        # Plot estimated total cost as dashed lines
        fig.add_trace(
            go.Scatter(
                x=list(range(len(actual_total_costs), len(total_costs) + 1)),
                y=[actual_total_costs[-1]] + estimated_total_costs,
                name="Total Cost",
                mode="lines",
                line=dict(dash="dash"),
            )
        )

        fig.update_layout(
            title="Player Cost Over Rounds",
            xaxis_title="Round Number",
            yaxis_title="Cost",
            legend_title="Players",
        )

        st.plotly_chart(fig)

    def _display_chat_history(self, rounds: List[RoundData]):
        """Displays the chat history for a player using st.chat_message."""
        with st.container(height=500, border=True):
            for i, round_data in enumerate(rounds):
                st.markdown(f"### Round {i}")
                if not round_data.prompts:
                    continue
                # 1. Prompt (Player Message)
                with st.chat_message("user"):
                    with st.expander("Prompt"):
                        st.write(round_data.prompts[0])

                # 2. Actions (System Message)
                if round_data.actions:  # Check if actions exist
                    with st.chat_message("system"):
                        st.write("Actions:")
                        for i, action in enumerate(round_data.actions):
                            st.write(f"{i}. {action}")

                # 3. Response (LLM Message)
                with st.chat_message("assistant"):
                    with st.expander("LLM Response"):  # Put llm_responses in expander
                        st.write(round_data.llm_responses[0])

                if len(round_data.prompts) > 1:
                    with st.chat_message("user"):
                        with st.expander("Action Prompt"):
                            st.write(round_data.prompts[1])
                    with st.chat_message("assistant"):
                        st.write(round_data.llm_responses[1])

                # 4. Action Result (System Message)
                if round_data.action_result:  # Check if action result exists
                    with st.chat_message("system"):
                        st.write(f"Action Result: {round_data.action_result}")

    def game_settings(self):
        """Displays the game settings tab for player configuration."""
        st.title("Game Settings")
        self._handle_tournament_file_selection(None)

        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            crewmate_count = st.number_input(
                "Number of Crewmates", min_value=1, max_value=10, value=4
            )
        with col2:
            impostor_count = st.number_input(
                "Number of Impostors", min_value=1, max_value=10 - 1, value=1
            )

        # Create a game engine instance
        game_engine = GameEngine()

        # Player configuration
        player_names = [
            "Alice",
            "Bob",
            "Charlie",
            "Dave",
            "Erin",
            "Frank",
            "Grace",
            "Henry",
            "Isabella",
            "Jack",
            "Katie",
            "Liam",
            "Mia",
            "Noah",
            "Olivia",
            "Peter",
            "Quinn",
            "Ryan",
            "Sara",
            "Tom",
            "Ursula",
            "Victor",
            "Wendy",
            "Xander",
            "Yara",
            "Zane",
        ]
        players = []
        models = sorted(
            TOKEN_COSTS.keys(), key=lambda x: TOKEN_COSTS[x]["input_tokens"]
        )
        col3, col4 = st.columns([1, 2])
        with col3:
            st.header("Crewmates:")

        def model_format_func(model: str):
            return (
                f"{model} - {TOKEN_COSTS[model]['input_tokens']}$ / "
                f"{TOKEN_COSTS[model]['output_tokens']}$"
            )

        with col4:
            crewmate_model = st.selectbox(
                "Model",
                models,
                format_func=model_format_func,
                key="crewmate_model_selection",
            )
        cols_crewmate = st.columns(crewmate_count)
        for i in range(crewmate_count + impostor_count):
            if i == crewmate_count:
                st.markdown("---")  # Separator for impostors
                col5, col6 = st.columns([1, 2])
                with col5:
                    st.header("Impostors:")
                with col6:
                    impostor_model = st.selectbox(
                        "Model",
                        models,
                        format_func=model_format_func,
                        key="impostor_model_selection",
                    )
                cols_impostor = st.columns(impostor_count)
            model_name = crewmate_model if i < crewmate_count else impostor_model
            if i < crewmate_count:
                with cols_crewmate[i]:
                    player_name = st.selectbox(
                        f"Player {i + 1}",
                        player_names,
                        index=i,
                        key=f"player_selection_{i}",
                    )
            else:
                with cols_impostor[i - crewmate_count]:
                    player_name = st.selectbox(
                        f"Player {i + 1}",
                        player_names,
                        index=i,
                        key=f"player_selection_{i}",
                    )
            # Create player object
            player = AIPlayer(
                name=player_name,
                llm_model_name=model_name,
                role=PlayerRole.IMPOSTOR
                if i >= crewmate_count
                else PlayerRole.CREWMATE,
            )
            players.append(player)
        # players = players[1:]  # Remove the first player
        # players.append(HumanPlayer(name="Mateusz", role=PlayerRole.CREWMATE))

        # Confirmation button to start the game
        if st.button("Start Game"):
            random.shuffle(players)
            game_engine.load_players(players, impostor_count=impostor_count)
            game_engine.state.set_stage(GamePhase.ACTION_PHASE)
            game_engine.save_state()

        # Configuration Settings
        st.markdown("---")  # Separator for configuration settings
        st.header("Game Consts Configuration")
        st.markdown("> Remember to save the settings after changing them!")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            num_short_tasks = st.number_input(
                "Number of Short Tasks",
                min_value=1,
                max_value=10,
                value=NUM_SHORT_TASKS,
            )
        with col2:
            num_long_tasks = st.number_input(
                "Number of Long Tasks", min_value=1, max_value=10, value=NUM_LONG_TASKS
            )
        with col3:
            num_chats = st.number_input(
                "Number of Discussion rounds",
                min_value=1,
                max_value=10,
                value=NUM_CHATS,
            )
        with col4:
            impostor_cooldown = st.number_input(
                "Impostor Cooldown", min_value=0, max_value=10, value=IMPOSTOR_COOLDOWN
            )
        with col5:
            state_file = st.text_input("Game State File", value=STATE_FILE)
        adventure_plan_system_prompt = st.text_area(
            "Adventure Plan System Prompt", value=ADVENTURE_PLAN_SYSTEM_PROMPT
        )
        adventure_plan_user_prompt = st.text_area(
            "Adventure Plan User Prompt", value=ADVENTURE_PLAN_USER_PROMPT
        )
        adventure_action_system_prompt = st.text_area(
            "Adventure Action System Prompt", value=ADVENTURE_ACTION_SYSTEM_PROMPT
        )
        adventure_action_user_prompt = st.text_area(
            "Adventure Action User Prompt", value=ADVENTURE_ACTION_USER_PROMPT
        )
        discussion_system_prompt = st.text_area(
            "Discussion System Prompt", value=DISCUSSION_SYSTEM_PROMPT
        )
        discussion_user_prompt = st.text_area(
            "Discussion User Prompt", value=DISCUSSION_USER_PROMPT
        )
        discussion_response_system_prompt = st.text_area(
            "Discussion Response System Prompt", value=DISCUSSION_RESPONSE_SYSTEM_PROMPT
        )
        discussion_response_user_prompt = st.text_area(
            "Discussion Response User Prompt", value=DISCUSSION_RESPONSE_USER_PROMPT
        )
        voting_system_prompt = st.text_area(
            "Voting System Prompt", value=VOTING_SYSTEM_PROMPT
        )
        voting_user_prompt = st.text_area(
            "Voting User Prompt", value=VOTING_USER_PROMPT
        )
        annotation_system_prompt = st.text_area(
            "Annotation System prompt", value=ANNOTATION_SYSTEM_PROMPT
        )
        if st.button("Save Settings"):
            # Here GAME_CONTEXT is integrated into the prompts
            with open("src/among_them/game/llm_prompts.py", "w") as f:
                f.write(
                    f'ANNOTATION_SYSTEM_PROMPT = """{annotation_system_prompt}"""\n\n'
                )
                f.write(
                    f'ADVENTURE_PLAN_SYSTEM_PROMPT = """{adventure_plan_system_prompt}"""\n\n'  # noqa: E501
                )
                f.write(
                    f'ADVENTURE_PLAN_USER_PROMPT = """{adventure_plan_user_prompt}"""\n\n'  # noqa: E501
                )
                f.write(
                    f'ADVENTURE_ACTION_SYSTEM_PROMPT = """{adventure_action_system_prompt}"""\n\n'  # noqa: E501
                )
                f.write(
                    f'ADVENTURE_ACTION_USER_PROMPT = """{adventure_action_user_prompt}"""\n\n'  # noqa: E501
                )
                f.write(
                    f'DISCUSSION_SYSTEM_PROMPT = """{discussion_system_prompt}"""\n\n'  # noqa: E501
                )
                f.write(f'DISCUSSION_USER_PROMPT = """{discussion_user_prompt}"""\n\n')
                f.write(
                    f'DISCUSSION_RESPONSE_SYSTEM_PROMPT = """{discussion_response_system_prompt}"""\n\n'  # noqa: E501
                )
                f.write(
                    f'DISCUSSION_RESPONSE_USER_PROMPT = """{discussion_response_user_prompt}"""\n\n'  # noqa: E501
                )
                f.write(f'VOTING_SYSTEM_PROMPT = """{voting_system_prompt}"""\n\n')
                f.write(f'VOTING_USER_PROMPT = """{voting_user_prompt}"""\n')
            with open("src/among_them/game/consts.py", "w") as f:
                f.write(f"NUM_SHORT_TASKS = {num_short_tasks}\n")
                f.write(f"NUM_LONG_TASKS = {num_long_tasks}\n")
                f.write(f"NUM_CHATS = {num_chats}\n")
                f.write(f"IMPOSTOR_COOLDOWN = {impostor_cooldown}\n")
                f.write(f'STATE_FILE = "{state_file}"\n')
                f.write(f"TOKEN_COSTS = {TOKEN_COSTS}")
            st.success("Settings saved successfully!")

    def _display_persuasion_techniques(self):
        st.title("Persuasion Techniques")
        st.markdown(PERSUASION_TECHNIQUES)