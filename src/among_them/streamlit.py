import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.path import Path

from among_them.config import STATE_FILE
from among_them.game_jsonencoder import game_object_hook
from among_them.models.action_type import ActionType
from among_them.models.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import get_map, Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole

st.set_page_config(
    page_title="Among Them - Game Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style
st.markdown("""
<style>
    .task-complete { color: green; }
    .task-incomplete { color: orange; }
    .player-impostor { color: red; }
    .player-crewmate { color: blue; }
    .highlight-action { background-color: #FFF0B3; padding: 5px; border-radius: 3px; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size:1.2rem;}
</style>
""", unsafe_allow_html=True)

# File selector in sidebar
st.sidebar.title("Among Them Game Visualizer")

# Load most recent by default, but allow selection of other files
data_dir = "data"
game_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
game_files.sort(reverse=True)  # Most recent first (if timestamps are in filenames)

if STATE_FILE.split('/')[-1] in game_files:
    game_files.remove(STATE_FILE.split('/')[-1])
    game_files = [STATE_FILE.split('/')[-1]] + game_files  # Put current game at top

selected_file = st.sidebar.selectbox("Select Game File", game_files)
file_path = os.path.join(data_dir, selected_file)

# Add auto-refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 30, 5)

last_modified_time = os.path.getmtime(file_path)
last_modified_str = datetime.fromtimestamp(last_modified_time).strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.write(f"Last Modified: {last_modified_str}")

# Load game state
@st.cache_data(ttl=refresh_interval)
def load_game_state(file_path, last_modified_time) -> Tuple[List[History], List[Player], float, GameConfig]:
    with open(file_path, 'r') as f:
        state_str = f.read()
        loaded_data = json.loads(state_str, object_hook=game_object_hook)
        if len(loaded_data) == 3:
            history, players, game_config = loaded_data
        elif len(loaded_data) == 2: # Handle old save files without config
            print("Loading old save file format. Using default GameConfig.")
            history, players = loaded_data
            game_config = GameConfig() # Initialize with defaults
        else:
            raise ValueError("Invalid save file format")
        return history, players, last_modified_time, game_config

try:
    history, players, _, game_config = load_game_state(file_path, last_modified_time)
    
    # Check if auto-refresh is enabled
    if auto_refresh:
        st.sidebar.write("Auto-refreshing...")
        time.sleep(2)  # Small delay to prevent excessive refreshing
        st.rerun()
        
    # Main tabs
    tabs = st.tabs(["Game Map", "Timeline", "Tasks", "Player Network", "Voting Patterns", "Raw Data"])

    with tabs[0]:  # Game Map
        st.header("Game Map Visualization")
        
        # Create a figure for the map
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the map (rooms and connections)
        for room_name, (x, y) in get_map(game_config.map_size)[1].items():
            circle = plt.Circle((x, y), 0.15, fill=True, alpha=0.2, color='lightgray')
            ax.add_patch(circle)
            ax.text(x, y, room_name.value, ha='center', va='center', fontsize=8)
        
        # Draw connections
        doors, room_coordinates, _ = get_map(game_config.map_size)
        for room_name, connected_rooms in doors.items():
            room_x, room_y = room_coordinates[room_name]
            for connected_room in connected_rooms:
                conn_x, conn_y = room_coordinates[connected_room]
                ax.plot([room_x, conn_x], [room_y, conn_y], 'k-', alpha=0.3, linewidth=1)
        
        # Get the latest history entry
        latest_entry = history[-1]
        
        # Track player positions
        player_positions: Dict[str, Optional[Location]] = {}
        
        # Go through history to find player last known locations
        for entry in history:
            player_name = entry.action_taken.player_name
            if player_name != "System" and hasattr(entry, 'location') and entry.location:
                player_positions[player_name] = entry.location
        
        # Get player roles
        player_roles = {player.name: player.role for player in players}
        
        # Collect all players and determine status (alive or dead)
        all_player_names = set()
        dead_players = set()
        
        for player in players:
            all_player_names.add(player.name)
        
        for entry in history:
            if entry.action_taken.type == ActionType.KILL and entry.action_taken.target_player_name:
                dead_players.add(entry.action_taken.target_player_name)
        
        # Draw players on map
        player_colors = {
            PlayerRole.CREWMATE: 'blue',
            PlayerRole.IMPOSTOR: 'red'
        }
        
        # Create legend elements
        legend_elements = [
            patches.Patch(facecolor='blue', edgecolor='black', label='Crewmate', alpha=0.6),
            patches.Patch(facecolor='red', edgecolor='black', label='Impostor', alpha=0.6),
            patches.Patch(facecolor='gray', edgecolor='black', label='Dead Player', alpha=0.3)
        ]
        
        # Plot player positions
        for player_name, location in player_positions.items():
            if location:
                location_name = location.value if hasattr(location, 'value') else location
                x, y = get_map(game_config.map_size)[1][location]
                
                # Offset player positions slightly to prevent overlap
                x += np.random.uniform(-0.05, 0.05)
                y += np.random.uniform(-0.05, 0.05)
                
                # Determine if player is alive and get their role color
                is_dead = player_name in dead_players
                role = player_roles.get(player_name, PlayerRole.CREWMATE)
                color = player_colors.get(role, 'gray')
                
                # Adjust appearance based on alive/dead status
                alpha = 0.3 if is_dead else 0.6
                marker = 'x' if is_dead else 'o'
                
                # Plot player
                ax.plot(x, y, marker, color=color, markersize=10, alpha=alpha)
                ax.text(x, y+0.08, player_name, ha='center', va='center', fontsize=8)
        
        # Add a legend
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Set equal aspect and limits
        ax.set_aspect('equal')
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 2.5)
        ax.set_title(f"Game Map - Turn {len(history)}")
        
        # Display the map
        st.pyplot(fig)
        
        # Current game phase
        current_phase = latest_entry.phase
        actions_left = latest_entry.actions_until_phase_ends
        st.info(f"Current Phase: {current_phase} (Actions remaining: {actions_left})")
        
        # Latest action information
        st.subheader("Latest Action")
        cols = st.columns(3)
        cols[0].write(f"Player: {latest_entry.action_taken.player_name}")
        cols[1].write(f"Action: {latest_entry.action_taken.type}")
        cols[2].write(f"Location: {latest_entry.location}")
        
        if latest_entry.action_taken.spectator:
            st.write(f"Result: {latest_entry.action_taken.spectator}")

    with tabs[1]:  # Timeline
        st.header("Game Timeline")
        
        # Create a dataframe of history entries
        timeline_data = []
        
        for i, entry in enumerate(history):
            timeline_data.append({
                "Turn": i + 1,
                "Player": entry.action_taken.player_name,
                "Phase": entry.phase,
                "Action": entry.action_taken.type,
                "Location": entry.location,
                "Result": entry.action_taken.spectator
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Allow filtering by player
        player_names = ["All"] + sorted(list(set(df_timeline["Player"])))
        selected_player = st.selectbox("Filter by Player", player_names)
        
        # Apply filter
        if selected_player != "All":
            df_filtered = df_timeline[df_timeline["Player"] == selected_player]
        else:
            df_filtered = df_timeline
        
        # Show the timeline
        st.dataframe(df_filtered, use_container_width=True)
        
        # Create a visual timeline with matplotlib
        if not df_filtered.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create colormap for different action types
            action_types = df_timeline["Action"].unique()
            cmap = plt.cm.get_cmap('tab10', len(action_types))
            
            action_colors = {}
            for i, action in enumerate(action_types):
                action_colors[action] = cmap(i)
            
            # Plot actions as points on the timeline
            for i, row in df_filtered.iterrows():
                action_color = action_colors.get(row["Action"], 'gray')
                ax.scatter(row["Turn"], 0, c=[action_color], s=100, zorder=2)
                
                # Add player name as text
                player_text = row["Player"]
                ax.text(row["Turn"], 0.1, player_text, ha='center', va='bottom', 
                        fontsize=8, rotation=45)
                
                # Add action type
                action_text = f"{row['Action']}"
                ax.text(row["Turn"], -0.1, action_text, ha='center', va='top', 
                        fontsize=8, rotation=45)
            
            # Draw a horizontal line through all points
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, zorder=1)
            
            # Add phase transitions
            phase_changes = []
            prev_phase = None
            for i, row in df_timeline.iterrows():
                if prev_phase != row["Phase"]:
                    phase_changes.append((row["Turn"], row["Phase"]))
                    prev_phase = row["Phase"]
            
            for turn, phase in phase_changes:
                ax.axvline(x=turn, color='red', linestyle='--', alpha=0.5, zorder=1)
                ax.text(turn, 0.3, f"{phase}", ha='center', va='bottom', 
                        fontsize=10, color='red', rotation=45)
            
            # Remove y-axis ticks and labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # Set x-axis limits and ticks
            ax.set_xlim(df_filtered["Turn"].min() - 0.5, df_filtered["Turn"].max() + 0.5)
            ax.set_xticks(df_filtered["Turn"])
            ax.set_ylim(-0.5, 0.5)
            
            # Add a title and labels
            ax.set_title("Game Timeline")
            ax.set_xlabel("Turn")
            
            # Create a legend for action types
            legend_elements = [
                patches.Patch(facecolor=action_colors.get(action), label=action)
                for action in action_colors
            ]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=3)
            
            # Display the timeline
            st.pyplot(fig)

    with tabs[2]:  # Tasks
        st.header("Task Completion Status")
        
        # Get task status from the last history entry
        tasks_status = latest_entry.tasks_left_to_do
        
        # Create columns for each player
        player_cols = st.columns(len(tasks_status))
        
        # Display tasks for each player
        for i, (player_name, tasks) in enumerate(tasks_status.items()):
            with player_cols[i]:
                # Determine if player is alive
                is_alive = player_name not in dead_players
                
                # Get player role
                role = player_roles.get(player_name, PlayerRole.CREWMATE)
                role_text = "👨‍🚀 Crewmate" if role == PlayerRole.CREWMATE else "👽 Impostor"
                
                # Display player header with status
                status_text = "🟢 Alive" if is_alive else "💀 Dead"
                st.subheader(f"{player_name} ({status_text})")
                st.write(f"Role: {role_text}")
                
                # Display tasks
                if not tasks:
                    st.write("No tasks remaining! 🎉")
                else:
                    task_complete = game_config.num_tasks - len(tasks)
                    task_total = game_config.num_tasks
                    
                    # Count completed tasks
                    for task in tasks:
                        if hasattr(task, 'completed') and task.completed:
                            task_complete += 1
                    
                    # Show progress bar
                    st.progress(task_complete / task_total)
                    st.write(f"{task_complete}/{task_total} tasks completed")
                    
                    # List individual tasks
                    for task in tasks:
                        task_name = task.name if hasattr(task, 'name') else str(task)
                        task_location = task.location.value if hasattr(task, 'location') and task.location else "Unknown"
                        task_status = "✅" if hasattr(task, 'completed') and task.completed else "⏳"
                        
                        # If it's a long task, show turns left
                        turns_info = ""
                        if hasattr(task, 'turns_left'):
                            turns_info = f" ({task.turns_left} turns left)"
                        
                        st.write(f"{task_status} {task_name} at {task_location}{turns_info}")

    with tabs[3]:  # Player Network
        st.header("Player Interaction Network")
        
        # Create a network graph
        G = nx.Graph()
        
        # Add nodes for each player
        for player in players:
            G.add_node(player.name, role=player.role)
        
        # Track interactions between players
        interactions: dict[tuple[str, str], int] = defaultdict(int)
        
        # Count interactions (players in same room)
        for entry in history:
            if entry.spectators_who_saw and len(entry.spectators_who_saw) > 1:
                for i, player1 in enumerate(entry.spectators_who_saw):
                    for player2 in entry.spectators_who_saw[i+1:]:
                        # Increment the count for this pair
                        if player1 != player2:  # Avoid self-loops
                            interactions[(player1, player2)] += 1
        
        # Add edges for interactions
        for (player1, player2), count in interactions.items():
            if count > 0:  # Only add edge if interaction occurred
                G.add_edge(player1, player2, weight=count)
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Set node colors based on role
        node_colors = []
        for node in G.nodes():
            try:
                player_obj = next((p for p in players if p.name == node), None)
                if player_obj:
                    role = player_obj.role
                    if role == PlayerRole.IMPOSTOR:
                        node_colors.append('red')
                    else:
                        node_colors.append('blue')
                else:
                    node_colors.append('gray')
            except:
                node_colors.append('gray')
        
        # Adjust node size based on degree
        node_size = [300 + 100 * G.degree(node) for node in G.nodes()]
        
        # Adjust edge width based on number of interactions
        edge_width = [0.5 + G[u][v]['weight'] / 2 for u, v in G.edges()]
        
        # Use a better layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the network
        nx.draw_networkx(
            G, 
            pos=pos, 
            node_color=node_colors,
            node_size=node_size,
            with_labels=True,
            font_size=10,
            alpha=0.7,
            width=edge_width
        )
        
        # Create a clear figure with a white background
        plt.tight_layout()
        plt.axis('off')
        
        # Add a title
        plt.title("Player Interaction Network")
        
        # Display the graph
        st.pyplot(plt.gcf())
        
        # Display interaction tables
        st.subheader("Player Co-Location Frequency")
        
        # Create a DataFrame for interactions
        interaction_data = []
        
        for (player1, player2), count in interactions.items():
            interaction_data.append({
                "Player 1": player1,
                "Player 2": player2,
                "Times in Same Room": count
            })
        
        # Sort by count (highest first)
        interaction_df = pd.DataFrame(interaction_data).sort_values(
            by="Times in Same Room", ascending=False
        )
        
        # Display table
        st.dataframe(interaction_df)
        
        # Display suspicion analysis
        st.subheader("Suspicion Analysis")
        
        # Extract SPEAK actions related to suspicion
        discussions = []
        
        for entry in history:
            if entry.action_taken.type == ActionType.SPEAK:
                speaker = entry.action_taken.player_name
                message = entry.action_taken.spectator
                discussions.append({
                    "Speaker": speaker,
                    "Message": message
                })
        
        # Display discussion records
        if discussions:
            discussion_df = pd.DataFrame(discussions)
            st.dataframe(discussion_df)
        else:
            st.write("No discussions recorded yet.")

    with tabs[4]:  # Voting Patterns
        st.header("Voting Pattern Visualization")
        
        # Extract all discussion phase entries with voting data
        voting_data = []
        discussion_entries = []
        
        for i, entry in enumerate(history):
            if entry.phase == GamePhase.DISCUSS and hasattr(entry, 'votes_before_this_discussion_message') and entry.votes_before_this_discussion_message is not None:
                discussion_entries.append(entry)
                
                # Store vote information
                if entry.votes_before_this_discussion_message:
                    turn_num = i + 1
                    for voter, vote_info in entry.votes_before_this_discussion_message.items():
                        votee = vote_info["voted_player"]
                        cot = vote_info.get("chain_of_thought", "")
                        
                        voting_data.append({
                            "Turn": turn_num,
                            "Voter": voter,
                            "Votee": votee,
                            "CoT": cot,
                            "Speaker": entry.action_taken.player_name,
                            "Message": entry.action_taken.spectator if entry.action_taken.spectator else "No message"
                        })
        
        if not voting_data:
            st.info("No voting data available. Voting data only appears during discussion phases.")
        else:
            # Create a dataframe from voting data
            df_votes = pd.DataFrame(voting_data)
            
            # Convert GamePhase objects to strings to avoid Arrow serialization issues
            if "Phase" in df_votes.columns:
                df_votes["Phase"] = df_votes["Phase"].astype(str)
            
            # Group voting data by turn
            vote_turns = sorted(df_votes["Turn"].unique())
            
            # Allow users to select which turns to compare
            st.subheader("Select Turns to Compare Votes")
            col1, col2 = st.columns(2)
            
            with col1:
                start_turn = st.selectbox(
                    "Starting Turn", 
                    vote_turns, 
                    index=0,
                    key="start_vote_turn"
                )
            
            with col2:
                # Filter for turns after the start turn
                valid_end_turns = [t for t in vote_turns if t > start_turn]
                end_turn = st.selectbox(
                    "Ending Turn", 
                    valid_end_turns if valid_end_turns else [start_turn], 
                    index=min(1, len(valid_end_turns)-1) if valid_end_turns else 0,
                    key="end_vote_turn"
                )
            
            # Get the voting data for selected turns
            start_votes = df_votes[df_votes["Turn"] == start_turn]
            end_votes = df_votes[df_votes["Turn"] == end_turn]
            
            # Create Sankey diagram to show vote changes
            if not start_votes.empty and not end_votes.empty:
                st.subheader(f"Vote Changes Between Turn {start_turn} and Turn {end_turn}")
                
                # Get the vote maps for easier comparison
                start_vote_map = dict(zip(start_votes["Voter"], start_votes["Votee"]))
                end_vote_map = dict(zip(end_votes["Voter"], end_votes["Votee"]))
                
                # Track which voters changed their votes
                vote_changes = {}
                for voter in start_vote_map.keys():
                    if voter in end_vote_map and start_vote_map[voter] != end_vote_map[voter]:
                        vote_changes[voter] = (start_vote_map[voter], end_vote_map[voter])
                
                # Create a directed graph to visualize vote changes
                vote_graph = nx.DiGraph()
                
                # Add nodes for all players
                all_players = sorted(list(set(list(start_votes["Voter"]) + list(start_votes["Votee"]) + 
                                           list(end_votes["Voter"]) + list(end_votes["Votee"]))))
                
                for player in all_players:
                    vote_graph.add_node(player)
                
                # Add edges for vote changes with labels
                for voter, (old_vote, new_vote) in vote_changes.items():
                    if old_vote != new_vote:  # Only show actual changes
                        vote_graph.add_edge(old_vote, new_vote, voter=voter, weight=2)
                
                if vote_changes:
                    # Create a directed graph visualization
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Position the nodes in a circle
                    pos = nx.circular_layout(vote_graph)
                    
                    # Draw the nodes
                    nx.draw_networkx_nodes(vote_graph, pos, node_size=2000, 
                                          node_color='lightblue', ax=ax)
                    
                    # Draw the node labels
                    nx.draw_networkx_labels(vote_graph, pos, font_size=12, font_weight='bold', ax=ax)
                    
                    # Draw the edges with appropriate styling
                    edges = vote_graph.edges()
                    if edges:
                        # Create edge colors based on weight
                        edge_colors = ['red' for _ in edges]
                        
                        # Draw the edges with arrows
                        nx.draw_networkx_edges(vote_graph, pos, edgelist=edges, width=2, 
                                               edge_color=edge_colors, arrows=True, 
                                               arrowsize=20, arrowstyle='->', 
                                               connectionstyle='arc3,rad=0.2', ax=ax)
                        
                        # Add edge labels (showing which player changed their vote)
                        edge_labels = {(u, v): vote_graph[u][v]['voter'] for u, v in vote_graph.edges()}
                        nx.draw_networkx_edge_labels(vote_graph, pos, edge_labels=edge_labels, 
                                                   font_size=10, label_pos=0.3, ax=ax)
                    
                    ax.set_title(f"Vote Changes Between Turn {start_turn} and Turn {end_turn}")
                    ax.axis('off')
                    
                    # Display the directed graph
                    st.pyplot(fig)
                    
                    # Display vote change details in a table
                    st.subheader("Players Who Changed Their Votes")
                    
                    # Create a more descriptive table of vote changes
                    change_data = []
                    for voter, (old_vote, new_vote) in vote_changes.items():
                        change_data.append({
                            "Player": voter,
                            "Initial Vote For": old_vote,
                            "Changed Vote To": new_vote
                        })
                    
                    if change_data:
                        # Display as a styled table
                        vote_change_df = pd.DataFrame(change_data)
                        st.table(vote_change_df)
                    else:
                        st.info("No players changed their votes between these turns.")
                else:
                    # Show before and after vote tables side by side
                    st.info("No vote changes detected between these turns.")
                    
                # Show the before and after votes side by side for comparison
                st.subheader("Vote Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Turn {start_turn} Votes:")
                    start_vote_df = pd.DataFrame(list(start_vote_map.items()), 
                                                columns=["Voter", "Voted For"])
                    st.table(start_vote_df)
                
                with col2:
                    st.write(f"Turn {end_turn} Votes:")
                    end_vote_df = pd.DataFrame(list(end_vote_map.items()), 
                                              columns=["Voter", "Voted For"])
                    st.table(end_vote_df)
            else:
                st.info("Please select valid turns with voting data to compare.")
            
            # Display a heatmap of voting patterns
            st.subheader("Voting Heatmap")
            
            # Create a matrix of who voted for whom
            vote_matrix = pd.pivot_table(
                df_votes, 
                index="Voter", 
                columns="Votee", 
                values="Turn",
                aggfunc="count",
                fill_value=0
            )
            
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.colormaps.get_cmap('YlOrRd')
            im = ax.imshow(vote_matrix, cmap=cmap)
            
            # Set up axes
            ax.set_xticks(np.arange(len(vote_matrix.columns)))
            ax.set_yticks(np.arange(len(vote_matrix.index)))
            ax.set_xticklabels(vote_matrix.columns)
            ax.set_yticklabels(vote_matrix.index)
            
            # Rotate column labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add a colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label("Number of Votes")
            
            # Add values to cells
            for i in range(len(vote_matrix.index)):
                for j in range(len(vote_matrix.columns)):
                    if vote_matrix.iloc[i, j] > 0:
                        ax.text(j, i, vote_matrix.iloc[i, j], ha="center", va="center", color="black")
            
            ax.set_title("Voting Pattern Heatmap")
            ax.set_xlabel("Player Voted For")
            ax.set_ylabel("Voter")
            fig.tight_layout()
            
            # Display the heatmap
            st.pyplot(fig)
            
            # Show vote progression over time
            st.subheader("Vote Progression Over Time")
            
            # Create line chart to show vote counts over time
            vote_counts = df_votes.groupby(["Turn", "Votee"]).size().reset_index(name="Count")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for player in all_players:
                player_votes = vote_counts[vote_counts["Votee"] == player]
                if not player_votes.empty:
                    ax.plot(player_votes["Turn"], player_votes["Count"], marker='o', label=player)
            
            ax.set_xlabel("Turn")
            ax.set_ylabel("Number of Votes")
            ax.set_title("Votes Received Over Time")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set x-ticks to match turns
            ax.set_xticks(vote_turns)
            
            st.pyplot(fig)
            
            # Display chain of thought for each vote
            st.subheader("Voting Reasoning (Chain of Thought)")
            
            # Group votes by turn for easier navigation
            vote_turns_for_cot = sorted(df_votes["Turn"].unique())
            selected_turn_for_cot = st.selectbox(
                "Select Turn to View Reasoning", 
                vote_turns_for_cot,
                key="turn_for_cot"
            )
            
            # Get votes for the selected turn
            turn_votes = df_votes[df_votes["Turn"] == selected_turn_for_cot]
            
            if not turn_votes.empty:
                for _, vote in turn_votes.iterrows():
                    with st.expander(f"{vote['Voter']} voted for {vote['Votee']}"):
                        # Format the chain of thought text
                        cot_text = vote['CoT']
                        # Remove <think> and </think> tags if present
                        cot_text = cot_text.replace("<think>", "").replace("</think>", "")
                        # Display the formatted text
                        st.markdown(f"**Chain of Thought:**\n\n{cot_text}")
            else:
                st.info("No voting data available for this turn.")
            
            # Show raw voting data for reference
            with st.expander("View Raw Voting Data"):
                st.dataframe(df_votes)
    
    with tabs[5]:  # Raw Data
        st.header("Raw Game Data")
        
        # Allow exploration of raw history data
        st.subheader("History Entries")
        
        # Select history entry
        history_index = st.slider("Select History Entry", 0, len(history) - 1, len(history) - 1)
        selected_entry = history[history_index]
        
        # Convert to dictionary for easier display
        entry_dict = {}
        for attr in dir(selected_entry):
            if not attr.startswith('_') and not callable(getattr(selected_entry, attr)):
                value = getattr(selected_entry, attr)
                # Special handling for votes_before_this_discussion_message
                if attr == 'votes_before_this_discussion_message' and value:
                    # Don't convert to string, we'll display it separately
                    entry_dict[attr] = "See formatted votes below"
                else:
                    entry_dict[attr] = str(value)
        
        # Display as JSON
        st.json(entry_dict)
        
        # Display votes in a more readable format if they exist
        if hasattr(selected_entry, 'votes_before_this_discussion_message') and selected_entry.votes_before_this_discussion_message:
            st.subheader("Votes Before Discussion")
            
            # Create a more readable format for the votes
            for voter, vote_info in selected_entry.votes_before_this_discussion_message.items():
                with st.expander(f"{voter} voted for {vote_info.get('voted_player', 'unknown')}"):
                    if 'chain_of_thought' in vote_info:
                        # Format the chain of thought text
                        cot_text = vote_info['chain_of_thought']
                        # Remove <think> and </think> tags if present
                        cot_text = cot_text.replace("<think>", "").replace("</think>", "")
                        # Display the formatted text
                        st.markdown(f"**Chain of Thought:**\n\n{cot_text}")
                    else:
                        st.write("No reasoning provided")
        
        # Display players
        st.subheader("Players")
        
        player_data = []
        for player in players:
            player_dict = {
                "Name": player.name,
                "Role": player.role.value if hasattr(player.role, 'value') else str(player.role)
            }
            player_data.append(player_dict)
        
        st.dataframe(pd.DataFrame(player_data))

except Exception as e:
    st.error(f"Error loading game state: {e}")
    # import traceback
    # traceback.print_exc()
    st.code(str(e))
