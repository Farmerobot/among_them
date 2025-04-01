import streamlit as st
import jsonpickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import os
from collections import defaultdict
import time
from datetime import datetime
import json

from among_them.models.location import ROOM_COORDINATES
from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.consts import STATE_FILE
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
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 30, 5)

last_modified_time = os.path.getmtime(file_path)
last_modified_str = datetime.fromtimestamp(last_modified_time).strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.write(f"Last Modified: {last_modified_str}")

# Load game state
@st.cache_data(ttl=refresh_interval)
def load_game_state(file_path, last_modified_time):
    with open(file_path, 'rb') as f:
        state_str = f.read().decode('utf-8')
        history, players = jsonpickle.decode(state_str)
    return history, players, last_modified_time

try:
    history, players, _ = load_game_state(file_path, last_modified_time)
    
    # Check if auto-refresh is enabled
    if auto_refresh:
        st.sidebar.write("Auto-refreshing...")
        time.sleep(0.1)  # Small delay to prevent excessive refreshing
        st.rerun()
        
    # Main tabs
    tabs = st.tabs(["Game Map", "Timeline", "Tasks", "Player Network", "Raw Data"])

    with tabs[0]:  # Game Map
        st.header("Game Map Visualization")
        
        # Create a figure for the map
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the map (rooms and connections)
        for room_name, (x, y) in ROOM_COORDINATES.items():
            circle = plt.Circle((x, y), 0.15, fill=True, alpha=0.2, color='lightgray', edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, y, room_name, ha='center', va='center', fontsize=8)
        
        # Draw connections
        from among_them.models.location import DOORS
        for room_name, connected_rooms in DOORS.items():
            room_x, room_y = ROOM_COORDINATES[room_name]
            for connected_room in connected_rooms:
                conn_x, conn_y = ROOM_COORDINATES[connected_room]
                ax.plot([room_x, conn_x], [room_y, conn_y], 'k-', alpha=0.3, linewidth=1)
        
        # Get the latest history entry
        latest_entry = history[-1]
        
        # Track player positions
        player_positions = {}
        
        # Go through history to find player last known locations
        for entry in history:
            player_name = entry.acted_by_player
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
            if entry.action_type == ActionType.KILL and entry.killed_or_reported_player_name:
                dead_players.add(entry.killed_or_reported_player_name)
        
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
                x, y = ROOM_COORDINATES[location_name]
                
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
        cols[0].write(f"Player: {latest_entry.acted_by_player}")
        cols[1].write(f"Action: {latest_entry.action_type}")
        cols[2].write(f"Location: {latest_entry.location}")
        
        if latest_entry.action_result_spectator_sees:
            st.write(f"Result: {latest_entry.action_result_spectator_sees}")

    with tabs[1]:  # Timeline
        st.header("Game Timeline")
        
        # Create a dataframe of history entries
        timeline_data = []
        
        for i, entry in enumerate(history):
            timeline_data.append({
                "Turn": i + 1,
                "Player": entry.acted_by_player,
                "Phase": entry.phase,
                "Action": entry.action_type,
                "Location": entry.location,
                "Result": entry.action_result_spectator_sees
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
                    task_complete = 0
                    task_total = len(tasks)
                    
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
        interactions = defaultdict(int)
        
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
        st.pyplot(plt)
        
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
            if entry.action_type == ActionType.SPEAK:
                speaker = entry.acted_by_player
                message = entry.action_result_spectator_sees
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

    with tabs[4]:  # Raw Data
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
                entry_dict[attr] = str(value)
        
        # Display as JSON
        st.json(entry_dict)
        
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
    st.code(str(e))
