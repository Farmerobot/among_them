import random
import uuid
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
from streamlit.delta_generator import DeltaGenerator
from annotated_text import annotated_text
import plotly.graph_objects as go
from llm_postor.game.game_state import GameState
from llm_postor.game.game_engine import GameEngine
from llm_postor.game.players.base_player import Player, PlayerRole
from llm_postor.game.models.history import PlayerState
from llm_postor.game.models.engine import ROOM_COORDINATES
import plotly.graph_objects as go
from llm_postor.game.chat_analyzer import ChatAnalyzer


class GUIHandler(BaseModel):

    def display_gui(self, game_engine: GameEngine, chat_analyzer: ChatAnalyzer):
        st.title("Among Us Game - LLMPostor")
        with st.sidebar:
            for i, player in enumerate(game_engine.state.players):
                with st.empty():
                    self._display_short_player_info(
                        player, i == game_engine.state.player_to_act_next, st
                    )


        # Create a button to trigger the next step
        if st.button("Make Step"):
            game_engine.perform_step()
            st.rerun()
        self._display_map(game_engine.state)
        st.json(game_engine.state.get_total_cost())
        st.text("\n".join(game_engine.state.playthrough))
        self._display_annotated_text(game_engine.state)
        st.json(game_engine.state.to_dict())

        # Analyze Chat Button
        if st.button("Analyze Chat"):
            results = chat_analyzer.analyze()
            st.write(results)

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

    def _display_player_info(self, player: Player, placeholder: DeltaGenerator):
        with placeholder.container():  # Clear previous content
            st.subheader(player.name)
            self._display_status(player)
            self._display_role(player)
            self._display_tasks_progress(player)
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
                f"{status_icon} {player.name} - ({complete_tasks}/{len(player.state.tasks)}) {role_icon} ⏳{player.kill_cooldown} {current_icon}"
            )
        else:
            st.write(
                f"{status_icon} {player.name} - ({complete_tasks}/{len(player.state.tasks)}) {role_icon} {current_icon}"
            )

    def _display_status(self, player: Player):
        status_icon = "✅" if player.state.life == PlayerState.ALIVE else "❌"
        st.write(f"Status: {status_icon} {player.state.life.value}")

    def _display_role(self, player: PlayerRole):
        role_icon = "😈" if player.role == PlayerRole.IMPOSTOR else "👤"
        st.write(f"Role: {role_icon} {player.role.value}")

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
        img_width = 836 * 2
        img_height = 470 * 2
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
                        x=[x * 200 + random.randint(-20, 20)],
                        y=[y * 200 + random.randint(-20, 20)],
                        mode="markers",
                        marker=dict(
                            color=marker_color,
                            size=marker_size,
                            symbol=marker_symbol,
                        ),
                        name=player.name,
                        customdata=[
                            f"<b>{player.name}</b><br>Role: {player.role.value}<br>Status: {player.state.life.value}"
                        ],
                        hovertemplate="%{customdata}",
                    )
                )

        update_player_markers(game_state)

        # Display the map
        map_placeholder = st.empty()
        map_placeholder.plotly_chart(fig, use_container_width=True, key=uuid.uuid4())

    def _display_annotated_text(self, game_state: GameState):
        annotated_text(
            "[Warek]: I agree that ",
            (
                "Wateusz's consistent activity in the Cafeteria raises suspicion",
                "Behavior Analysis",
            ),
            ", while ",
            (
                "Waciej and Wafał's movement to Medbay should be monitored closely",
                "Behavior Analysis",
            ),
            ".",
            "[Wojtek]: I saw ",
            ("Wafał eliminate Waciej in Medbay", "Accusation and Finger-Pointing"),
            ", which clearly makes him the impostor we need to vote out.",
            "[Wafał]: I was in Medbay with both Waciej and Wojtek, and I can confirm ",
            ("I did not eliminate anyone", "Alibi Construction"),
            "; we should focus on ",
            ("Wateusz's suspicious activity instead", "Scapegoating"),
            ".",
            "[Wateusz]: I was in the Cafeteria completing tasks and can confirm ",
            ("I did not eliminate Waciej", "Alibi Construction"),
            "; we should focus on verifying ",
            ("Wojtek's claim against Wafał", "Information Manipulation"),
            ".",
            "[Warek]: I suggest we closely examine ",
            ("Wojtek's claim against Wafał", "Information Manipulation"),
            " and gather more details about ",
            ("Wateusz's activities in the Cafeteria", "Behavior Analysis"),
            " before making any decisions.",
            "[Wojtek]: I stand by my claim that ",
            (
                "Wafał is the impostor for eliminating Waciej",
                "Accusation and Finger-Pointing",
            ),
            ", and we need to vote him out immediately.",
            "[Wafał]: I was in Medbay with both Waciej and Wojtek, and I did not eliminate anyone; let's focus on ",
            (
                "Wateusz's suspicious activities in the Cafeteria instead",
                "Scapegoating",
            ),
            ".",
            "[Wateusz]: I was in the Cafeteria completing tasks and did not eliminate Waciej; we need to clarify ",
            ("Wojtek's claim about Wafał before voting", "Information Manipulation"),
            ".",
            "[Warek]: I believe we should focus on verifying ",
            ("Wojtek's claim about Wafał", "Information Manipulation"),
            " before making any voting decisions.",
            "[Wojtek]: I urge everyone to vote out ",
            ("Wafał immediately based on my eyewitness account", "Confidence Display"),
            " of him eliminating Waciej in Medbay.",
            "[Wafał]: I was in Medbay with both Waciej and Wojtek, and I did not eliminate anyone; we should focus on verifying ",
            ("Wateusz's suspicious activities instead", "Scapegoating"),
            ".",
            "[Wateusz]: I agree that we need to verify ",
            (
                "Wojtek's claim about Wafał before making any voting decisions",
                "Information Manipulation",
            ),
            ".",
            "[Warek]: I propose we ask ",
            (
                "Wojtek for specific details about the elimination",
                "Information Manipulation",
            ),
            " before making any voting decisions.",
            "[Wojtek]: I stand by my eyewitness account of ",
            ("Wafał eliminating Waciej", "Accusation and Finger-Pointing"),
            " and urge everyone to vote him out immediately.",
            "[Wafał]: I was in Medbay with Waciej and Wojtek, and I didn't eliminate anyone; we should focus on verifying ",
            ("Wateusz's tasks instead", "Scapegoating"),
            ".",
            "[Wateusz]: I agree we need to clarify ",
            (
                "Wojtek's claim about Wafał before making any voting decisions",
                "Information Manipulation",
            ),
            ".",
            "[Warek]: ",
            (
                "Wojtek, please provide specific details about the elimination you witnessed",
                "Information Manipulation",
            ),
            " to clarify your claim against Wafał.",
            "[Wojtek]: I clearly saw ",
            ("Wafał eliminate Waciej in Medbay", "Accusation and Finger-Pointing"),
            ", and we need to vote him out immediately.",
            "[Wafał]: I agree with investigating ",
            ("Wateusz further", "Scapegoating"),
            ", but let's also press ",
            (
                "Wojtek for details on his accusation against me",
                "Information Manipulation",
            ),
            " to clarify the situation.",
        )
