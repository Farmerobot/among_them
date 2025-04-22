#!/bin/bash

mlx_lm.generate \
    --model "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B" \
    --max-tokens 2000 \
    --adapter-path adapters \
    --temp 1 \
    --prompt "<purpose>
  Serve as an AI assistant for a player in a text-based social deduction game. 
  Operate strictly within defined game rules and mechanics to help the player achieve victory based on their assigned role.
</purpose>

<game_context>
  <phases>
    <phase type=\"Task\">
      <rules>
        - Movement and actions permitted  
        - No verbal communication
        - All actions taken seen by co-located players
      </rules>
    </phase>
    <phase type=\"Discussion\">
      <rules>
        - Free-form conversation enabled
        - Analyze behavior patterns
        - Strategize voting decisions
      </rules>
    </phase>
    <phase type=\"Voting\">
      <rules>
        - Majority vote determines ejection
        - Skip voting allowed
        - Game continues until victory conditions met
      </rules>
    </phase>
  </phases>

  <roles>
    <role type=\"Crewmate\">
      <objectives>
        - Complete all tasks
        - Identify impostors
        - Report corpses
      </objectives>
      <constraints>
        - Cannot kill players
        - Cannot detect impostor faking tasks
      </constraints>
    </role>
    
    <role type=\"Impostor\">
      <objectives>
        - Eliminate crewmates
        - Avoid detection
      </objectives>
      <abilities>
        - Murder co-located players
        - Fake task performance
      </abilities>
    </role>
  </roles>

  <mechanics>
    <mechanic type=\"Places\">
      - Players can move between places 
      - There is nothing else players can do in places
      - All actions taken seen by co-located players
    </mechanic>
    <mechanic type=\"Tasks\">
      - Tasks are assigned to each player and completed tasks are removed from the list
      - Tasks are independent of each other and of other players
      - All tasks must be completed to win
    </mechanic>
    <mechanic type=\"Corpse Reporting\">
      - Any player can report corpses during task phase
      - Triggers discussion phase
      - No automatic death alerts
    </mechanic>
    <mechanic type=\"Victory Conditions\">
      - Crewmates win at 100% tasks
      - Impostors win when kill all crewmates
      - Ejection affects team ratios. Crewmates win when all impostors are ejected
    </mechanic>
    <mechanic type=\"Action taking\">
      - Players can take only one action at a time
      - Only properly formatted actions will be accepted
      - Only impostors can kill crewmates
    </mechanic>
  </mechanics>
</game_context>
<player_info>\n<player_name>Charlie</player_name>\n<current_role>Crewmate</current_role>\n<current_location>Weapons</current_location>\n<impostor_count>1</impostor_count>\n<potential_impostors>\n<player>David</player>\n<player>Eve</player>\n<player>Alice</player>\n<player>Bob</player>\n</potential_impostors>\n<tasks_left>\n<task><name>Check catalyzer in lower engine</name><location>Lower Engine</location></task>\n</tasks_left>\n</player_info>\n\n<game_history>\n<action type=\"\"observed\"\">The game started</action>\n<player_thought><thought>\nOkay, so I'm Charlie, a crewmate in this game. My goal is to complete all the tasks before the impostors can either kill enough of us or get voted out. Right now, I'm in the Cafeteria with David, Eve, Alice, and Bob. The available actions are wait, move to Medbay, Admin, or Weapons.\n\nI have four tasks left: Fix wiring in storage, Clear asteroids in weapons, Check catalyzer in lower engine, and Fix wiring in weapons. Since I'm a crewmate, completing these tasks is crucial for us to win.\n\nLooking at the available actions, moving might be necessary to get closer to completing tasks or to check if there are any dead bodies that could trigger discussion. Right now, no one has died yet, so it's all about tasks and blending in without suspicion.\n\nSince I'm in the Cafeteria with others, maybe moving somewhere else where I can work on my tasks would be better. Let me see which room is best for each task. Fixing wiring in storage might be in the Storage room, but that's not an option here. Clear asteroids in weapons probably requires me to go to the Weapons room because that's where the action is. So moving to Weapons could allow me to work on two tasks: Check catalyzer and Fix wiring in weapons.\n\nAlternatively, if I stay here, I can't do any tasks. So moving makes sense. Let me choose to move to Weapons.\n</thought></player_thought>\n<action type=\"\"player_action\"\">You [Charlie] moved to location Weapons</action>\n</game_history>\n\n<current_state>\n<companions>none</companions>\n<location>Weapons</location>\n<phase>Task</phase>\n<note>You cannot speak during this phase.</note>\n</current_state>\n\n\n<available_actions>\n<action>wait</action>\n<action>move to location Medbay</action>\n<action>move to location Admin</action>\n<action>move to location Weapons</action>\n</available_actions>\n\n\nChoose one action. Respond in the following xml format: <action>action</action>"
