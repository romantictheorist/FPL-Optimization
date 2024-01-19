# ----------------------------------------
# Import packages and set working directory
# ----------------------------------------

import sys

import numpy as np
import pandas as pd
import requests
import json
import pulp
from pulp import *


# ----------------------------------------
# Load and process data
# ----------------------------------------

slim_elements_df = pd.read_csv("../../data/processed/slim_elements.csv")
slim_elements_df.set_index("id", inplace=True)  # Set index to player id

element_types_df = pd.read_csv("../../data/raw/element_types.csv")
element_types_df.set_index("id", inplace=True)  # Set index to element type id

teams_df = pd.read_csv("../../data/raw/teams.csv")
teams_df.set_index("id", inplace=True)  # Set index to team id

# ----------------------------------------
# List of player IDs, element type  (position) IDs and team IDs for indexing
# ----------------------------------------

players = slim_elements_df.index.to_list()
positions = element_types_df.index.to_list()
teams = teams_df.index.to_list()

# ----------------------------------------
# Initialising the model
# ----------------------------------------

model = LpProblem("SinglePeriod", sense=LpMaximize)

# ----------------------------------------
# Defining the decision variables
# ----------------------------------------

squad = LpVariable.dicts("squad", players, cat="Binary")
lineup = LpVariable.dicts("lineup", players, cat="Binary")
captain = LpVariable.dicts("captain", players, cat="Binary")
vice_captain = LpVariable.dicts("vice_captain", players, cat="Binary")

# ----------------------------------------
# Defining the objective function (maximize expected points, captain gets 2x and vice captain gets 1.5x)
# ----------------------------------------

total_ep_next = lpSum(
    [
        slim_elements_df.loc[p, "ep_next"]
        * (lineup[p] + captain[p] + 0.1 * vice_captain[p])
        for p in players
    ]
)

model += total_ep_next

# ----------------------------------------
# Defining constraints
# ----------------------------------------

# Total number of players in squad must be 15
model += lpSum([squad[p] for p in players]) == 15

# Total number of players in lineup must be 11
model += lpSum([lineup[p] for p in players]) == 11

# Only 1 captain
model += lpSum([captain[p] for p in players]) == 1

# Only 1 vice captain
model += lpSum([vice_captain[p] for p in players]) == 1

# Lineup player must be in squad (but reverse can not be true)
for p in players:
    model += lineup[p] <= squad[p]

# Captain must be in lineup
for p in players:
    model += captain[p] <= lineup[p]

# Vice captain must be in lineup
for p in players:
    model += vice_captain[p] <= lineup[p]

# Captain and vice captain can not be the same player
for p in players:
    model += captain[p] + vice_captain[p] <= 1

# Dictionary that counts the number of players in each position in lineup
lineup_position_count = {
    pos: lpSum(
        [lineup[p] for p in players if slim_elements_df.loc[p, "element_type"] == pos]
    )
    for pos in positions
}

# Dictionary that counts the number of players in each position in squad
squad_position_count = {
    pos: lpSum(
        [squad[p] for p in players if slim_elements_df.loc[p, "element_type"] == pos]
    )
    for pos in positions
}

# Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play)
for pos in positions:
    # Minimum number of players in lineup
    model += lineup_position_count[pos] >= element_types_df.loc[pos, "squad_min_play"]
    # Maximum number of players in lineup
    model += lineup_position_count[pos] <= element_types_df.loc[pos, "squad_max_play"]


# Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select)
for pos in positions:
    model += squad_position_count[pos] == element_types_df.loc[pos, "squad_select"]


# Total cost of entire squad must be less than or equal to budget
budget = 1000
squad_cost = lpSum([slim_elements_df.loc[p, "now_cost"] * squad[p] for p in players])
model += squad_cost <= budget

# Dictionary that counts the number of players in each team in squad
squad_team_count = {
    team: lpSum([squad[p] for p in players if slim_elements_df.loc[p, "team"] == team])
    for team in teams
}

# Number of players in each team in squad must be less than or equal to 3
for team in teams:
    model += squad_team_count[team] <= 3

# ----------------------------------------
# Solve the model and print results
# ----------------------------------------

model.solve(PULP_CBC_CMD(msg=0))

print("-" * 40)

if model.status == 1:
    print("Status:", LpStatus[model.status])

    squad = []
    lineup = []

    # Get player IDs of squad and lineup
    for v in model.variables():
        if v.varValue == 1:
            if v.name[0] == "s":
                squad.append(int(v.name[6:]))
            elif v.name[0] == "l":
                lineup.append(int(v.name[7:]))

    print("Lineup:", lineup)
    print("Squad:", squad)

    # Get expected points of lineup and bench
    print("Total expected points:", value(model.objective))

    # Get the total cost of the squad
    total_cost = 0
    for p in squad:
        total_cost += slim_elements_df.loc[p, "now_cost"]

    print("Total cost:", total_cost)
    print("-" * 40)

    # Get data for entire squad using the IDs and sort by position
    squad_df = slim_elements_df.loc[squad].sort_values(by="element_type")

    # Get data for lineup using the IDs
    lineup_df = slim_elements_df.loc[lineup].sort_values(by="element_type")

else:
    print("Failed to solve problem.")
    print("Status:", LpStatus[model.status])

# ----------------------------------------
# Export model to MPS
# ----------------------------------------


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# model.writeMPS("../../models/single_period_EV_max.mps")
# model.to_json("../../models/single_period_EV_max.json", cls=NpEncoder)

# ----------------------------------------
# Import model back from json
# ----------------------------------------
