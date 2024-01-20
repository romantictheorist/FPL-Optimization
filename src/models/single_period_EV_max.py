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
# Load data and set IDs as index
# ----------------------------------------

gw = 21
processed_path = "../../data/processed/gw_" + str(gw)

slim_elements_df = pd.read_csv(processed_path + "/slim_elements.csv")
slim_elements_df.set_index("id", inplace=True)  # Set index to player id

slim_element_types_df = pd.read_csv(processed_path + "/slim_element_types.csv")
slim_element_types_df.set_index("id", inplace=True)  # Set index to element type id

slim_teams_df = pd.read_csv(processed_path + "/slim_teams.csv")
slim_teams_df.set_index("id", inplace=True)  # Set index to team id

# ----------------------------------------
# Create lists of player IDs, element types  (position) IDs and team IDs for indexing
# ----------------------------------------

players = slim_elements_df.index.to_list()
positions = slim_element_types_df.index.to_list()
teams = slim_teams_df.index.to_list()

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
    model += (
        lineup_position_count[pos] >= slim_element_types_df.loc[pos, "squad_min_play"]
    )
    # Maximum number of players in lineup
    model += (
        lineup_position_count[pos] <= slim_element_types_df.loc[pos, "squad_max_play"]
    )


# Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select)
for pos in positions:
    model += squad_position_count[pos] == slim_element_types_df.loc[pos, "squad_select"]


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
# Solve the model and print result
# ----------------------------------------

# Solve the model using the default solver (CBC) and surpress output
model.solve(pulp.PULP_CBC_CMD(msg=0))

# ----------------------------------------
# Store results in dataframe
# ----------------------------------------

# If model was solved to optimality, get results
if model.status == 1:
    print("-" * 40)
    print("Solved successfully.")
    print("Status:", LpStatus[model.status])
    print("Expected points:", round(value(model.objective), 1))
    print("Cost:", round(value(squad_cost), 1))
    print("-" * 40)

    # Create empty lists for each variable
    squad = []
    lineup = []
    captain = []
    vice_captain = []

    # Get model variables that are 1
    for v in model.variables():
        if v.varValue == 1:
            # If variable is a squad player
            if v.name[0] == "s":
                id = int(v.name[6:])
                squad.append(id)

            # If variable is a lineup player
            elif v.name[0] == "l":
                id = int(v.name[7:])
                lineup.append(id)

            # If variable is a captain
            elif v.name[0] == "c":
                id = int(v.name[8:])
                captain.append(id)

            # If variable is a vice captain
            elif v.name[0] == "v":
                id = int(v.name[13:])
                vice_captain.append(id)

    # Create  empty list for results
    results = []

    for p in squad:
        player_data = slim_elements_df.loc[p]
        in_lineup = 1 if p in lineup else 0
        is_captain = 1 if p in captain else 0
        is_vice_captain = 1 if p in vice_captain else 0

        results.append(
            [
                p,
                player_data["web_name"],
                player_data["position"],
                player_data["element_type"],
                player_data["team_name"],
                player_data["gw_current"],
                player_data["gw_next"],
                player_data["now_cost"],
                player_data["ep_next"],
                in_lineup,
                is_captain,
                is_vice_captain,
            ]
        )

    # Create dataframe with results
    results_df = pd.DataFrame(
        results,
        columns=[
            "id",
            "web_name",
            "position",
            "element_type",
            "team",
            "gw_current",
            "gw_next",
            "now_cost",
            "ep_next",
            "in_lineup",
            "is_captain",
            "is_vice_captain",
        ],
    )

    # Sort results by in_lineup, element_type and ep_next
    results_df.sort_values(
        by=["in_lineup", "element_type", "ep_next"],
        ascending=[False, True, True],
        inplace=True,
    )

    # Reset index
    results_df.reset_index(drop=True, inplace=True)

else:
    print("Failed to solve problem.")
    print("Status:", LpStatus[model.status])
    results_df = None

display(results_df)

# ----------------------------------------
# Export solved model to MPS file
# ----------------------------------------

# # Define path to MPS file
# model_path = "../../models/single_period/"

# # If path does not exist, create it
# if not os.path.exists(model_path):
#     os.makedirs(model_path)

# # Export model to MPS file
# model.writeMPS(model_path + "solved_EV_max_gw_" + str(gw) + ".mps")

# ----------------------------------------
