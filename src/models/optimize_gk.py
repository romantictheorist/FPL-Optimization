# ----------------------------------------
# Import packages and set working directory
# ----------------------------------------

import sys

import numpy as np
import pandas as pd
import pulp
from pulp import *

sys.path.append("..")

# ----------------------------------------
# Load and process data
# ----------------------------------------

# Read in slim elements data
slim_elements_df = pd.read_csv("../../data/processed/slim_elements.csv")

# Select only goalkeepers
goalkeepers_df = (
    slim_elements_df[slim_elements_df.position == "GKP"].copy().reset_index(drop=True)
)

# Set index to player id
goalkeepers_df.set_index("id", inplace=True)

# ----------------------------------------
# Define a function to optimize goalkeepers
# ----------------------------------------


def optimize_goalkeeper(goalkeepers_df, budget):
    # ----------------------------------------
    # List of player IDs (for indexing)
    # ----------------------------------------

    goalkeepers_ids = goalkeepers_df.index.to_list()

    # ----------------------------------------
    # Initialising the model
    # ----------------------------------------

    model = LpProblem("SelectGoalkeepers", sense=LpMaximize)

    # ----------------------------------------
    # Defining the decision variables
    # ----------------------------------------

    lineup = LpVariable.dicts("lineup", goalkeepers_ids, cat="Binary")
    bench = LpVariable.dicts("bench", goalkeepers_ids, cat="Binary")

    # ----------------------------------------
    # Defining the objective function
    # ----------------------------------------

    # Maximize the total ep_next
    model += lpSum(
        [goalkeepers_df.loc[i, "ep_next"] * lineup[i] for i in goalkeepers_ids]
    ) + 0.1 * lpSum(
        [goalkeepers_df.loc[i, "ep_next"] * bench[i] for i in goalkeepers_ids]
    )

    # ----------------------------------------
    # Defining the constraints
    # ----------------------------------------

    # Constraint 1: GK can only be in lineup or bench
    for i in goalkeepers_ids:
        model += lineup[i] + bench[i] <= 1

    # Constraint 2: 1 GK in lineup
    model += lpSum([lineup[i] for i in goalkeepers_ids]) == 1

    # Constraint 3: 1 GK in bench
    model += lpSum([bench[i] for i in goalkeepers_ids]) == 1

    # Constraint 4: Total cost of lineup and bench must be less than or equal to budget
    model += (
        lpSum(
            (lineup[i] + bench[i]) * goalkeepers_df.loc[i, "now_cost"]
            for i in goalkeepers_ids
        )
    ) <= budget

    # ----------------------------------------
    # Solving the model and printing solution
    # ----------------------------------------

    # solve without printing messages
    model.solve(PULP_CBC_CMD(msg=0))

    if model.status == 1:
        print("-" * 40)
        print("Status:", LpStatus[model.status])

        lineup_players = []
        bench_players = []

        # Get IDs of lineup and bench players
        for v in model.variables():
            if v.varValue == 1:
                if v.name[0] == "l":
                    lineup_players.append(int(v.name[7:]))
                elif v.name[0] == "b":
                    bench_players.append(int(v.name[6:]))

        print("Lineup:", lineup_players)
        print("Bench:", bench_players)

        print("Total expected points:", value(model.objective))

        # Get the total cost of the lineup and bench
        total_cost = 0
        for i in lineup_players:
            total_cost += goalkeepers_df.loc[i, "now_cost"]
        for i in bench_players:
            total_cost += goalkeepers_df.loc[i, "now_cost"]
        print("Total cost:", total_cost)
        print("-" * 40)

    else:
        print("Failed to solve problem.")
        print("Status:", LpStatus[model.status])


# ----------------------------------------
# Testing the function
# ----------------------------------------

test = optimize_goalkeeper(goalkeepers_df, 1000)
