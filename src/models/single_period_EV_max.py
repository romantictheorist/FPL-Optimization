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
import time

sys.path.append("..")

pd.options.mode.chained_assignment = None  # default='warn'


# ----------------------------------------
# Define function to get data for model
# ----------------------------------------


def get_model_data():
    """
    Returns a dictionary of slim dataframes for the current gameweek.
    """

    # Run make_dataset.py to get latest data from FPL API
    import data.make_dataset

    # Run build_features.py to create slim dataframes (clean and processed data) from latest data
    import features.build_features

    # Define path to processed data
    processed_path = "../../data/processed/"

    # Get gameweek number of newest data, i.e. current gameweek
    current_gw = max(
        [
            int(folder.split("_")[1])
            for folder in os.listdir(processed_path)
            if os.path.isdir(processed_path + folder)
        ]
    )

    # Define processed path for current gameweek
    current_gw_processed_path = "../../data/processed/gw_" + str(current_gw)

    # Load data and set IDs as index
    slim_elements_df = pd.read_csv(current_gw_processed_path + "/slim_elements.csv")
    slim_elements_df.set_index("id", inplace=True)  # Set index to player id

    slim_element_types_df = pd.read_csv(
        current_gw_processed_path + "/slim_element_types.csv"
    )
    slim_element_types_df.set_index("id", inplace=True)  # Set index to element type id

    slim_teams_df = pd.read_csv(current_gw_processed_path + "/slim_teams.csv")
    slim_teams_df.set_index("id", inplace=True)  # Set index to team id

    return {
        "slim_elements_df": slim_elements_df,
        "slim_element_types_df": slim_element_types_df,
        "slim_teams_df": slim_teams_df,
        "current_gw": current_gw,
    }


# ----------------------------------------
# Define function to solve single period model
# ----------------------------------------


def solve_single_period_model(budget):
    # ----------------------------------------
    # Get data for current gameweek
    # ----------------------------------------

    model_data = get_model_data()

    # Get slim dataframes
    slim_elements_df = model_data["slim_elements_df"]
    slim_element_types_df = model_data["slim_element_types_df"]
    slim_teams_df = model_data["slim_teams_df"]

    # Get current gameweek
    gw = model_data["current_gw"]

    # Get list of player IDs, element type IDs and team IDs (for use in model)
    players = list(slim_elements_df.index)
    positions = list(slim_element_types_df.index)
    teams = list(slim_teams_df.index)

    # ----------------------------------------
    # Initialise model
    # ----------------------------------------

    model = LpProblem("SinglePeriod", sense=LpMaximize)

    # ----------------------------------------
    # Define decision variables
    # ----------------------------------------

    squad = LpVariable.dicts("squad", players, cat="Binary")
    lineup = LpVariable.dicts("lineup", players, cat="Binary")
    captain = LpVariable.dicts("captain", players, cat="Binary")
    vice_captain = LpVariable.dicts("vice_captain", players, cat="Binary")

    # ----------------------------------------
    # Define objective function
    # ----------------------------------------

    # Maximize expected points, captain gets 2x and vice captain gets 1.1x)
    total_ep_next = lpSum(
        [
            slim_elements_df.loc[p, "ep_next"]
            * (lineup[p] + captain[p] + 0.1 * vice_captain[p])
            for p in players
        ]
    )

    model += total_ep_next

    # ----------------------------------------
    # Define constraints
    # ----------------------------------------

    ## Total number of players in squad must be 15
    model += lpSum([squad[p] for p in players]) == 15

    ## Total number of players in lineup must be 11
    model += lpSum([lineup[p] for p in players]) == 11

    ## Only 1 captain
    model += lpSum([captain[p] for p in players]) == 1

    ## Only 1 vice captain
    model += lpSum([vice_captain[p] for p in players]) == 1

    ## Lineup player must be in squad (but reverse can not be true)
    for p in players:
        model += lineup[p] <= squad[p]

    ## Captain must be in lineup
    for p in players:
        model += captain[p] <= lineup[p]

    ## Vice captain must be in lineup
    for p in players:
        model += vice_captain[p] <= lineup[p]

    ## Captain and vice captain can not be the same player
    for p in players:
        model += captain[p] + vice_captain[p] <= 1

    ## Dictionary that counts the number of players in each position in lineup
    lineup_position_count = {
        pos: lpSum(
            [
                lineup[p]
                for p in players
                if slim_elements_df.loc[p, "element_type"] == pos
            ]
        )
        for pos in positions
    }

    ## Dictionary that counts the number of players in each position in squad
    squad_position_count = {
        pos: lpSum(
            [
                squad[p]
                for p in players
                if slim_elements_df.loc[p, "element_type"] == pos
            ]
        )
        for pos in positions
    }

    ## Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play)
    for pos in positions:
        # Minimum number of players in lineup
        model += (
            lineup_position_count[pos]
            >= slim_element_types_df.loc[pos, "squad_min_play"]
        )
        # Maximum number of players in lineup
        model += (
            lineup_position_count[pos]
            <= slim_element_types_df.loc[pos, "squad_max_play"]
        )

    ## Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select)
    for pos in positions:
        model += (
            squad_position_count[pos] == slim_element_types_df.loc[pos, "squad_select"]
        )

    ## Total cost of entire squad must be less than or equal to budget
    budget = budget
    squad_cost = lpSum(
        [slim_elements_df.loc[p, "now_cost"] * squad[p] for p in players]
    )
    model += squad_cost <= budget

    ## Dictionary that counts the number of players in each team in squad
    squad_team_count = {
        team: lpSum(
            [squad[p] for p in players if slim_elements_df.loc[p, "team"] == team]
        )
        for team in teams
    }

    ## Number of players in each team in squad must be less than or equal to 3
    for team in teams:
        model += squad_team_count[team] <= 3

    ## Every lineup player must play at have at least 75% chance of playing next gameweek
    for p in players:
        model += squad[p] <= (
            slim_elements_df.loc[p, "chance_of_playing_next_round"] >= 75
        )

    # ----------------------------------------
    # Solve model and get results
    # ----------------------------------------

    # Solve model using CBC solver and surpress output
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # If model was solved to optimality, save and get results
    if model.status == 1:
        print("Model solved successfully.")
        print(
            "Results exported to models/single_period/solved_EV_max_gw_"
            + str(gw)
            + "_budget_"
            + str(budget)
        )
        print("-" * 40)
        print("Status:", LpStatus[model.status])
        print("Current gameweek:", gw)
        print("Budget:", budget)
        print("Expected points:", round(value(model.objective), 1))
        print("Cost:", round(value(squad_cost), 1))
        print("-" * 40)

        # Export model to MPS file with current gameweek and budget in name
        model.writeMPS(
            "../../models/single_period/solved_EV_max_gw_"
            + str(gw)
            + "_budget_"
            + str(budget)
            + ".mps"
        )

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
                    player_data["chance_of_playing_next_round"],
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
                "chance_of_playing_next_round",
                "in_lineup",
                "is_captain",
                "is_vice_captain",
            ],
        )

        # Sort results by in_lineup, element_type, ep_next and now_cost
        results_df.sort_values(
            by=["in_lineup", "element_type", "ep_next", "now_cost"],
            ascending=[False, True, False, True],
            inplace=True,
        )

        # Export results to csv file
        results_df.to_csv(
            "../../models/single_period/solved_EV_max_gw_"
            + str(gw)
            + "_budget_"
            + str(budget)
            + ".csv",
            index=False,
        )

        return {
            "model": model,
            "results": results_df,
            "squad:": squad,
            "lineup": lineup,
            "captain": captain,
            "vice_captain": vice_captain,
            "total_ep_next": round(value(model.objective), 1),
            "squad_cost": squad_cost,
        }

    else:
        print("Failed to solve problem.")
        print("Status:", LpStatus[model.status])
        results_df = None


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    results = []

    for budget in range(650, 1001, 25):
        r = solve_single_period_model(budget=budget)["total_ep_next"]
        results.append([budget, r])

    results_df = pd.DataFrame(results, columns=["budget", "total_ep_next"])
    print(results_df)
    print("Time spent in loop:", round(time.time() - t0, 1), "seconds")

# ----------------------------------------

df = pd.read_csv("../../models/single_period/solved_EV_max_gw_21_budget_1000.csv")
df
