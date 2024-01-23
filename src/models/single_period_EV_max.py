# ----------------------------------------
# Import packages and set working directory
# ----------------------------------------

import sys

import numpy as np
import pandas as pd
import pulp
from pulp import *
import time
import multiprocessing

import sys

sys.path.append("..")

from data.pull_data import pull_general_data

pd.options.mode.chained_assignment = None  # default='warn'


# ----------------------------------------
# Define function to merge fpl_form_data with elements_df
# ----------------------------------------


def merge_fpl_form_data(elements_df, fpl_form_data):
    """
    Merge fpl_form_data with elements_df.
    """

    # Replace GK with GKP to match elements_df
    fpl_form_data["Pos"] = fpl_form_data["Pos"].replace("GK", "GKP")

    # Merge fpl_form_data with elements_df
    merged_elements_df = pd.merge(
        elements_df,
        fpl_form_data,
        left_on=[
            "id",
            "web_name",
            "position",
        ],
        right_on=["ID", "Name", "Pos"],
    )

    # Drop duplicate columns
    merged_elements_df.drop(
        columns=["ID", "Name", "Pos", "Team", "Price"],
        inplace=True,
    )

    # If merge was successful, return merged_elements_df
    if len(merged_elements_df) == len(elements_df):
        print("FPL form data successfully merged with elements_df.")
        return merged_elements_df
    else:
        print("Merge was unsuccessful.")
        return None


# ----------------------------------------
# Define function to solve single period model
# ----------------------------------------


def solve_single_period_model(budget):
    print("-" * 80)

    # ----------------------------------------
    # Pull data from FPL API
    # ----------------------------------------

    general_data = pull_general_data()
    elements_df = general_data["elements"]
    element_types_df = general_data["element_types"]
    teams_df = general_data["teams"]
    current_gw = general_data["current_gw"]
    next_gw = general_data["next_gw"]

    # ----------------------------------------
    # Read in fpl_form_data and merge with elements_df
    # ----------------------------------------

    # Read in fpl_form_data
    fpl_form_data = pd.read_csv("../../data/raw/fpl-form-predicted-points.csv")

    # Merge fpl_form_data with elements_df
    merged_elements_df = merge_fpl_form_data(elements_df, fpl_form_data)

    # ----------------------------------------
    # Set index to 'id's
    # ----------------------------------------

    merged_elements_df.set_index("id", inplace=True)
    element_types_df.set_index("id", inplace=True)
    teams_df.set_index("id", inplace=True)

    # ----------------------------------------
    # List of player IDs, positions and teams (to be used in model)
    # ----------------------------------------

    players = list(merged_elements_df.index)
    positions = list(element_types_df.index)
    teams = list(teams_df.index)

    # ----------------------------------------
    # Initialise model
    # ----------------------------------------

    model = LpProblem("SinglePeriod", sense=LpMaximize)

    # ----------------------------------------
    # Define decision variables
    # ----------------------------------------

    squad = LpVariable("squad", players, cat="Binary")
    lineup = LpVariable("lineup", players, cat="Binary")
    captain = LpVariable("captain", players, cat="Binary")
    vice_captain = LpVariable("vice_captain", players, cat="Binary")

    # ----------------------------------------
    # Define objective function: Maximize expected points in next gameweek
    # ----------------------------------------

    # Maximize {next_gw}_pts_no_prob: captain gets 2x and vice captain gets 1.1x
    total_ep_next = lpSum(
        [
            float(merged_elements_df.loc[p, f"{next_gw}_pts_no_prob"])
            * (lineup[p] + captain[p] + 0.1 * vice_captain[p])
            for p in players
        ]
    )

    model += total_ep_next

    # ----------------------------------------
    # Define constraints
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
            [
                lineup[p]
                for p in players
                if merged_elements_df.loc[p, "element_type"] == pos
            ]
        )
        for pos in positions
    }

    # Dictionary that counts the number of players in each position in squad
    squad_position_count = {
        pos: lpSum(
            [
                squad[p]
                for p in players
                if merged_elements_df.loc[p, "element_type"] == pos
            ]
        )
        for pos in positions
    }

    # Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play)
    for pos in positions:
        # Minimum number of players in lineup
        model += (
            lineup_position_count[pos] >= element_types_df.loc[pos, "squad_min_play"]
        )
        # Maximum number of players in lineup
        model += (
            lineup_position_count[pos] <= element_types_df.loc[pos, "squad_max_play"]
        )

    # Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select)
    for pos in positions:
        model += squad_position_count[pos] == element_types_df.loc[pos, "squad_select"]

    # Total cost of entire squad must be less than or equal to budget
    squad_cost = lpSum(
        [merged_elements_df.loc[p, "now_cost"] * squad[p] for p in players]
    )
    model += squad_cost <= budget

    # Dictionary that counts the number of players in each team in squad
    squad_team_count = {
        team: lpSum(
            [squad[p] for p in players if merged_elements_df.loc[p, "team"] == team]
        )
        for team in teams
    }

    # Number of players in each team in squad must be less than or equal to 3
    for team in teams:
        model += squad_team_count[team] <= 3

    # Probability of squad player appearing in next gameweek must be greater than or equal to 50%,
    # while probability of lineup player appearing in next gameweek must be greater than or equal to 75%
    for p in players:
        model += squad[p] <= (merged_elements_df.loc[p, f"{next_gw}_prob"] >= 0.5)
        model += lineup[p] <= (merged_elements_df.loc[p, f"{next_gw}_prob"] >= 0.75)

    # ----------------------------------------
    # Solve model and get results
    # ----------------------------------------

    # Solve model using CBC solver and surpress output
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # If model was solved to optimality, save and get results
    if model.status == 1:
        model_name = f"solved_EV_max_gw_{current_gw}_budget_{budget}"
        model_path = "../../models/single_period/"

        # Check if model/single_period folder exists, if not create it
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            pass

        # Write model to .mps file
        model.writeMPS(model_path + model_name + ".mps")

        # Empty lists for each variable
        squad = []
        lineup = []
        captain = []
        vice_captain = []

        # Loop through model variables and get variables that are 1 (i.e. selected)
        for v in model.variables():
            if v.varValue == 1:
                # If variable is a squad player, add to squad list
                if v.name[0] == "s":
                    id = int(v.name[6:])
                    squad.append(id)

                # If variable is a lineup player, add to lineup list
                elif v.name[0] == "l":
                    id = int(v.name[7:])
                    lineup.append(id)

                # If variable is a captain, add to captain list
                elif v.name[0] == "c":
                    id = int(v.name[8:])
                    captain.append(id)

                # If variable is a vice captain, add to vice captain list
                elif v.name[0] == "v":
                    id = int(v.name[13:])
                    vice_captain.append(id)

        # Get total expected points
        total_ep_next = round(value(model.objective), 1)

        # Get total squad cost
        squad_cost = round(value(squad_cost), 1)

        # Create a results dataframe with only the selected players
        results = []

        for p in squad:
            player_data = merged_elements_df.loc[p]
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
                    player_data["now_cost"],
                    player_data["form"],
                    player_data[f"{next_gw}_prob"],
                    player_data[f"{next_gw}_pts_no_prob"],
                    in_lineup,
                    is_captain,
                    is_vice_captain,
                ]
            )

        results_df = pd.DataFrame(
            results,
            columns=[
                "id",
                "web_name",
                "position",
                "element_type",
                "team",
                "now_cost",
                "form",
                f"gw_{next_gw}_prob",
                f"gw_{next_gw}_ep",
                "in_lineup",
                "is_captain",
                "is_vice_captain",
            ],
        )

        # Sort results by in_lineup, element_type, ep_next and now_cost
        results_df.sort_values(
            by=["in_lineup", "element_type", f"gw_{next_gw}_ep", "now_cost"],
            ascending=[False, True, False, True],
            inplace=True,
        )

        # Drop 'element_type' column
        results_df.drop(columns=["element_type"], inplace=True)

        # Reset index
        results_df.reset_index(drop=True, inplace=True)

        # Round all float columns to 2 decimal places
        results_df = results_df.round(2)

        # Export results to csv file
        results_df.to_csv(model_path + model_name + ".csv", index=False)

        # Print message to console
        print("Model status:", LpStatus[model.status])
        print(
            f"Model successfully solved for gameweek {current_gw} with a budget of {budget}."
        )
        print("Results exported to " + model_path + model_name)
        print("Expected points:", total_ep_next)
        print("Squad cost:", squad_cost)
        print("-" * 80)

        return {
            "model": model,
            "results": results_df,
            "squad:": squad,
            "lineup": lineup,
            "captain": captain,
            "vice_captain": vice_captain,
            "total_ep_next": total_ep_next,
            "squad_cost": squad_cost,
        }

    else:
        print("Failed to solve problem.")
        print("Status:", LpStatus[model.status])
        results_df = None

    print("-" * 80)


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    # Set start time
    t0 = time.time()

    # List of budgets to run model for
    budgets = list(range(650, 1001, 25))

    # ----------------------------------------
    # Loop through budgets and solve model for each budget
    # ----------------------------------------

    # results = []

    # for budget in budgets:
    #     r = solve_single_period_model(budget=budget)["total_ep_next"]
    #     results.append([budget, r])

    # # Make a dataframe with results
    # results_df = pd.DataFrame(results, columns=["budget", "total_ep_next"])

    # # Print results_df and time spent in loop
    # print(results_df)
    # print("Time spent in loop:", round(time.time() - t0, 1), "seconds")

    # ----------------------------------------
    # Parallel processing using multiprocessing
    # ----------------------------------------

    with multiprocessing.Pool() as pool:
        # Map budgets to solve_single_period_model function
        responses = pool.map(solve_single_period_model, budgets)

        # Get total_ep_next from results
        total_ep_next = [r["total_ep_next"] for r in responses]

        # Zip budgets and total_ep_next into a list of lists
        results = zip(budgets, total_ep_next)

        # Make a dataframe with total_ep_next_results
        results_df = pd.DataFrame(results, columns=["budget", "total_ep_next"])

        # Print total_ep_next_results_df
        print(results_df)

        # Print time spent parallel processing
        print("Time spent parallel processing:", round(time.time() - t0, 1), "seconds")

    # ----------------------------------------
