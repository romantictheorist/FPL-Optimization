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

from data.pull_data import pull_general_data, pull_squad
from features.build_features import merge_fpl_form_data

pd.options.mode.chained_assignment = None  # default='warn'

# ----------------------------------------
# Define function to solve single period model
# ----------------------------------------


def solve_single_period_model(
    team_id,
    gameweek,
    num_free_transfers,
    money_in_bank,
    horizon,
    objective,
    decay_base=0.85,
):
    print("-" * 80)

    # ----------------------------------------
    # Pull data from FPL API-
    # ----------------------------------------

    # Latest general data
    general_data = pull_general_data()
    elements_df = general_data["elements"]
    element_types_df = general_data["element_types"]
    teams_df = general_data["teams"]

    previous_gw = general_data["previous_gw"]
    current_gw = general_data["current_gw"]
    next_gw = general_data["next_gw"]

    # Initial squad (i.e. squad from gameweek prior to the one we are solving for)
    initial_squad = pull_squad(team_id=team_id, gw=gameweek - 1)

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
    # Sets
    # ----------------------------------------

    players = list(merged_elements_df.index)
    positions = list(element_types_df.index)
    teams = list(teams_df.index)
    future_gameweeks = list(range(next_gw, next_gw + horizon))
    all_gameweeks = [current_gw] + future_gameweeks

    # ----------------------------------------
    # Initialise model
    # ----------------------------------------

    model_name = f"solve_EV_max_gw_{gameweek}_horizon_{horizon}_objective_{objective}"
    model = LpProblem(model_name, sense=LpMaximize)

    # ----------------------------------------
    # Define decision variables
    # ----------------------------------------

    squad = LpVariable("squad", (players, all_gameweeks), cat="Binary")
    lineup = LpVariable("lineup", (players, future_gameweeks), cat="Binary")
    captain = LpVariable("captain", (players, future_gameweeks), cat="Binary")
    vice_captain = LpVariable("vice_captain", (players, future_gameweeks), cat="Binary")
    transfers_in = LpVariable("transfers_in", (players, future_gameweeks), cat="Binary")
    transfers_out = LpVariable(
        "transfers_in", (players, future_gameweeks), cat="Binary"
    )
    available_balance = LpVariable(
        "available_balance", (all_gameweeks), cat="Integer", lowBound=0
    )
    free_transfers_available = LpVariable(
        "free_transfers_available",
        (all_gameweeks),
        cat="Integer",
        lowBound=1,
        upBound=2,
    )
    penalised_transfers = LpVariable(
        "penalised_transfers", (future_gameweeks), cat="Integer", lowBound=0
    )
    auxiliary_var = LpVariable("auxiliary_var", (future_gameweeks), cat="Binary")

    # ----------------------------------------
    # Define objective function: Maximize expected points in next gameweek
    # ----------------------------------------

    # Captain gets 2x and vice captain gets 1.1x
    total_ep_next = lpSum(
        [
            float(merged_elements_df.loc[p, f"{next_gw}_pts_no_prob"])
            * (lineup[p] + captain[p] + 0.1 * vice_captain[p])
            for p in players
        ]
    )

    model += total_ep_next

    # ----------------------------------------
    # Create dictionaries to use for constraints
    # ----------------------------------------

    # Dictionary that stores the cost of each player
    player_cost = merged_elements_df["now_cost"].to_dict()

    # Dictionary that stores the revenue made transfers from each gameweek (i.e. amount sold)
    transfer_revenue = {
        gw: lpSum([player_cost[p] * transfers_out[p, gw] for p in players])
        for gw in future_gameweeks
    }

    # Dictionary that stores the amount spent on transfers in each gameweek (i.e. amount bought)
    transfer_spent = {
        gw: lpSum([player_cost[p] * transfers_in[p, gw] for p in players])
        for gw in future_gameweeks
    }

    # Dictionary that stores the expected points of each player in each gameweek
    # Use player ID and gameweek number as keys, and expected points as values
    player_xp_gw = {
        (p, gw): merged_elements_df.loc[p, f"{gw}_pts_no_prob"]
        for p in players
        for gw in future_gameweeks
    }

    # Dictionary of the total points scored by the lineup in each gameweek (i.e. sum of points scored by each player in lineup)
    # Captains get 2x and vice captains get 1.1x
    lineup_xp_gw = {
        gw: lpSum(
            [
                player_xp_gw[p, gw] * (lineup[p] + captain[p] + 0.1 * vice_captain[p])
                for p in players
            ]
        )
        for gw in future_gameweeks
    }

    # Dictionary of the final xp of the team for each gameweek (taking into account penalty for transfers: -4 points per transfer over allowed free transfers)
    final_xp_gw = {
        gw: lineup_xp_gw[gw] - 4 * penalised_transfers[gw] for gw in future_gameweeks
    }

    # Dictionary that counts the numbe of players in squad in each gameweek
    squad_count = {gw: lpSum([squad[p, gw] for p in players]) for gw in all_gameweeks}

    # Dictionary that counts the number of players in each position in lineup in each gameweek
    lineup_position_count = {
        (pos, gw): lpSum(
            [
                lineup[p, gw]
                for p in players
                if merged_elements_df.loc[p, "element_type"] == pos
            ]
        )
        for pos in positions
        for gw in future_gameweeks
    }

    # Dictionary that counts the number of players in each position in squad in each gameweek
    squad_position_count = {
        (pos, gw): lpSum(
            [
                squad[p, gw]
                for p in players
                if merged_elements_df.loc[p, "element_type"] == pos
            ]
        )
        for pos in positions
        for gw in all_gameweeks
    }

    # Dictionary that counts the number of players in each team in squad
    squad_team_count = {
        team: lpSum(
            [squad[p] for p in players if merged_elements_df.loc[p, "team"] == team]
        )
        for team in teams
    }

    # Dictionary that stores the number of transfers out in each gameweek
    transfers_out_count = {
        gw: lpSum([transfers_out[p, gw] for p in players]) for gw in future_gameweeks
    }

    # Dictionary that counts number of difference between transfers out and free transfers available in each gameweek
    # A positive value means that we have made more transfers out than the free transfers available, and those will be penalised
    # A negative value means that we have made less transfers out than the free transfers available, and those will not be penalised
    transfer_diff = {
        gw: free_transfers_available[gw] - transfers_out_count[gw]
        for gw in future_gameweeks
    }

    # ----------------------------------------
    # Define initial conditions
    # ----------------------------------------

    # Players in squad in current gameweek must be in current_squad
    for p in players:
        model += squad[p, current_gw] == 1 if p in initial_squad else 0

    # Available balance in current gameweek must be equal to in_the_bank (i.e. amount in the bank)
    model += available_balance[current_gw] == money_in_bank

    # Free transfers for current gameweek must be equal to num_free_transfers
    model += free_transfers_available[current_gw] == num_free_transfers

    # ? Assume we have already made 1 transfer out in current gameweek
    model += transfers_out_count[current_gw] == 1

    # ----------------------------------------
    # Defining squad and lineup constraints
    # ----------------------------------------

    # Total number of players in squad in each gameweek must be equal to 15
    for gw in all_gameweeks:
        model += squad_count[gw] == 15

    # Total number of players in lineup in each gameweek must be equal to 11
    for gw in future_gameweeks:
        model += lpSum([lineup[p, gw] for p in players]) == 11

    # Lineup player must be in squad (but reverse can not be true) in each gameweek
    for gw in future_gameweeks:
        for p in players:
            model += lineup[p, gw] <= squad[p, gw]

    # ----------------------------------------
    # Defining captain and vice captain constraints
    # ----------------------------------------

    # Only 1 captain in each gameweek
    for gw in future_gameweeks:
        model += lpSum([captain[p, gw] for p in players]) == 1

    # Only 1 vice captain in each gameweek
    for gw in future_gameweeks:
        model += lpSum([vice_captain[p, gw] for p in players]) == 1

    # Captain must be in lineup in each gameweek
    for gw in future_gameweeks:
        for p in players:
            model += captain[p, gw] <= lineup[p, gw]

    # Vice captain must be in lineup in each gameweek
    for gw in future_gameweeks:
        for p in players:
            model += vice_captain[p, gw] <= lineup[p, gw]

    # Captain and vice captain can not be the same player in each gameweek
    for gw in future_gameweeks:
        for p in players:
            model += captain[p, gw] + vice_captain[p, gw] <= 1

    # ----------------------------------------
    # Defining position / formation constraints
    # ----------------------------------------

    # Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play)
    # for every gameweek
    for pos in positions:
        for gw in future_gameweeks:
            model += (
                lineup_position_count[pos, gw]
                >= element_types_df.loc[pos, "squad_min_play"]
            )
            model += (
                lineup_position_count[pos, gw]
                <= element_types_df.loc[pos, "squad_max_play"]
            )

    # Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select) for every gameweek
    for pos in positions:
        for gw in all_gameweeks:
            model += (
                squad_position_count[pos, gw]
                == element_types_df.loc[pos, "squad_select"]
            )

    # ----------------------------------------
    # Defining team count constraints
    # ----------------------------------------

    #! Total cost of entire squad must be less than or equal to budget
    #! squad_cost = lpSum(
    #!     [merged_elements_df.loc[p, "now_cost"] * squad[p] for p in players]
    #! )
    #! model += squad_cost <= budget

    # Number of players in each team in squad must be less than or equal to 3 for every gameweek
    for team in teams:
        for gw in all_gameweeks:
            model += squad_team_count[team] <= 3

    # ----------------------------------------
    # Defining player appearance constraints
    # ----------------------------------------

    # Probability of squad player appearing in next gameweek must be greater than or equal to 50%,
    # while probability of lineup player appearing in next gameweek must be greater than or equal to 75%
    # for every gameweek
    for p in players:
        for gw in future_gameweeks:
            model += squad[p, gw] <= (merged_elements_df.loc[p, f"{gw}_prob"] >= 0.5)
            model += lineup[p, gw] <= (merged_elements_df.loc[p, f"{gw}_prob"] >= 0.75)

    # ----------------------------------------
    # Defining transfer constraints
    # ----------------------------------------

    # Players in next gameweek squad must either be in current gameweek squad or transferred in
    # Players not in next gameweek squad must be transferred out
    for p in players:
        for gw in future_gameweeks:
            model += (
                squad[p, gw]
                == squad[p, gw - 1] + transfers_in[p, gw] - transfers_out[p, gw]
            )

    # Available balance in each gameweek must be equal to previous gameweek balance plus transfer revenue minus transfer spent
    for gw in future_gameweeks:
        model += (
            available_balance[gw]
            == available_balance[gw - 1] + transfer_revenue[gw] - transfer_spent[gw]
        )

    # ----------------------------------------
    # Defining free transfer constraints
    # ----------------------------------------

    # Free transfers available in each gameweek is equal to auxillary variable in each gameweek plus 1
    for gw in future_gameweeks:
        model += free_transfers_available[gw] == auxiliary_var[gw] + 1

    # Equality 1: F1 - Tout <= 2 * Aux
    for gw in future_gameweeks:
        model += (
            free_transfers_available[gw - 1] - transfers_out_count[gw - 1]
            <= 2 * auxiliary_var[gw]
        )

    # Equation 2: F1 - Tout >= Aux + (-14) * (1 - Aux)
    for gw in future_gameweeks:
        model += free_transfers_available[gw - 1] - transfers_out_count[
            gw - 1
        ] >= auxiliary_var[gw] + (-14) * (1 - auxiliary_var[gw])

    # Number of penalised transfers in each gameweek must be equal to or greater than the difference between transfers out and free transfers available
    for gw in future_gameweeks:
        model += penalised_transfers[gw] >= transfer_diff[gw]

    # ----------------------------------------
    # Defining objective functions
    # ----------------------------------------

    # Objective function 1 (regular): Maximize final expected points in each gameweek
    if objective == "regular":
        final_xp_obj = lpSum([final_xp_gw[gw] for gw in future_gameweeks])
        model += final_xp_obj

    # Objective function 2 (decay): Maximize final expected points in each gameweek, with decay factor
    elif objective == "decay":
        decay_obj = lpSum(
            [final_xp_gw[gw] * pow(decay_base, gw - next_gw) for gw in future_gameweeks]
        )
        model += decay_obj
        model_name += "_decay_base_" + str(decay_base)

        # Rename model
        model.name = model_name

    # ----------------------------------------
    # Solve model and get results
    # ----------------------------------------

    # Solve model using CBC solver and surpress output
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # If model was solved to optimality, save and get results
    if model.status == 1:
        model_path = "../../models/multi_period/"

        # Check if model_path exists, if not create it
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
        transfers_out = []
        transfers_in = []

        # Loop through model variables for each gameweek and get variables that are 1 (i.e. selected)
        for gw in all_gameweeks:
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

                    # If variable is a transfer out, add to transfers_out list
                    elif v.name[0] == "t" and v.name[1] == "o":
                        id = int(v.name[13:])
                        transfers_out.append(id)

                    # If variable is a transfer in, add to transfers_in list
                    elif v.name[0] == "t" and v.name[1] == "i":
                        id = int(v.name[12:])
                        transfers_in.append(id)

        # ----------------------------------------

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
                    player_data["current_gw"],
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
                "current_gw",
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
    results = solve_single_period_model(1000)["results"]

    pass
