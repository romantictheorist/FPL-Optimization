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
# Set parameters of model (will be passed in through function later)
# ----------------------------------------

team_id = 216079 # Team ID
gameweek = 22 # Upcoming (i.e. next) gameweek
bank_balance = 4.7 # Money in the bank
horizon = 7 # Number of gameweeks we are solving for
objective = "decay" # "regular" or "decay
num_free_transfers = 1 # Number of free transfers available
decay_base = 0.9 # Decay base for decay objective function

# ----------------------------------------
# Pull data from FPL API-
# ----------------------------------------

# Latest general data
general_data = pull_general_data()
elements_df = general_data["elements"]
element_types_df = general_data["element_types"]
teams_df = general_data["teams"]

# Current squad, i.e. the squad from gameweek prior to the one we are solving for)
# initial_squad = pull_squad(team_id=team_id, gw=gameweek - 1)
initial_squad = [275, 369, 342, 506, 19, 526, 664, 14, 117, 60, 343, 230, 129, 112, 126]

# Dataframe of players in initial squad
initial_squad_df = elements_df[elements_df["id"].isin(initial_squad)]
initial_squad_list = initial_squad_df["web_name"].values.tolist()

# ----------------------------------------
# Read in fpl_form_data and merge with elements_df
# ----------------------------------------

fpl_form_data = pd.read_csv("../../data/raw/fpl-form-predicted-points.csv")
merged_elements_df = merge_fpl_form_data(elements_df, fpl_form_data)

# ----------------------------------------
# Set index to 'id's
# ----------------------------------------

merged_elements_df.set_index("id", inplace=True)
element_types_df.set_index("id", inplace=True)
teams_df.set_index("id", inplace=True)

# ----------------------------------------
# Create sets
# ----------------------------------------

players = list(merged_elements_df.index)
positions = list(element_types_df.index)
teams = list(teams_df.index)
future_gameweeks = list(range(gameweek, gameweek + horizon))
all_gameweeks = [gameweek - 1] + future_gameweeks

# ----------------------------------------
# Initialise model
# ----------------------------------------

model_name = f"solve_EV_max_gw_{gameweek}_horizon_{horizon}_objective_{objective}"
model = LpProblem(model_name, sense=LpMaximize)

# ----------------------------------------
# Define decision variables
# ----------------------------------------

squad = LpVariable.dicts("squad", (players, all_gameweeks), cat="Binary")
lineup = LpVariable.dicts("lineup", (players, future_gameweeks), cat="Binary")
captain = LpVariable.dicts("captain", (players, future_gameweeks), cat="Binary")
vice_captain = LpVariable.dicts("vice_captain", (players, future_gameweeks), cat="Binary")
transfer_in = LpVariable.dicts("transfer_in", (players, future_gameweeks), cat="Binary")
transfer_out = LpVariable.dicts("transfer_out", (players, future_gameweeks), cat="Binary")
money_in_bank = LpVariable.dicts("money_in_bank", (all_gameweeks), lowBound=0, cat="Continuous")
free_transfers_available = LpVariable.dicts("free_transfers_available", (all_gameweeks), lowBound=1, upBound=2, cat="Integer")
penalised_transfers = LpVariable.dicts("penalised_transfers", (future_gameweeks), cat="Integer", lowBound=0)
aux = LpVariable.dicts("auxiliary_variable", (future_gameweeks), cat="Binary")

# ----------------------------------------
# Create dictionaries to use for constraints
# ----------------------------------------

# Dictionary that stores the cost of each player
player_cost = merged_elements_df["now_cost"].to_dict()

# Dictionary that stores the expected points of each player in each gameweek
player_xp_gw = {(p, gw): merged_elements_df.loc[p, f"{gw}_pts_no_prob"] for p in players for gw in future_gameweeks}

# Dictionary that stores the probability of each player appearing in each gameweek
player_prob_gw = {(p, gw): merged_elements_df.loc[p, f"{gw}_prob"] for p in players for gw in future_gameweeks}

# Dictionary that counts the number of players in squad in each gameweek
squad_count = {gw: lpSum([squad[p][gw] for p in players]) for gw in future_gameweeks}

# Dictionary that counts the number of players in lineup in each gameweek
lineup_count = {gw: lpSum([lineup[p][gw] for p in players]) for gw in future_gameweeks}

# Dictionary that counts the number of players in each position in lineup in each gameweek
lineup_position_count = {(pos, gw): lpSum([lineup[p][gw] for p in players if merged_elements_df.loc[p, "element_type"] == pos]) for pos in positions for gw in future_gameweeks}

# Dictionary that counts the number of players in each position in squad in each gameweek
squad_position_count = {(pos, gw): lpSum([squad[p][gw] for p in players if merged_elements_df.loc[p, "element_type"] == pos]) for pos in positions for gw in future_gameweeks}

# Dictionary that stores the number of players from each team in squad in each gameweek
squad_team_count = {(team, gw): lpSum([squad[p][gw] for p in players if merged_elements_df.loc[p, "team"] == team]) for team in teams for gw in future_gameweeks}

# Dictionary that stores the revenue made transfers from each gameweek (i.e. amount sold)
revenue = {gw: lpSum([player_cost[p] * transfer_out[p][gw] for p in players]) for gw in future_gameweeks}

# Dictionary that stores the amount spent on transfers in each gameweek (i.e. expenditure)
expenditure = {gw: lpSum([player_cost[p] * transfer_in[p][gw] for p in players]) for gw in future_gameweeks}

# Dictionary that stores the number of transfers made in each gameweek (i.e. number of transfers in OR number of transfers out, as they are equal)
transfers_made = {gw: lpSum([transfer_in[p][gw] for p in players]) for gw in future_gameweeks}

# Assume we have already made 1 transfer in current gameweek (does not affect number of free transfers available for following gameweeks)
transfers_made[gameweek - 1] = 1

# Dictionary that counts number of difference between transfers made and free transfers available in each gameweek
# A positive value means that we have made more transfers than allowed, and those will be penalised
# A negative value means that we have made less transfers out allowed, and those will not be penalised
transfer_diff = {gw: (transfers_made[gw] - free_transfers_available[gw]) for gw in future_gameweeks}

# ----------------------------------------
# Define initial conditions
# ----------------------------------------

# Players in initial squad must be in squad in current gameweek
for p in [player for player in players if player in initial_squad]:
    model += squad[p][gameweek - 1] == 1, f"In initial squad constraint for player {p}"

# Players not in initial squad must not be in squad in current gameweek
for p in [player for player in players if player not in initial_squad]:
    model += squad[p][gameweek - 1] == 0, f"Not initial squad constraint for player {p}"
    
# Money in bank at current gameweek must be equal to bank balance
model += money_in_bank[gameweek - 1] == bank_balance, f"Initial money in bank constraint"

# Number of free transfers available in current gameweek must be equal to num_free_transfers
model += free_transfers_available[gameweek - 1] == num_free_transfers, f"Initial free transfers available constraint"

# ----------------------------------------
# Defining squad and lineup constraints
# ----------------------------------------

# Total number of players in squad in each gameweek must be equal to 15
for gw in future_gameweeks:
    model += squad_count[gw] == 15, f"Squad count constraint for gameweek {gw}"

# Total number of players in lineup in each gameweek must be equal to 11
for gw in future_gameweeks:
    model += lineup_count[gw] == 11, f"Lineup count constraint for gameweek {gw}"

# Lineup player must be in squad (but reverse can not be true) in each gameweek
for gw in future_gameweeks:
    for p in players:
        model += lineup[p][gw] <= squad[p][gw], f"Lineup player must be in squad constraint for player {p} in gameweek {gw}"

# ----------------------------------------
# Defining captain and vice captain constraints
# ----------------------------------------

# Only 1 captain in each gameweek
for gw in future_gameweeks:
    model += lpSum([captain[p][gw] for p in players]) == 1, f"Captain count constraint for gameweek {gw}"

# Only 1 vice captain in each gameweek
for gw in future_gameweeks:
    model += lpSum([vice_captain[p][gw] for p in players]) == 1, f"Vice captain count constraint for gameweek {gw}"

# Captain must be in lineup in each gameweek
for gw in future_gameweeks:
    for p in players:
        model += captain[p][gw] <= lineup[p][gw], f"Captain must be in lineup constraint for player {p} in gameweek {gw}"

# Vice captain must be in lineup in each gameweek
for gw in future_gameweeks:
    for p in players:
        model += vice_captain[p][gw] <= lineup[p][gw], f"Vice captain must be in lineup constraint for player {p} in gameweek {gw}"

# Captain and vice captain can not be the same player in each gameweek
for gw in future_gameweeks:
    for p in players:
        model += captain[p][gw] + vice_captain[p][gw] <= 1, f"Captain and vice captain can not be the same player constraint for player {p} in gameweek {gw}"
        
# ----------------------------------------
# Defining position / formation constraints
# ----------------------------------------

# Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for every gameweek
for gw in future_gameweeks:
    for pos in positions:
        model += (lineup_position_count[pos, gw] >= element_types_df.loc[pos, "squad_min_play"]), f"Min lineup players in position {pos} in gameweek {gw}"
        model += (lineup_position_count[pos, gw] <= element_types_df.loc[pos, "squad_max_play"]), f"Max lineup players in position {pos} in gameweek {gw}"


# # Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select) for every gameweek
for gw in future_gameweeks:
    for pos in positions:
        model += (squad_position_count[pos, gw] == element_types_df.loc[pos, "squad_select"]), f"Squad players in position {pos} in gameweek {gw}"

# ----------------------------------------
# Defining team count constraints
# ----------------------------------------

# Number of players in each team in squad must be less than or equal to 3 for every gameweek
for gw in future_gameweeks:
    for team in teams:
        model += (squad_team_count[team, gw] <= 3), f"Max players from team {team} in gameweek {gw}"

# ----------------------------------------
# Defining player appearance constraints
# ----------------------------------------

# For every gameweek the probability of squad player appearing in next gameweek must be >= 50%, while probability of lineup player > 75%
for gw in future_gameweeks:
    for p in players:
        model += squad[p][gw] <= (player_prob_gw[p, gw] >= 0.5), f"Probability of appearance for squad player {p} for gameweek {gw}"
        model += lineup[p][gw] <= (player_prob_gw[p, gw] >= 0.75), f"Probability of appearance for lineup player {p} for gameweek {gw}"
    
# ----------------------------------------
# Defining budget/money constraints
# ----------------------------------------

# Money in bank in each gameweek must be equal to previous gameweek money in bank plus transfer revenue minus transfer expenditure
for gw in future_gameweeks:
    model += (money_in_bank[gw] == (money_in_bank[gw - 1] + revenue[gw] - expenditure[gw])), f"Money in bank constraint for gameweek {gw}"

# ----------------------------------------
# Defining transfer constraints
# ----------------------------------------

# Players in next gameweek squad must either be in current gameweek squad or transferred in
# And players not in next gameweek squad must be transferred out
for gw in future_gameweeks:
    for p in players:
        model += (squad[p][gw] == (squad[p][gw - 1] + transfer_in[p][gw] - transfer_out[p][gw])), f"Player {p} squad/transfer constraint for gameweek {gw}"

# Number of transfers made in each gameweek cannot exceed 5
for gw in future_gameweeks:
    model += transfers_made[gw] <= 20, f"Transfers made constraint for gameweek {gw}"
    
# ----------------------------------------
# Defining free transfer constraints
# ----------------------------------------

# Free transfers available and auxiliary variable conditions for each gameweek
for gw in future_gameweeks:
    model += (free_transfers_available[gw] == (aux[gw] + 1)), f"FTA and Aux constraint for gameweek {gw}"

# Equality 1: FTA_{1} - TM_{1} <= 2 * Aux_{2}
for gw in future_gameweeks:
    model += free_transfers_available[gw - 1] - transfers_made[gw - 1] <= 2 * aux[gw], f"FTA and TM Equality 1 constraint for gameweek {gw}"
    
# Equality 2: FTA_{1} - TM_{1} >= Aux_{2} + (-14) * (1 - Aux_{2})
for gw in future_gameweeks:
    model += free_transfers_available[gw - 1] - transfers_made[gw - 1] >= aux[gw] + (-14) * (1 - aux[gw]), f"FTA and TM Equality 2 constraint for gameweek {gw}"

# Number of penalised transfers in each gameweek must be equal to or greater than the transfer difference (i.e. number of transfers made minus number of free transfers available)
# I.e. only penalise transfers if we have made more transfers than allowed
for gw in future_gameweeks:
    model += penalised_transfers[gw] >= transfer_diff[gw], f"Penalised transfers constraint for gameweek {gw}"
    
# ----------------------------------------
# Defining objective functions
# ----------------------------------------

# Dictionary of total expected points for each gameweek (i.e. sum of expected points for each player in lineup) with weights for captain and vice captain
gw_xp_before_pen = {gw: lpSum([player_xp_gw[p, gw] * (lineup[p][gw] + captain[p][gw] + 0.1 * vice_captain[p][gw]) for p in players]) for gw in future_gameweeks}

# Dictionary of final expected points for each gameweek (i.e. with transfer penalty of -4 points for each penalised transfer)
gw_xp_after_pen = {gw: gw_xp_before_pen[gw] - 4 * penalised_transfers[gw] for gw in future_gameweeks}

# Objective function 1 (regular): Maximize total expected points over all gameweeks 
if objective == "regular":
    total_xp = lpSum([gw_xp_after_pen [gw] for gw in future_gameweeks])
    model += total_xp
    
# Objective function 2 (decay): Maximize final expected points in each gameweek, with decay factor
elif objective == "decay":
    total_xp = lpSum([gw_xp_after_pen[gw] * pow(decay_base, gw - gameweek) for gw in future_gameweeks])
    model += total_xp
    model_name += "_decay_base_" + str(decay_base)
    model.name = model_name

# ----------------------------------------
# Solve model and get results
# ----------------------------------------

# Solve model using CBC solver and surpress output
model.solve(pulp.PULP_CBC_CMD(msg=0))

# If model was solved to optimality, save and get results
if model.status == 1:
    print("Status: Model solved to optimality.")
    model_path = "../../models/multi_period/"

    # Check if model_path exists, if not create it
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        pass

    # Write solved model to .mps file
    model.writeMPS(model_path + model_name + ".mps")
    
    # Get results for each gameweek
    results = []

    for gw in future_gameweeks:
        for p in players:
            if squad[p][gw].varValue == 1 or transfer_out[p][gw].varValue == 1:
                results.append(
                    {   
                        "gw": gw,
                        "player_id": p,
                        "player_name": merged_elements_df.loc[p, "web_name"],
                        "team": merged_elements_df.loc[p, "team_name"],
                        "position": merged_elements_df.loc[p, "position"],
                        "position_id": merged_elements_df.loc[p, "element_type"],
                        "cost": player_cost[p],
                        "prob_appearance": player_prob_gw[p, gw],
                        "xp": player_xp_gw[p, gw],
                        "squad": squad[p][gw].varValue,
                        "lineup": lineup[p][gw].varValue,
                        "captain": captain[p][gw].varValue,
                        "vice_captain": vice_captain[p][gw].varValue,
                        "transfer_in": transfer_in[p][gw].varValue,
                        "transfer_out": transfer_out[p][gw].varValue,
                    }
                )
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results).round(2)
    
    # Sort results
    results_df.sort_values(by=["gw", "squad", "lineup", "position_id", "xp"], ascending=[True, False, False, True, False], inplace=True)
    
    # ----------------------------------------
    # Check results
    # ----------------------------------------
    
    # For every gameweek, check the following:
    
    # 1. Number of players in squad is equal to 15
    # 2. Number of players in lineup is equal to 11
    # 3. Number of transfer_in is equal to number of transfer_out
    # 4. Number of players from a team in squad is less than or equal to 3
    # 5. Number of players in each position in squad is equal to squad_select (defined in element_types_df)
    # 6. Number of players in each position in lineup is within the allowed range (defined in element_types_df as squad_min_play and squad_max_play)
    # 7. Probability of appearance for each player in squad is greater than 50%
    # 8. Probability of appearance for each player in lineup is greater than 75%
    # 9. Only 1 captain
    # 10. Only 1 vice captain
    # 11. Captain must be in lineup
    # 12. Vice captain must be in lineup
 
    # If any of the above are not true, print a warning message
    # If all are true, print a success message
    
    checks_dict = {} # True if all checks are passed, False otherwise
    
    for gw in future_gameweeks:
        condition_1 = results_df[results_df["gw"] == gw].squad.sum() == 15
        condition_2 = results_df[results_df["gw"] == gw].lineup.sum() == 11
        condition_3 = results_df[results_df["gw"] == gw].transfer_in.sum() == results_df[results_df["gw"] == gw].transfer_out.sum()
        condition_4 = results_df[(results_df["gw"] == gw) & (results_df["squad"] == 1)].team.value_counts().max() <= 3
        condition_5 = all(results_df[results_df["gw"] == gw].groupby("position_id").squad.sum() == element_types_df["squad_select"])
        condition_6a = all(results_df[results_df["gw"] == gw].groupby("position_id").lineup.sum() >= element_types_df["squad_min_play"])
        condition_6b = all(results_df[results_df["gw"] == gw].groupby("position_id").lineup.sum() <= element_types_df["squad_max_play"])
        condition_7 = all(results_df[(results_df["gw"] == gw) & (results_df["squad"] == 1)].prob_appearance > 0.5)
        condition_8 = all(results_df[(results_df["gw"] == gw) & (results_df["lineup"] == 1)].prob_appearance > 0.75)
        condition_9 = results_df[results_df["gw"] == gw].captain.sum() == 1
        condition_10 = results_df[results_df["gw"] == gw].vice_captain.sum() == 1
        condition_11 = all(results_df[(results_df["gw"] == gw) & (results_df["captain"] == 1)].lineup == 1)
        condition_12 = all(results_df[(results_df["gw"] == gw) & (results_df["vice_captain"] == 1)].lineup == 1)
  
        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6a and condition_6b and condition_7 and condition_8 and condition_9 and condition_10 and condition_11 and condition_12:
            checks_dict[gw] = True
        else:
            checks_dict[gw] = False
            print(f"WARNING: Results for gameweek {gw} are not correct.")
            if not condition_1:
                print(f"WARNING: Number of players in squad for gameweek {gw} is not 15.")
            if not condition_2:
                print(f"WARNING: Number of players in lineup for gameweek {gw} is not 11.")
            if not condition_3:
                print(f"WARNING: Number of transfers in is not equal to number of transfer out for gameweek {gw}.")
            if not condition_4:
                print(f"WARNING: Number of players from each team in squad exceeds the limit of 3 for gameweek {gw}.")
            if not condition_5:
                print(f"WARNING: Number of players in each position in squad is not equal to squad_select (defined in element_types_df) for gameweek {gw}.")
            if not condition_6a:
                print(f"WARNING: Number of players in each position in lineup is greater than the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for gameweek {gw}.")
            if not condition_6b:
                print(f"WARNING: Number of players in each position in lineup is less than the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for gameweek {gw}.")
            if not condition_7:
                print(f"WARNING: Probability of appearance for each player in squad is not greater than 50% for gameweek {gw}.")
            if not condition_8:
                print(f"WARNING: Probability of appearance for each player in lineup is not greater than 75% for gameweek {gw}.")
            if not condition_9:
                print(f"WARNING: Number of captains is not equal to 1 for gameweek {gw}.")
            if not condition_10:
                print(f"WARNING: Number of vice captains is not equal to 1 for gameweek {gw}.")
            if not condition_11:
                print(f"WARNING: Captain is not in lineup for gameweek {gw}.")
            if not condition_12:
                print(f"WARNING: Vice captain is not in lineup for gameweek {gw}.")
    
            print("\n")
            
    # If all checks are passed, print a success message
    if all(value == True for value in checks_dict.values()):
        print("All checks passed.")
        
    # ----------------------------------------
    # Summary of actions
    # ----------------------------------------
    
    # For every gameweek, print the following:
    
    # 1. Money in bank
    # 2. Number of free transfers available
    # 3. Number of transfers made
    # 4. Number of penalised transfers
    # 5. Players transferred in (ID, name, team, position, expected points)
    # 6. Players transferred out (ID, name, team, position, expected points)
    # 6. Players captained (ID, name, team, position, expected points)
    # 7. Players vice captained (ID, name, team, position, expected points)
    
    # If no players are transferred in/out, benched, captained or vice captained, print a message saying so
    
    for gw in future_gameweeks:
        print("-" * 50)
        print(f"Gameweek {gw} summary:")
        print("-" * 50)
        print(f"Total expected points: {round(value(gw_xp_after_pen[gw]))}")
        print(f"Money in bank: {money_in_bank[gw].varValue}")
        print(f"Free transfers available: {int(free_transfers_available[gw].varValue)}")
        print(f"Transfers made: {int(value(transfers_made[gw]))}")
        print(f"Penalised transfers: {int(penalised_transfers[gw].varValue)}")
        
        for p in players:
            if transfer_in[p][gw].varValue == 1:
                print(f"Player {p} ({merged_elements_df.loc[p, 'web_name']} @ {merged_elements_df.loc[p, 'team_name']}) transferred in.")
            if transfer_out[p][gw].varValue == 1:
                print(f"Player {p} ({merged_elements_df.loc[p, 'web_name']} @ {merged_elements_df.loc[p, 'team_name']}) transferred out.")
                
else:
    print("Model could not be solved.")
    print("Status:", LpStatus[model.status])
    
# ----------------------------------------





    
