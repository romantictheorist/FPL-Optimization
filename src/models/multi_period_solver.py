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


class OptimizeMultiPeriod:
    def __init__(self, team_id, gameweek, bank_balance, num_free_transfers, horizon, objective='regular', decay_base=0.85):
        self.team_id = team_id
        self.gameweek = gameweek
        self.bank_balance = bank_balance
        self.num_free_transfers = num_free_transfers
        self.horizon = horizon
        self.objective = objective
        self.decay_base = decay_base
        self.model = None
        self.results = None
        self.summary = None
        self.total_xp = None
        self.gw_xp = None
        self.checks = None

    def prepare_data(self):
        data = pull_general_data()
        #initial_squad = pull_squad(self.team_id)
        initial_squad = [275, 369, 342, 506, 19, 526, 664, 14, 117, 60, 343, 230, 129, 112, 126]
        
        elements_df = data["elements"]
        elements_types_df = data["element_types"]
        teams_df = data["teams"]
        
        form_data = pd.read_csv("../../data/raw/fpl-form-predicted-points.csv")
        merged_elements_df = merge_fpl_form_data(elements_df, form_data)
        
        merged_elements_df.set_index("id", inplace=True)
        elements_types_df.set_index("id", inplace=True)
        teams_df.set_index("id", inplace=True)
        
        players = merged_elements_df.index.tolist()
        positions = elements_types_df.index.tolist()
        teams = teams_df.index.tolist()
        future_gameweeks = list(range(self.gameweek, self.gameweek + self.horizon))
        all_gameweeks = [self.gameweek - 1] + future_gameweeks
        
        return {'merged_elements_df': merged_elements_df, 'elements_types_df': elements_types_df, 'teams_df': teams_df, 'initial_squad': initial_squad,
                'players': players, 'positions': positions, 'teams': teams, 'future_gameweeks': future_gameweeks, 'all_gameweeks': all_gameweeks}
        
    
    def define_problem(self):
        name = f"solve_EV_max_gw_{self.gameweek}_horizon_{self.horizon}_objective_{self.objective}"
        self.model = LpProblem(name, LpMaximize)
        pass
    
    
    def define_variables(self, data):
        # Lists
        players = data["players"]
        future_gameweeks = data["future_gameweeks"]
        all_gameweeks = data["all_gameweeks"]
        
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
        
        return {'squad': squad, 'lineup': lineup, 'captain': captain, 'vice_captain': vice_captain, 'transfer_in': transfer_in, 'transfer_out': transfer_out, 'money_in_bank': money_in_bank, 'free_transfers_available': free_transfers_available, 'penalised_transfers': penalised_transfers, 'aux': aux}
        
    def define_dictionaries(self, data, variables):
        # Dataframe
        merged_elements_df = data["merged_elements_df"]
  
        # Lists
        players = data["players"]
        positions = data["positions"]
        teams = data["teams"]
        future_gameweeks = data["future_gameweeks"]
        all_gameweeks = data["all_gameweeks"]
        
        # Variables
        squad = variables["squad"]
        lineup = variables["lineup"]
        transfer_in = variables["transfer_in"]
        transfer_out = variables["transfer_out"]
        free_transfers_available = variables["free_transfers_available"]
        
        # Cost of each player
        player_cost = merged_elements_df["now_cost"].to_dict()

        # Expected points of each player in each gameweek
        player_xp_gw = {(p, gw): merged_elements_df.loc[p, f"{gw}_pts_no_prob"] for p in players for gw in future_gameweeks}

        # Probability of each player appearing in each gameweek
        player_prob_gw = {(p, gw): merged_elements_df.loc[p, f"{gw}_prob"] for p in players for gw in future_gameweeks}

        # Number of players in squad in each gameweek
        squad_count = {gw: lpSum([squad[p][gw] for p in players]) for gw in future_gameweeks}

        # Number of players in lineup in each gameweek
        lineup_count = {gw: lpSum([lineup[p][gw] for p in players]) for gw in future_gameweeks}

        # Number of players in each position in lineup in each gameweek
        lineup_position_count = {(pos, gw): lpSum([lineup[p][gw] for p in players if merged_elements_df.loc[p, "element_type"] == pos]) for pos in positions for gw in future_gameweeks}

        # Nnumber of players in each position in squad in each gameweek
        squad_position_count = {(pos, gw): lpSum([squad[p][gw] for p in players if merged_elements_df.loc[p, "element_type"] == pos]) for pos in positions for gw in future_gameweeks}

        # Number of players from each team in squad in each gameweek
        squad_team_count = {(team, gw): lpSum([squad[p][gw] for p in players if merged_elements_df.loc[p, "team"] == team]) for team in teams for gw in future_gameweeks}

        # Transfer revenue in each gameweek (i.e. amount sold)
        revenue = {gw: lpSum([player_cost[p] * transfer_out[p][gw] for p in players]) for gw in future_gameweeks}

        # Transfer spend in each gameweek 
        expenditure = {gw: lpSum([player_cost[p] * transfer_in[p][gw] for p in players]) for gw in future_gameweeks}

        # Number of transfers made in each gameweek (i.e. number of transfers in OR number of transfers out, as they are equal)
        transfers_made = {gw: lpSum([transfer_in[p][gw] for p in players]) for gw in future_gameweeks}

        # Assume we have already made 1 transfer in current gameweek (does not affect number of free transfers available for following gameweeks)
        transfers_made[self.gameweek - 1] = 1

        # Difference between transfers made and free transfers available in each gameweek
        # A positive value means that we have made more transfers than allowed, and those will be penalised
        # A negative value means that we have made less transfers out allowed, and those will not be penalised
        transfer_diff = {gw: (transfers_made[gw] - free_transfers_available[gw]) for gw in future_gameweeks}
        
        return {'player_cost': player_cost, 'player_xp_gw': player_xp_gw, 'player_prob_gw': player_prob_gw, 'squad_count': squad_count, 'lineup_count': lineup_count, 'lineup_position_count': lineup_position_count, 'squad_position_count': squad_position_count, 'squad_team_count': squad_team_count, 'revenue': revenue, 'expenditure': expenditure, 'transfers_made': transfers_made, 'transfer_diff': transfer_diff}
    
    def define_initial_conditions(self, data, variables):
        # Lists
        players = data["players"]
        initial_squad = data["initial_squad"]
        
        # Variables
        squad = variables["squad"]
        money_in_bank = variables["money_in_bank"]
        free_transfers_available = variables["free_transfers_available"]
        
        # Players in initial squad must be in squad in current gameweek
        for p in [player for player in players if player in initial_squad]:
            self.model += squad[p][self.gameweek - 1] == 1, f"In initial squad constraint for player {p}"

        # Players not in initial squad must not be in squad in current gameweek
        for p in [player for player in players if player not in initial_squad]:
            self.model += squad[p][self.gameweek - 1] == 0, f"Not initial squad constraint for player {p}"
            
        # Money in bank at current gameweek must be equal to bank balance
        self.model += money_in_bank[self.gameweek - 1] == self.bank_balance, f"Initial money in bank constraint"

        # Number of free transfers available in current gameweek must be equal to num_free_transfers
        self.model += free_transfers_available[self.gameweek - 1] == self.num_free_transfers, f"Initial free transfers available constraint"
            
        
    def define_constraints(self, data, variables, dictionaries):
        # Dataframe
        element_types_df = data["elements_types_df"]
                
        # Lists
        players = data["players"]
        positions = data["positions"]
        teams = data["teams"]
        future_gameweeks = data["future_gameweeks"]

        # Variables
        squad = variables["squad"]
        lineup = variables["lineup"]
        captain = variables["captain"]
        vice_captain = variables["vice_captain"]
        transfer_in = variables["transfer_in"]
        transfer_out = variables["transfer_out"]
        money_in_bank = variables["money_in_bank"]
        free_transfers_available = variables["free_transfers_available"]
        penalised_transfers = variables["penalised_transfers"]
        aux = variables["aux"]
        
        # Dictionaries
        player_prob_gw = dictionaries["player_prob_gw"]
        squad_count = dictionaries["squad_count"]
        lineup_count = dictionaries["lineup_count"]
        lineup_position_count = dictionaries["lineup_position_count"]
        squad_position_count = dictionaries["squad_position_count"]
        squad_team_count = dictionaries["squad_team_count"]
        revenue = dictionaries["revenue"]
        expenditure = dictionaries["expenditure"]
        transfers_made = dictionaries["transfers_made"]
        transfer_diff = dictionaries["transfer_diff"]
        
        # ----------------------------------------
        # Squad and lineup constraints
        # ----------------------------------------

        # Total number of players in squad in each gameweek must be equal to 15
        for gw in future_gameweeks:
            self.model += squad_count[gw] == 15, f"Squad count constraint for gameweek {gw}"

        # Total number of players in lineup in each gameweek must be equal to 11
        for gw in future_gameweeks:
            self.model += lineup_count[gw] == 11, f"Lineup count constraint for gameweek {gw}"

        # Lineup player must be in squad (but reverse can not be true) in each gameweek
        for gw in future_gameweeks:
            for p in players:
                self.model += lineup[p][gw] <= squad[p][gw], f"Lineup player must be in squad constraint for player {p} in gameweek {gw}"

        # ----------------------------------------
        # Captain and vice captain constraints
        # ----------------------------------------

        # Only 1 captain in each gameweek
        for gw in future_gameweeks:
            self.model += lpSum([captain[p][gw] for p in players]) == 1, f"Captain count constraint for gameweek {gw}"

        # Only 1 vice captain in each gameweek
        for gw in future_gameweeks:
            self.model += lpSum([vice_captain[p][gw] for p in players]) == 1, f"Vice captain count constraint for gameweek {gw}"

        # Captain must be in lineup in each gameweek
        for gw in future_gameweeks:
            for p in players:
                self.model += captain[p][gw] <= lineup[p][gw], f"Captain must be in lineup constraint for player {p} in gameweek {gw}"

        # Vice captain must be in lineup in each gameweek
        for gw in future_gameweeks:
            for p in players:
                self.model += vice_captain[p][gw] <= lineup[p][gw], f"Vice captain must be in lineup constraint for player {p} in gameweek {gw}"

        # Captain and vice captain can not be the same player in each gameweek
        for gw in future_gameweeks:
            for p in players:
                self.model += captain[p][gw] + vice_captain[p][gw] <= 1, f"Captain and vice captain can not be the same player constraint for player {p} in gameweek {gw}"
                
        # ----------------------------------------
        # Position / Formation constraints
        # ----------------------------------------

        # Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for every gameweek
        for gw in future_gameweeks:
            for pos in positions:
                self.model += (lineup_position_count[pos, gw] >= element_types_df.loc[pos, "squad_min_play"]), f"Min lineup players in position {pos} in gameweek {gw}"
                self.model += (lineup_position_count[pos, gw] <= element_types_df.loc[pos, "squad_max_play"]), f"Max lineup players in position {pos} in gameweek {gw}"


        # Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select) for every gameweek
        for gw in future_gameweeks:
            for pos in positions:
                self.model += (squad_position_count[pos, gw] == element_types_df.loc[pos, "squad_select"]), f"Squad players in position {pos} in gameweek {gw}"

        # ----------------------------------------
        # Team played for constraints
        # ----------------------------------------

        # Number of players in each team in squad must be less than or equal to 3 for every gameweek
        for gw in future_gameweeks:
            for team in teams:
                self.model += (squad_team_count[team, gw] <= 3), f"Max players from team {team} in gameweek {gw}"

        # ----------------------------------------
        # Probability of appearance constraints
        # ----------------------------------------

        # For every gameweek the probability of squad player appearing in next gameweek must be >= 50%, while probability of lineup player > 75%
        for gw in future_gameweeks:
            for p in players:
                self.model += squad[p][gw] <= (player_prob_gw[p, gw] >= 0.5), f"Probability of appearance for squad player {p} for gameweek {gw}"
                self.model += lineup[p][gw] <= (player_prob_gw[p, gw] >= 0.75), f"Probability of appearance for lineup player {p} for gameweek {gw}"
            
        # ----------------------------------------
        # Budgeting / Financial constraints
        # ----------------------------------------

        # Money in bank in each gameweek must be equal to previous gameweek money in bank plus transfer revenue minus transfer expenditure
        for gw in future_gameweeks:
            self.model += (money_in_bank[gw] == (money_in_bank[gw - 1] + revenue[gw] - expenditure[gw])), f"Money in bank constraint for gameweek {gw}"

        # ----------------------------------------
        # General transfer constraints
        # ----------------------------------------

        # Players in next gameweek squad must either be in current gameweek squad or transferred in
        # And players not in next gameweek squad must be transferred out
        for gw in future_gameweeks:
            for p in players:
                self.model += (squad[p][gw] == (squad[p][gw - 1] + transfer_in[p][gw] - transfer_out[p][gw])), f"Player {p} squad/transfer constraint for gameweek {gw}"

        # Number of transfers made in each gameweek cannot exceed 5
        for gw in future_gameweeks:
            self.model += transfers_made[gw] <= 20, f"Transfers made constraint for gameweek {gw}"
            
        # ----------------------------------------
        # Free transfer constraints
        # ----------------------------------------

        # Free transfers available and auxiliary variable conditions for each gameweek
        for gw in future_gameweeks:
            self.model += (free_transfers_available[gw] == (aux[gw] + 1)), f"FTA and Aux constraint for gameweek {gw}"

        # Equality 1: FTA_{1} - TM_{1} <= 2 * Aux_{2}
        for gw in future_gameweeks:
            self.model += free_transfers_available[gw - 1] - transfers_made[gw - 1] <= 2 * aux[gw], f"FTA and TM Equality 1 constraint for gameweek {gw}"
            
        # Equality 2: FTA_{1} - TM_{1} >= Aux_{2} + (-14) * (1 - Aux_{2})
        for gw in future_gameweeks:
            self.model += free_transfers_available[gw - 1] - transfers_made[gw - 1] >= aux[gw] + (-14) * (1 - aux[gw]), f"FTA and TM Equality 2 constraint for gameweek {gw}"

        # Number of penalised transfers in each gameweek must be equal to or greater than the transfer difference (i.e. number of transfers made minus number of free transfers available)
        # I.e. only penalise transfers if we have made more transfers than allowed
        for gw in future_gameweeks:
            self.model += penalised_transfers[gw] >= transfer_diff[gw], f"Penalised transfers constraint for gameweek {gw}"
        
        
    def define_objective(self, data, variables, dictionaries):
        # Lists
        players = data["players"]
        future_gameweeks = data["future_gameweeks"]
   
        # Variables
        lineup = variables["lineup"]
        captain = variables["captain"]
        vice_captain = variables["vice_captain"]
        penalised_transfers = variables["penalised_transfers"]
        
        # Dictionaries
        player_xp_gw = dictionaries["player_xp_gw"]
        gw_xp_before_pen = {gw: lpSum([player_xp_gw[p, gw] * (lineup[p][gw] + captain[p][gw] + 0.1 * vice_captain[p][gw]) for p in players]) for gw in future_gameweeks}
        gw_xp_after_pen = {gw: gw_xp_before_pen[gw] - 4 * penalised_transfers[gw] for gw in future_gameweeks}

        # Objective function 1 (regular): Maximize total expected points over all gameweeks 
        if self.objective == "regular":
            total_xp = lpSum([gw_xp_after_pen [gw] for gw in future_gameweeks])
            self.model += total_xp
            
        # Objective function 2 (decay): Maximize final expected points in each gameweek, with decay factor
        elif self.objective == "decay":
            total_xp = lpSum([gw_xp_after_pen[gw] * pow(self.decay_base, gw - self.gameweek) for gw in future_gameweeks])
            self.model += total_xp
            model_name += "_decay_base_" + str(self.decay_base)
            self.model.name = model_name
            
    
    def extract_results(self, data, variables, dictionaries):
        if self.model.status != 1:
            print("Cannot extract results available since model is not solved.")
            return None
        else:
            # Dataframe
            merged_elements_df = data["merged_elements_df"]
            
            # Lists
            players = data["players"]
            future_gameweeks = data["future_gameweeks"]
            
            # Variables
            squad = variables["squad"]
            lineup = variables["lineup"]
            captain = variables["captain"]
            vice_captain = variables["vice_captain"]
            transfer_in = variables["transfer_in"]
            transfer_out = variables["transfer_out"]
            penalised_transfers = variables["penalised_transfers"]
            
            # Dictionaries
            player_cost = dictionaries["player_cost"]
            player_prob_gw = dictionaries["player_prob_gw"]
            player_xp_gw = dictionaries["player_xp_gw"]
            
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
            
            # Sort results and reset index
            results_df.sort_values(by=["gw", "squad", "lineup", "position_id", "xp"], ascending=[True, False, False, True, False], inplace=True)
            results_df.reset_index(drop=True, inplace=True)
            
            # Update results attribute
            self.results = results_df
            
            # Update gw_xp attribute
            self.gw_xp = {gw: round(value(lpSum([player_xp_gw[p, gw] * (lineup[p][gw] + captain[p][gw] - (4 * penalised_transfers[gw])) for p in players])), 2) for gw in future_gameweeks}
            
            # Update total_xp attribute
            self.total_xp = round(value(lpSum([self.gw_xp[gw] for gw in future_gameweeks])), 2)
            
            return self.results
        
   
            
    def check_results(self, data):
        # Dataframes
        element_types_df = data["elements_types_df"]
        
        # Lists
        future_gameweeks = data["future_gameweeks"]
        
        # Set up dictionary to store results of checks
        checks_dict = {} # True if all checks are passed, False otherwise
        
        if self.results is None:
            print("WARNING: No results available to check.")
            return None
        else:
            for gw in future_gameweeks:
                condition_1 = self.results[self.results["gw"] == gw].squad.sum() == 15
                condition_2 = self.results[self.results["gw"] == gw].lineup.sum() == 11
                condition_3 = self.results[self.results["gw"] == gw].transfer_in.sum() == self.results[self.results["gw"] == gw].transfer_out.sum()
                condition_4 = self.results[(self.results["gw"] == gw) & (self.results["squad"] == 1)].team.value_counts().max() <= 3
                condition_5 = all(self.results[self.results["gw"] == gw].groupby("position_id").squad.sum() == element_types_df["squad_select"])
                condition_6a = all(self.results[self.results["gw"] == gw].groupby("position_id").lineup.sum() >= element_types_df["squad_min_play"])
                condition_6b = all(self.results[self.results["gw"] == gw].groupby("position_id").lineup.sum() <= element_types_df["squad_max_play"])
                condition_5 = True
                condition_6a = True
                condition_6b = True
                condition_7 = all(self.results[(self.results["gw"] == gw) & (self.results["squad"] == 1)].prob_appearance > 0.5)
                condition_8 = all(self.results[(self.results["gw"] == gw) & (self.results["lineup"] == 1)].prob_appearance > 0.75)
                condition_9 = self.results[self.results["gw"] == gw].captain.sum() == 1
                condition_10 = self.results[self.results["gw"] == gw].vice_captain.sum() == 1
                condition_11 = all(self.results[(self.results["gw"] == gw) & (self.results["captain"] == 1)].lineup == 1)
                condition_12 = all(self.results[(self.results["gw"] == gw) & (self.results["vice_captain"] == 1)].lineup == 1)

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
                print("Results passed checks.")
            
            # Update checks attribute
            self.checks = checks_dict
            
            return self.checks


    def solve_problem(self):
        # Get data
        data = self.prepare_data()
   
        # Define problem
        self.define_problem()
        
        # Define problem variables
        variables = self.define_variables(data)

        # Define dictionaries
        dictionaries = self.define_dictionaries(data, variables)
        
        # Define initial conditions
        self.define_initial_conditions(data, variables)
        
        # Define constraints
        self.define_constraints(data, variables, dictionaries)
        
        # Define objective
        self.define_objective(data, variables, dictionaries)
        
        # Solve problem
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if self.model.status != 1:
            print("Model could not solved.")
            print("Status:", self.model.status)
            return None
        else:
            print("Model solved.")
            print("Status:", self.model.status)
            print("Time:", round(self.model.solutionTime, 2))
            
        # Extract results
        self.extract_results(data, variables, dictionaries)
        
        # Check results
        self.check_results(data)
        
        return self.model


if __name__ == "__main__":
    
    # Initialize optimizer
    optimizer = OptimizeMultiPeriod(team_id=1, gameweek=22, bank_balance=4.2, num_free_transfers=1, horizon=3)
    
    # Solve problem
    optimizer.solve_problem()

    # Print results
    print(optimizer.results)
    print(optimizer.total_xp)
    print(optimizer.gw_xp)
    print(optimizer.checks)
    
    # Print modelname
    print(optimizer.model.name)
