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
        
        """
        Summary:
        --------
        Class to solve the multi-period optimization problem for Fantasy Premier League.
        
        Parameters:
        -----------
        team_id: int
            Team ID of the team to optimize.
        gameweek: int
            Gameweek to optimize for.
        bank_balance: float
            Bank balance at the start of the gameweek (currently)
        num_free_transfers: int
            Number of free transfers available currently.
        horizon: int
            Number of gameweeks to optimize for.
        objective: str
            Objective function to optimize for. Can be either "regular" or "decay".
        decay_base: float
            Decay base to use for "decay" objective function.
        """
        
        # Set attributes
        self.team_id = team_id
        self.gameweek = gameweek
        self.bank_balance = bank_balance
        self.num_free_transfers = num_free_transfers
        self.horizon = horizon
        self.objective = objective
        self.decay_base = decay_base
        
        
    def solve_problem(self):
        self.get_data()
        self.set_problem()
        self.set_variables()
        self.get_dictionaries()
        self.set_initial_conditions()
        self.set_constraints()
        self.set_objective(objective=self.objective)
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if self.model.status != 1:
            print("Model could not solved.")
            print("Status:", self.model.status)
            return None
        else:
            print("Model solved.")
            print("Status:", self.model.status)
            print("Time:", round(self.model.solutionTime, 2))
            
        self.get_results()
        self.check_results(results=self.results)
        self.get_summary()
        
        return {"model": self.model, "results": self.results, "summary": self.summary, "total_xp": self.total_xp, "gw_xp": self.gw_xp, "checks": self.checks}    
        
    
    def get_data(self):
        """
        Summary:
        --------
        Function to prepare data for the optimization problem. Pulls general data and initial squad from FPL API, and merges with FPL form data.
        
        Returns:
        --------
        Dictionary with the following keys:
            - merged_elements_df: Dataframe with merged data from FPL API and FPL form data.
            - element_types_df: Dataframe with element types (position) data.
            - teams_df: Dataframe with teams data.
            - players: List of player IDs.
            - positions: List of position IDs.
            - teams: List of team IDs.
            - future_gameweeks: List of future gameweeks to optimize for.
            - all_gameweeks: List of all gameweeks (current gameweek + future gameweeks)
            - initial_squad: List of player IDs in initial squad.
        """
        
        # Pull data from FPL API
        data = pull_general_data()
        # self.initial_squad = pull_squad(self.team_id)
        self.initial_squad = [275, 369, 342, 506, 19, 526, 664, 14, 117, 60, 343, 230, 129, 112, 126]
        
        # Get dataframes from dictionary
        self.elements_df = data["elements"]
        self.element_types_df = data["element_types"]
        self.teams_df = data["teams"]
        
        # Merge elements_df with form data
        form_data = pd.read_csv("../../data/raw/fpl-form-predicted-points.csv")
        self.merged_elements_df = merge_fpl_form_data(self.elements_df, form_data)
        
        # Set index for dataframes
        self.merged_elements_df.set_index("id", inplace=True)
        self.element_types_df.set_index("id", inplace=True)
        self.teams_df.set_index("id", inplace=True)
        
        # Get lists of players, positions, teams, and gameweeks
        self.players = self.merged_elements_df.index.tolist()
        self.positions = self.element_types_df.index.tolist()
        self.teams = self.teams_df.index.tolist()
        self.future_gameweeks = list(range(self.gameweek, self.gameweek + self.horizon))
        self.all_gameweeks = [self.gameweek - 1] + self.future_gameweeks
        
        return {"merged_elements_df": self.merged_elements_df, "element_types": self.element_types_df, "teams_df": self.teams_df, "players": self.players, "positions": self.positions, 
                "teams": self.teams, "future_gameweeks": self.future_gameweeks, "all_gameweeks": self.all_gameweeks, "initial_squad": self.initial_squad}
        
    
    def set_problem(self):
        """
        Summary:
        --------
        Function to define the optimization problem.

        Returns:
        --------
        PuLP model object.
        """
        name = f"MultiPeriodOptimization_team_{self.team_id}_gw_{self.gameweek}_horizon_{self.horizon}_objective_{self.objective}"
        self.model = LpProblem(name, LpMaximize)
        
        return self.model
    
    
    def set_variables(self):
        """
        Summary:
        --------
        Function to define the optimization variables. 
        
        Returns:
        --------
        Dictionary with the following keys:
            - squad: Binary variable for whether player is in squad in each gameweek.
            - lineup: Binary variable for whether player is in lineup in each gameweek.
            - captain: Binary variable for whether player is captain in each gameweek.
            - vice_captain: Binary variable for whether player is vice captain in each gameweek.
            - transfer_in: Binary variable for whether player is transferred in in each gameweek.
            - transfer_out: Binary variable for whether player is transferred out in each gameweek.
            - money_in_bank: Continuous variable for money in bank in each gameweek.
            - free_transfers_available: Integer variable for number of free transfers available in each gameweek.
            - penalised_transfers: Integer variable for number of penalised transfers in each gameweek.
            - aux: Binary variable for auxiliary variable in each gameweek.
        """
        
        self.squad = LpVariable.dicts("squad", (self.players, self.all_gameweeks), cat="Binary")
        self.lineup = LpVariable.dicts("lineup", (self.players, self.future_gameweeks), cat="Binary")
        self.captain = LpVariable.dicts("captain", (self.players, self.future_gameweeks), cat="Binary")
        self.vice_captain = LpVariable.dicts("vice_captain", (self.players, self.future_gameweeks), cat="Binary")
        self.transfer_in = LpVariable.dicts("transfer_in", (self.players, self.future_gameweeks), cat="Binary")
        self.transfer_out = LpVariable.dicts("transfer_out", (self.players, self.future_gameweeks), cat="Binary")
        self.money_in_bank = LpVariable.dicts("money_in_bank", (self.all_gameweeks), lowBound=0, cat="Continuous")
        self.free_transfers_available = LpVariable.dicts("free_transfers_available", (self.all_gameweeks), lowBound=1, upBound=2, cat="Integer")
        self.penalised_transfers = LpVariable.dicts("penalised_transfers", (self.future_gameweeks), cat="Integer", lowBound=0)
        self.aux = LpVariable.dicts("auxiliary_variable", (self.future_gameweeks), cat="Binary")
        
        return {"squad": self.squad, "lineup": self.lineup, "captain": self.captain, "vice_captain": self.vice_captain, "transfer_in": self.transfer_in, "transfer_out": self.transfer_out, 
                "money_in_bank": self.money_in_bank, "free_transfers_available": self.free_transfers_available, "penalised_transfers": self.penalised_transfers, "aux": self.aux}
        
    def get_dictionaries(self):
        """
        Summary:
        --------
        Function to define the optimization dictionaries. These are used to help define the constraints and objective function.
        
        Returns:
        --------
        Dictionary with the following keys:
            - player_cost: Dictionary with player cost for each player.
            - player_xp_gw: Dictionary with expected points for each player in each gameweek.
            - player_prob_gw: Dictionary with probability of appearance for each player in each gameweek.
            - squad_count: Dictionary with number of players in squad in each gameweek.
            - lineup_count: Dictionary with number of players in lineup in each gameweek.
            - lineup_position_count: Dictionary with number of players in each position in lineup in each gameweek.
            - squad_position_count: Dictionary with number of players in each position in squad in each gameweek.
            - squad_team_count: Dictionary with number of players from each team in squad in each gameweek.
            - revenue: Dictionary with transfer revenue in each gameweek.
            - expenditure: Dictionary with transfer expenditure in each gameweek.
            - transfers_made: Dictionary with number of transfers made in each gameweek.
            - transfer_diff: Dictionary with difference between transfers made and free transfers available in each gameweek.
        """
 
        self.player_cost = self.merged_elements_df["now_cost"].to_dict()
        self.player_xp_gw = {(p, gw): self.merged_elements_df.loc[p, f"{gw}_pts_no_prob"] for p in self.players for gw in self.future_gameweeks}
        self.player_prob_gw = {(p, gw): self.merged_elements_df.loc[p, f"{gw}_prob"] for p in self.players for gw in self.future_gameweeks}
        self.squad_count = {gw: lpSum([self.squad[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.lineup_count = {gw: lpSum([self.lineup[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.lineup_position_count = {(pos, gw): lpSum([self.lineup[p][gw] for p in self.players if self.merged_elements_df.loc[p, "element_type"] == pos]) for pos in self.positions for gw in self.future_gameweeks}
        self.squad_position_count = {(pos, gw): lpSum([self.squad[p][gw] for p in self.players if self.merged_elements_df.loc[p, "element_type"] == pos]) for pos in self.positions for gw in self.future_gameweeks}
        self.squad_team_count = {(team, gw): lpSum([self.squad[p][gw] for p in self.players if self.merged_elements_df.loc[p, "team"] == team]) for team in self.teams for gw in self.future_gameweeks}
        self.revenue = {gw: lpSum([self.player_cost[p] * self.transfer_out[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.expenditure = {gw: lpSum([self.player_cost[p] * self.transfer_in[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.transfers_made = {gw: lpSum([self.transfer_in[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.transfers_made[self.gameweek - 1] = 1
        self.transfer_diff = {gw: (self.transfers_made[gw] - self.free_transfers_available[gw]) for gw in self.future_gameweeks}
        
        return {"player_cost": self.player_cost, "player_xp_gw": self.player_xp_gw, "player_prob_gw": self.player_prob_gw, "squad_count": self.squad_count, "lineup_count": self.lineup_count, 
                "lineup_position_count": self.lineup_position_count, "squad_position_count": self.squad_position_count, "squad_team_count": self.squad_team_count, "revenue": self.revenue, 
                "expenditure": self.expenditure, "transfers_made": self.transfers_made, "transfer_diff": self.transfer_diff}
        
    
    def set_initial_conditions(self):
        """
        Summary:
        --------
        Function to define the initial conditions for the optimization problem. These are the conditions that must be satisfied at the start of the gameweek.
        Initial conditions are added to model as constraints by using the += operator.
        Note that constraints, and therefore initial conditions, must include conditional operators (<=, >=, ==).
        """
        # Players in initial squad must be in squad in current gameweek
        for p in [player for player in self.players if player in self.initial_squad]:
            self.model += self.squad[p][self.gameweek - 1] == 1, f"In initial squad constraint for player {p}"

        # Players not in initial squad must not be in squad in current gameweek
        for p in [player for player in self.players if player not in self.initial_squad]:
            self.model += self.squad[p][self.gameweek - 1] == 0, f"Not initial squad constraint for player {p}"
            
        # Money in bank at current gameweek must be equal to bank balance
        self.model += self.money_in_bank[self.gameweek - 1] == self.bank_balance, f"Initial money in bank constraint"

        # Number of free transfers available in current gameweek must be equal to num_free_transfers
        self.model += self.free_transfers_available[self.gameweek - 1] == self.num_free_transfers, f"Initial free transfers available constraint"
            
        
    def set_constraints(self):
        """
        Summary:
        --------
        Function to define the constraints for the optimization problem. 
        Constraints must include conditional operators (<=, >=, ==), and are added to model by using the += operator.
        
        Constraints are added to model by using the += operator 
        
        Constraints include:
            - Squad and lineup constraints
            - Captain and vice captain constraints
            - Position / Formation constraints
            - Team played for constraints
            - Probability of appearance constraints
            - Budgeting / Financial constraints
            - General transfer constraints
            - Free transfer constraints
        """
        
        # ----------------------------------------
        # Squad and lineup constraints
        # ----------------------------------------
        
        for gw in self.future_gameweeks:
            # Total number of players in squad in each gameweek must be equal to 15
            self.model += self.squad_count[gw] == 15, f"Squad count constraint for gameweek {gw}"

            # Total number of players in lineup in each gameweek must be equal to 11
            self.model += self.lineup_count[gw] == 11, f"Lineup count constraint for gameweek {gw}"

            # Lineup player must be in squad (but reverse can not be true) in each gameweek
            for p in self.players:
                self.model += self.lineup[p][gw] <= self.squad[p][gw], f"Lineup player must be in squad constraint for player {p} in gameweek {gw}"

        # ----------------------------------------
        # Captain and vice captain constraints
        # ----------------------------------------

        for gw in self.future_gameweeks:
            # Only 1 captain in each gameweek
            self.model += lpSum([self.captain[p][gw] for p in self.players]) == 1, f"Captain count constraint for gameweek {gw}"
            
            # Only 1 vice captain in each gameweek
            self.model += lpSum([self.vice_captain[p][gw] for p in self.players]) == 1, f"Vice captain count constraint for gameweek {gw}"

            # Captain must be in lineup in each gameweek
            for p in self.players:
                self.model += self.captain[p][gw] <= self.lineup[p][gw], f"Captain must be in lineup constraint for player {p} in gameweek {gw}"
            
            # Vice captain must be in lineup in each gameweek
            for p in self.players:
                self.model += self.vice_captain[p][gw] <= self.lineup[p][gw], f"Vice captain must be in lineup constraint for player {p} in gameweek {gw}"

            # Captain and vice captain can not be the same player in each gameweek
            for p in self.players:
                self.model += self.captain[p][gw] + self.vice_captain[p][gw] <= 1, f"Captain and vice captain can not be the same player constraint for player {p} in gameweek {gw}"
                
        # ----------------------------------------
        # Position / Formation constraints
        # ----------------------------------------
        
        for gw in self.future_gameweeks:
            for pos in self.positions:
                # Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for every gameweek
                self.model += (self.lineup_position_count[pos, gw] >= self.element_types_df.loc[pos, "squad_min_play"]), f"Min lineup players in position {pos} in gameweek {gw}"
                self.model += (self.lineup_position_count[pos, gw] <= self.element_types_df.loc[pos, "squad_max_play"]), f"Max lineup players in position {pos} in gameweek {gw}"

                # Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select) for every gameweek
                self.model += (self.squad_position_count[pos, gw] == self.element_types_df.loc[pos, "squad_select"]), f"Squad players in position {pos} in gameweek {gw}"

        # ----------------------------------------
        # Team played for constraints
        # ----------------------------------------

        # Number of players in each team in squad must be less than or equal to 3 for every gameweek
        for gw in self.future_gameweeks:
            for team in self.teams:
                self.model += (self.squad_team_count[team, gw] <= 3), f"Max players from team {team} in gameweek {gw}"

        # ----------------------------------------
        # Probability of appearance constraints
        # ----------------------------------------

        # For every gameweek the probability of squad player appearing in next gameweek must be >= 50%, while probability of lineup player > 75%
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += self.squad[p][gw] <= (self.player_prob_gw[p, gw] >= 0.5), f"Probability of appearance for squad player {p} for gameweek {gw}"
                self.model += self.lineup[p][gw] <= (self.player_prob_gw[p, gw] >= 0.75), f"Probability of appearance for lineup player {p} for gameweek {gw}"
            
        # ----------------------------------------
        # Budgeting / Financial constraints
        # ----------------------------------------

        # Money in bank in each gameweek must be equal to previous gameweek money in bank plus transfer revenue minus transfer expenditure
        for gw in self.future_gameweeks:
            self.model += (self.money_in_bank[gw] == (self.money_in_bank[gw - 1] + self.revenue[gw] - self.expenditure[gw])), f"Money in bank constraint for gameweek {gw}"

        # ----------------------------------------
        # General transfer constraints
        # ----------------------------------------

        for gw in self.future_gameweeks:
            # Players in next gameweek squad must either be in current gameweek squad or transferred in
            # And players not in next gameweek squad must be transferred out
            for p in self.players:
                self.model += (self.squad[p][gw] == (self.squad[p][gw - 1] + self.transfer_in[p][gw] - self.transfer_out[p][gw])), f"Player {p} squad/transfer constraint for gameweek {gw}"

            # Number of transfers made in each gameweek cannot exceed 5
            self.model += self.transfers_made[gw] <= 20, f"Transfers made constraint for gameweek {gw}"
            
        # ----------------------------------------
        # Free transfer constraints
        # ----------------------------------------

        for gw in self.future_gameweeks:
            # Free transfers available and auxiliary variable conditions for each gameweek
            self.model += (self.free_transfers_available[gw] == (self.aux[gw] + 1)), f"FTA and Aux constraint for gameweek {gw}"
            
            # Equality 1: FTA_{1} - TM_{1} <= 2 * Aux_{2}
            self.model += self.free_transfers_available[gw - 1] - self.transfers_made[gw - 1] <= 2 * self.aux[gw], f"FTA and TM Equality 1 constraint for gameweek {gw}"
            
            # Equality 2: FTA_{1} - TM_{1} >= Aux_{2} + (-14) * (1 - Aux_{2})
            self.model += self.free_transfers_available[gw - 1] - self.transfers_made[gw - 1] >= self.aux[gw] + (-14) * (1 - self.aux[gw]), f"FTA and TM Equality 2 constraint for gameweek {gw}"

            # Number of penalised transfers in each gameweek must be equal to or greater than the transfer difference (i.e. number of transfers made minus number of free transfers available)
            # I.e. only penalise transfers if we have made more transfers than allowed
            self.model += self.penalised_transfers[gw] >= self.transfer_diff[gw], f"Penalised transfers constraint for gameweek {gw}"
        
        
    def set_objective(self, objective: str):
        """
        Summary:
        --------
        Function to define the objective function for the optimization problem.
        Objective function is added to model by using the += operator.
        
        Objective functions include:
            - Regular: Maximize total expected points over all gameweeks 
            - Decay: Maximize total expected points in each gameweek, with decay factor
            
        Args:
        ----------
        objective: str
            Objective function to optimize for. Can be either "regular" or "decay".
        """
        
        gw_xp_before_pen = {gw: lpSum([self.player_xp_gw[p, gw] * (self.lineup[p][gw] + self.captain[p][gw] + 0.1 * self.vice_captain[p][gw]) for p in self.players]) for gw in self.future_gameweeks}
        gw_xp_after_pen = {gw: gw_xp_before_pen[gw] - 4 * self.penalised_transfers[gw] for gw in self.future_gameweeks}
        
        
        if objective == "regular":
            total_xp = lpSum([gw_xp_after_pen [gw] for gw in self.future_gameweeks])
            self.model += total_xp
            
        elif objective == "decay":
            total_xp = lpSum([gw_xp_after_pen[gw] * pow(self.decay_base, gw - self.gameweek) for gw in self.future_gameweeks])
            self.model += total_xp
            model_name += "_decay_base_" + str(self.decay_base)
            self.model.name = model_name
            
    
    def get_results(self):
        """
        Summary:
        --------
        Function to extract results from the optimization problem.
        
        Returns:
        --------
        Dictionary with the following keys:
            - model: PuLP model object (solved).
            - dataframe: Dataframe with results.
            - total_xp: Total expected points.
            - gw_xp: Dictionary with expected points for each gameweek.
        """
        
        if self.model.status != 1:
            print("Cannot extract results available since model is not solved.")
            return None
        else:
            results = []
            
            for gw in self.future_gameweeks:
                for p in self.players:
                    if self.squad[p][gw].varValue == 1 or self.transfer_out[p][gw].varValue == 1:
                        results.append(
                            {   
                                "gw": gw,
                                "player_id": p,
                                "player_name": self.merged_elements_df.loc[p, "web_name"],
                                "team": self.merged_elements_df.loc[p, "team_name"],
                                "position": self.merged_elements_df.loc[p, "position"],
                                "position_id": self.merged_elements_df.loc[p, "element_type"],
                                "cost": self.player_cost[p],
                                "prob_appearance": self.player_prob_gw[p, gw],
                                "xp": self.player_xp_gw[p, gw],
                                "squad": self.squad[p][gw].varValue,
                                "lineup": self.lineup[p][gw].varValue,
                                "captain": self.captain[p][gw].varValue,
                                "vice_captain": self.vice_captain[p][gw].varValue,
                                "transfer_in": self.transfer_in[p][gw].varValue,
                                "transfer_out": self.transfer_out[p][gw].varValue,
                            }
                        )
                        
            # Convert results to dataframe
            results_df = pd.DataFrame(results).round(2)
            
            # Sort results and reset index
            results_df.sort_values(by=["gw", "squad", "lineup", "position_id", "xp"], ascending=[True, False, False, True, False], inplace=True)
            results_df.reset_index(drop=True, inplace=True)
            
            # Update attributes
            self.results = results_df
            self.gw_xp = {gw: round(value(lpSum([self.player_xp_gw[p, gw] * (self.lineup[p][gw] + self.captain[p][gw] - (4 * self.penalised_transfers[gw])) for p in self.players])), 2) for gw in self.future_gameweeks}
            self.total_xp = round(value(lpSum([self.gw_xp[gw] for gw in self.future_gameweeks])), 2)
            
            return {"model": self.model, "dataframe": self.results, "total_xp": self.total_xp, "gw_xp": self.gw_xp}
        
   
    def check_results(self, results: pd.DataFrame):
        """
        Summary:
        --------
        Function to check results of the optimization problem. Checks include:
            - Number of players in squad for each gameweek is 15
            - Number of players in lineup for each gameweek is 11
            - Number of transfers in is equal to number of transfers out for each gameweek
            - Number of players from each team in squad is less than or equal to 3 for each gameweek
            - Number of players in each position in squad is equal to squad_select (defined in element_types_df) for each gameweek
            - Number of players in each position in lineup is greater than the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for each gameweek
            - Number of players in each position in lineup is less than the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for each gameweek
            - Probability of appearance for each player in squad is greater than 50% for each gameweek
            - Probability of appearance for each player in lineup is greater than 75% for each gameweek
            - Number of captains is equal to 1 for each gameweek
            - Number of vice captains is equal to 1 for each gameweek
            - Captain is in lineup for each gameweek
            - Vice captain is in lineup for each gameweek
        
        If all checks are passed, prints "Results passed checks".
        

        Args:
        --------
        Results (pd.DataFrame): Dataframe with results from optimization problem.

        Returns:
        --------
        Dictionary with results of checks for each gameweek (True if all checks are passed, False otherwise)
        """
        checks_dict = {}

        if results is None:
            print("WARNING: No results available to check.")
            return None
        else:
            for gw in self.future_gameweeks:
                gw_results = results[results["gw"] == gw]

                condition_1 = gw_results.squad.sum() == 15
                condition_2 = gw_results.lineup.sum() == 11
                condition_3 = gw_results.transfer_in.sum() == gw_results.transfer_out.sum()
                condition_4 = gw_results[gw_results["squad"] == 1].team.value_counts().max() <= 3
                condition_5 = all(gw_results.groupby("position_id").squad.sum() == self.element_types_df["squad_select"])
                condition_6a = all(gw_results.groupby("position_id").lineup.sum() >= self.element_types_df["squad_min_play"])
                condition_6b = all(gw_results.groupby("position_id").lineup.sum() <= self.element_types_df["squad_max_play"])
                condition_7 = all(gw_results[gw_results["squad"] == 1].prob_appearance > 0.5)
                condition_8 = all(gw_results[gw_results["lineup"] == 1].prob_appearance > 0.75)
                condition_9 = gw_results.captain.sum() == 1
                condition_10 = gw_results.vice_captain.sum() == 1
                condition_11 = gw_results[gw_results["captain"] == 1].lineup.sum() == 1
                condition_12 = gw_results[gw_results["vice_captain"] == 1].lineup.sum() == 1
                
                if all(
                    [
                        condition_1,
                        condition_2,
                        condition_3,
                        condition_4,
                        condition_5,
                        condition_6a,
                        condition_6b,
                        condition_7,
                        condition_8,
                        condition_9,
                        condition_10,
                        condition_11,
                        condition_12,
                    ]
                ):
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

            if all(value for value in checks_dict.values()):
                print("Results passed checks.")

            self.checks = checks_dict

            return self.checks
        
        
    def get_summary(self):
        """
        Summary:
        --------
        Function to extract summary of actions over all gameweeks from the optimization problem.
        For example, which players are transferred in/out, how much money is in the bank, how many free transfers are available, etc.
        
        Returns:
        --------
        String with summary of actions over all gameweeks.
        """
        # Initialize summary list
        summary_list = []
        
        for gw in self.future_gameweeks:
            summary_list.append("-" * 50)
            summary_list.append(f"Gameweek {gw} summary:")
            summary_list.append("-" * 50)
            
            gw_xp = self.gw_xp[gw]
            money_in_bank = self.money_in_bank[gw].varValue
            free_transfers_available = int(self.free_transfers_available[gw].varValue)
            transfers_made = int(value(self.transfers_made[gw]))
            penalised_transfers = int(self.penalised_transfers[gw].varValue)
            
            summary_list.append(f"Total expected points: {gw_xp}")
            summary_list.append(f"Money in bank: {money_in_bank}")
            summary_list.append(f"Free transfers available: {free_transfers_available}")
            summary_list.append(f"Transfers made: {transfers_made}")
            summary_list.append(f"Penalised transfers: {penalised_transfers}")
            
            for p in self.players:
                if self.transfer_in[p][gw].varValue == 1:
                    web_name = self.merged_elements_df.loc[p, 'web_name']
                    team_name = self.merged_elements_df.loc[p, 'team_name']
                    summary_list.append(f"Player {p} ({web_name} @ {team_name}) transferred in.")
                if self.transfer_out[p][gw].varValue == 1:
                    web_name = self.merged_elements_df.loc[p, 'web_name']
                    team_name = self.merged_elements_df.loc[p, 'team_name']
                    summary_list.append(f"Player {p} ({web_name} @ {team_name}) transferred out.")
        
        # Join the summary list into a single string
        summary = "\n".join(summary_list)
        
        # Update summary attribute
        self.summary = summary
        
        return self.summary
        

if __name__ == "__main__":
    
    horizons = [1, 2, 3, 4, 5]
    
    for h in horizons:
        
        print(f"Optimizing for horizon {h}...")
        
        optimizer = OptimizeMultiPeriod(team_id=1, gameweek=22, bank_balance=4.2, num_free_transfers=1, horizon=h)
        optimizer.solve_problem()
        
        r = optimizer.get_results()
        s = optimizer.get_summary()
        n = optimizer.model.name
        
        r["model"].writeLP(f"../../models/multi_period/{n}_model.lp")
        r["dataframe"].to_csv(f"../../models/multi_period/{n}_results.csv", index=False)
        
        with open(f"../../models/multi_period/{n}_summary.txt", "w") as f:
            f.write(s)

        

