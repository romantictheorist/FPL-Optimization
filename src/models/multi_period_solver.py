# ----------------------------------------
# Imports
# ----------------------------------------

import numpy as np
import pandas as pd
import pulp
from pulp import *
import time
import multiprocessing
from datetime import datetime
import os
from pulp import value
import sys
sys.path.append("..")

from data.get_data import FPLDataPuller, FPLFormScraper
from features.build_features import ProcessData

pd.options.mode.chained_assignment = None  # default='warn'

    
class OptimizeMultiPeriod(FPLDataPuller, FPLFormScraper, ProcessData):
    # Class variable to store the data pulled from the FPL API and FPLForm.com
    _cached_data = None
    
    def __init__(self, team_id: int, gameweek: int, num_free_transfers: int, horizon: int, objective='regular', decay_base=0.85):
        self.team_id = team_id
        self.gameweek = gameweek
        self.num_free_transfers = num_free_transfers
        self.horizon = horizon
        self.objective = objective
        self.decay_base = decay_base
        self.future_gameweeks = list(range(self.gameweek, self.gameweek + self.horizon))
        self.all_gameweeks = [self.gameweek - 1] + list(range(self.gameweek, self.gameweek + self.horizon))
        
        # Fecth the data only if it hasn't been fetched yet
        if OptimizeMultiPeriod._cached_data is None:
            print("Fetching data...")
            
            # Fetch the raw data
            data = self._get_data(team_id=self.team_id) 
            # Get the current gameweek
            current_gw = data["current_gw"]
            # List of keys to save
            data_to_save = ['gameweek_df', 'positions_df', 'teams_df', 'predicted_points_df']
            # Save raw data
            self._save_data(data={key: data[key] for key in data_to_save if key in data}, destination=f"../../data/raw/gw_{current_gw}/")
            # Process the raw data
            processed_data = self._process_data(data=data)
            # Save the processed data
            self._save_data(data={'merged_df': processed_data['merged_df']}, destination=f"../../data/processed/gw_{current_gw}/")
            # Set the class variable to the processed data
            OptimizeMultiPeriod._cached_data = processed_data
        
        # Unpack the needed (processed) data from the class variable into instance variables
        self.merged_df = self._cached_data["merged_df"]
        self.positions_df = self._cached_data["positions_df"]
        self.teams_df = self._cached_data["teams_df"]
        self.current_gw = self._cached_data["current_gw"]
        self.initial_squad = self._cached_data["initial_squad"]
        self.bank_balance = self._cached_data["bank_balance"]
        self.players = self._cached_data["players"]
        self.positions = self._cached_data["positions"]
        self.teams = self._cached_data["teams"]

    
    def solve_problem(self):
        
        # Step 1: Check if the gameweek is valid
        self._check_gameweek
        
        # Step 2: Set the optimization problem
        self._set_problem()
        
        # Step 3: Set the optimization variables for the problem
        self._set_variables()
        
        # Step 4: Set the optimization dictionaries (i.e. player cost, expected points, probability of appearance, etc.)
        self._set_dictionaries()
        
        # Step 5: Set the initial conditions for the optimization problem
        self._set_initial_conditions()
        
        # Step 6: Set the constraints for the optimization problem
        self._set_constraints()
        
        # Step 7: Set the objective function for the optimization problem
        self._set_objective()
    
        # Step 8: Solve the optimization problem
        print("Solving model...")
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if self.model.status != 1:
            print("Model could not solved.")
            print("Status:", self.model.status)
            return None
        else:
            print("Model solved.")
            print("Time:", round(self.model.solutionTime, 2))
            print("Objective:", round(self.model.objective.value(), 2))
          
        # Step 9: Get the results of the optimization problem  
        self.get_results()
        
        # Step 10: Check the results of the optimization problem
        self._check_results(results=self.results_df)
        

    def _get_data(self, team_id: int):
        
        general_data = self._get_general_data()
        gameweek_df = general_data["gameweek"]
        positions_df = general_data["positions"]
        teams_df = general_data["teams"]
        current_gw = general_data["current_gw"]
        
        initial_squad = self._get_team_ids(team_id=team_id, gameweek=current_gw)
        bank_balance = self._get_team_data(team_id=team_id)["bank_balance"]
        
        predicted_points_df = self._get_predicted_points()
        
        data = {
            "gameweek_df": gameweek_df,
            "positions_df": positions_df,
            "teams_df": teams_df,
            "predicted_points_df": predicted_points_df,
            "current_gw": current_gw,
            "initial_squad": initial_squad,
            "bank_balance": bank_balance
        }
        
        return data
    
    def _process_data(self, data: dict):
        gameweek_df = self._process_gameweek(gameweek_df=data["gameweek_df"])
        gameweek_df = self._map_teams_to_gameweek(gameweek_df=gameweek_df, teams_df=data["teams_df"])
        predicted_points_df = self._process_predicted_points(predicted_points_df=data["predicted_points_df"])
        merged_df = self._merge_gameweek_and_predicted_points(gameweek_df=gameweek_df, predicted_points_df=predicted_points_df)
        
        # Set index as IDs for needed dataframes
        for df in [merged_df, data["positions_df"], data["teams_df"]]:
            df.set_index("id", inplace=True)
            
        players = merged_df.index.tolist()
        positions = data["positions_df"].index.tolist()
        teams = data["teams_df"].index.tolist()
            
        # Update data dictionary
        data["merged_df"] = merged_df
        data["gameweek_df"] = gameweek_df
        data["predicted_points_df"] = predicted_points_df
        data["players"] = players
        data["positions"] = positions
        data["teams"] = teams
        
        return data
    
    
    def _save_data(self, data: dict, destination: str):
        if not os.path.exists(destination):
            os.makedirs(destination)
            
        for key, value in data.items():
            # If data is predicted points, save with current date
            if key != "predicted_points_df":
                value.to_csv(f"{destination}{key}.csv", index=False)
            else:
                current_date = datetime.today().strftime("%Y_%m_%d")
                value.to_csv(f"{destination}{key}_{current_date}.csv", index=False)
                
         
    def _check_gameweek(self):
        """
        Summary:
        --------
        Function to if the gameweek is valid..
        """
        
         # Check if gameweek is valid
        if self.gameweek > self.current_gw + 1:
            raise ValueError(f"Cannot optimize for GW{self.gameweek}. Optimization can only start from the next gameweek (i.e. GW{self.current_gw + 1}).")
        elif self.gameweek < self.current_gw + 1:
            raise ValueError(f"Cannot optimize for GW{self.gameweek} since it has already passed. Optimization can only start from the next gameweek (i.e. GW{self.current_gw + 1}).")
        else:
            pass


    def _set_problem(self):
        """
        Summary:
        --------
        Function to define the optimization problem.

        Returns:
        --------
        
        """
        name = f"MultiPeriodOptimization_team_{self.team_id}_gw_{self.gameweek}_horizon_{self.horizon}_objective_{self.objective}"
        self.model = LpProblem(name, LpMaximize)
    
        
    def _set_variables(self):
        """
        Summary:
        --------
        Function to set the optimization variables for the problem.
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
        
    
    def _set_dictionaries(self):
        """
        Summary:
        --------
        Function to define the optimization dictionaries. These are used to help define the constraints and objective function.
        
        Dictionaries include:
            - Player cost
            - Player expected points for each gameweek
            - Player probability of appearing for each gameweek
            - Squad count for each gameweek
            - Lineup count for each gameweek
            - Lineup position count for each gameweek
            - Squad position count for each gameweek
            - Squad team count for each gameweek
            - Revenue for each gameweek
            - Expenditure for each gam
            - Transfers made for each gameweek
            - Transfer difference for each gameweek (i.e. number of transfers made minus number of free transfers available)
            
        """
         
        self.player_cost = self.merged_df["cost"].to_dict()
        self.player_xp_gw = {(p, gw): self.merged_df.loc[p, f"gw_{gw}_xp"] for p in self.players for gw in self.future_gameweeks}
        self.player_prob_gw = {(p, gw): self.merged_df.loc[p, f"gw_{gw}_prob_of_appearing"] for p in self.players for gw in self.future_gameweeks}
        self.squad_count = {gw: lpSum([self.squad[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.lineup_count = {gw: lpSum([self.lineup[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.lineup_position_count = {(pos, gw): lpSum([self.lineup[p][gw] for p in self.players if self.merged_df.loc[p, "element_type"] == pos]) for pos in self.positions for gw in self.future_gameweeks}
        self.squad_position_count = {(pos, gw): lpSum([self.squad[p][gw] for p in self.players if self.merged_df.loc[p, "element_type"] == pos]) for pos in self.positions for gw in self.future_gameweeks}
        self.squad_team_count = {(team, gw): lpSum([self.squad[p][gw] for p in self.players if self.merged_df.loc[p, "team_id"] == team]) for team in self.teams for gw in self.future_gameweeks}
        self.revenue = {gw: lpSum([self.player_cost[p] * self.transfer_out[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.expenditure = {gw: lpSum([self.player_cost[p] * self.transfer_in[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.transfers_made = {gw: lpSum([self.transfer_in[p][gw] for p in self.players]) for gw in self.future_gameweeks}
        self.transfers_made[self.gameweek - 1] = 1
        self.transfer_diff = {gw: (self.transfers_made[gw] - self.free_transfers_available[gw]) for gw in self.future_gameweeks}
        
   
        
    def _set_initial_conditions(self):
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
            
        
    def _set_constraints(self):
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
                # Number of players in each position in lineup must be within the allowed range (defined in positions_df as squad_min_play and squad_max_play) for every gameweek
                self.model += (self.lineup_position_count[pos, gw] >= self.positions_df.loc[pos, "squad_min_play"]), f"Min lineup players in position {pos} in gameweek {gw}"
                self.model += (self.lineup_position_count[pos, gw] <= self.positions_df.loc[pos, "squad_max_play"]), f"Max lineup players in position {pos} in gameweek {gw}"

                # Number of players in each position in squad must be satisfied (defined in positions_df as squad_select) for every gameweek
                self.model += (self.squad_position_count[pos, gw] == self.positions_df.loc[pos, "squad_select"]), f"Squad players in position {pos} in gameweek {gw}"

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

        # For every gameweek the probability of squad player appearing in next gameweek must be >= 75%, while probability of lineup player > 90%
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += self.squad[p][gw] <= (self.player_prob_gw[p, gw] >= 0.75), f"Probability of appearance for squad player {p} for gameweek {gw}"
                self.model += self.lineup[p][gw] <= (self.player_prob_gw[p, gw] >= 0.90), f"Probability of appearance for lineup player {p} for gameweek {gw}"
            
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
        
        
    def _set_objective(self):
        """
        Summary:
        --------
        Function to define the objective function for the optimization problem.
        Objective function is added to model by using the += operator.
        
        Objective functions include:
            - Regular: Maximize total expected points over all gameweeks 
            - Decay: Maximize total expected points in each gameweek, with decay factor
            
        """
        
        gw_xp_before_pen = {gw: lpSum([self.player_xp_gw[p, gw] * (self.lineup[p][gw] + self.captain[p][gw] + 0.1 * self.vice_captain[p][gw]) for p in self.players]) for gw in self.future_gameweeks}
        gw_xp_after_pen = {gw: (gw_xp_before_pen[gw] - 4 * self.penalised_transfers[gw]) for gw in self.future_gameweeks}
        
        if self.objective == "regular":
            obj_func = lpSum([gw_xp_after_pen[gw] for gw in self.future_gameweeks])
            self.model += obj_func
        
        elif self.objective == "decay":
            obj_func = lpSum([gw_xp_after_pen[gw] * pow(self.decay_base, gw - self.gameweek) for gw in self.future_gameweeks])
            self.model += obj_func
            base_str = "_base_" + str(self.decay_base)
            self.model.name += base_str
            
    
    def get_results(self):
        """
        Summary:
        --------
        Function to extract results from the optimization problem.
        
        Returns:
        --------
        Dictionary with the following keys:
            - dataframe: Dataframe with results.
            - total_xp: Total expected points.
            - gw_xp: Dictionary with expected points for each gameweek.
        """
        
        if self.model.status != 1:
            print("Results unavailable since model is not solved.")
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
                                "name": self.merged_df.loc[p, "name"],
                                "team": self.merged_df.loc[p, "team"],
                                "position": self.merged_df.loc[p, "position"],
                                "position_id": self.merged_df.loc[p, "element_type"],
                                "cost": self.player_cost[p],
                                "prob_of_appearing": self.player_prob_gw[p, gw],
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
            
            # Update attribute
            self.results_df = results_df
            
            # Expected points for each gameweek
            self.gw_xp = {gw: round(value(lpSum([self.player_xp_gw[p, gw] * (self.lineup[p][gw] + self.captain[p][gw]) for p in self.players])), 2) for gw in self.future_gameweeks}
            
            # Total expected points for all optimized gameweeks (i.e. sum of values in gw_xp dictionary)
            self.total_xp = sum(self.gw_xp.values())

            return {'dataframe': self.results_df, 'total_xp': self.total_xp, 'gw_xp': self.gw_xp}
        
        
    def _check_results(self, results: pd.DataFrame):
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
            - Probability of appearance for each player in squad is greater than 75% for each gameweek
            - Probability of appearance for each player in lineup is greater than 90% for each gameweek
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
                condition_5 = all(gw_results.groupby("position_id").squad.sum() == self.positions_df["squad_select"])
                condition_6a = all(gw_results.groupby("position_id").lineup.sum() >= self.positions_df["squad_min_play"])
                condition_6b = all(gw_results.groupby("position_id").lineup.sum() <= self.positions_df["squad_max_play"])
                condition_7 = all(gw_results[gw_results["squad"] == 1].prob_of_appearing >= 0.75)
                condition_8 = all(gw_results[gw_results["lineup"] == 1].prob_of_appearing >= 0.90)
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
                        print(f"WARNING: Probability of appearance for each player in squad is not greater than 75% for gameweek {gw}.")
                    if not condition_8:
                        print(f"WARNING: Probability of appearance for each player in lineup is not greater than 90% for gameweek {gw}.")
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
                print("Results check: passed")

            return checks_dict

        
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
        
        # Only write summary if model is solved
        if self.model.status != 1:
            print("Summary unavailable since model is not solved.")
            return None
        else:
            pass
    
        # Initialize summary list
        summary_list = []
        
        for gw in self.future_gameweeks:
            summary_list.append("-" * 50)
            summary_list.append(f"Gameweek {gw} summary:")
            summary_list.append("-" * 50)
            summary_list.append(f"Total expected points: {self.gw_xp[gw]}")
            summary_list.append(f"Money in bank: {self.money_in_bank[gw].varValue}")
            summary_list.append(f"Free transfers available: {int(self.free_transfers_available[gw].varValue)}")
            summary_list.append(f"Transfers made: {int(value(self.transfers_made[gw]))}")
            summary_list.append(f"Penalised transfers: {int(self.penalised_transfers[gw].varValue)}")
            
            for p in self.players:
                if self.transfer_in[p][gw].varValue == 1:
                    name = self.merged_df.loc[p, 'name']
                    team = self.merged_df.loc[p, 'team']
                    summary_list.append(f"Player {p} ({name} @ {team}) transferred in.")
                if self.transfer_out[p][gw].varValue == 1:
                    name = self.merged_df.loc[p, 'name']
                    team = self.merged_df.loc[p, 'team']
                    summary_list.append(f"Player {p} ({name} @ {team}) transferred out.")
        
        # Join the summary list into a single string
        summary = "\n".join(summary_list)
        
        return summary
        
        
# ----------------------------------------
# Main
# ----------------------------------------       
        
if __name__ == "__main__":

    t0 = time.time()
    
    # Parameters to optimize over
    team_id = 10599528
    horizons = [1, 2, 3, 4, 5]
    objectives = ["regular"]
    gameweek = 24
    
    # Create and solve optimization problem for each combination of horizon and objective
    optimizers = [OptimizeMultiPeriod(team_id=team_id, gameweek=gameweek, num_free_transfers=1, horizon=h, objective=objective) for h in horizons for objective in objectives]
    
    for optimizer in optimizers:
        print("-" * 100)
        print(f"Optimizing for team {optimizer.team_id}, gameweek {optimizer.gameweek}, horizon {optimizer.horizon}, objective {optimizer.objective}")
        print("-" * 100)
        optimizer.solve_problem()
        solved_model = optimizer.model
        results = optimizer.get_results()
        summary = optimizer.get_summary()
        name = optimizer.model.name
        
        if results is not None:
            solved_model.writeLP(f"../../models/multi_period/{name}_model.lp")
            results["dataframe"].to_csv(f"../../data/external/{name}_results.csv", index=False)
            
            with open(f"../../reports/{name}_summary.txt", "w") as f:
                f.write(summary)
                f.close()
    
    print("*" * 100)
    print("Total time in loop:", round(time.time() - t0, 2))
    print("*" * 100)
    
    

        

