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




class OptimiseTeam:
    def __init__(self, 
                 team_id, 
                 gameweek, 
                 bank_balance, 
                 num_free_transfers,
                 horizon,
                 objective='regular',
                 decay_base=0.85,
    ):
        
        self.team_id = team_id
        self.gameweek = gameweek
        self.bank_balance = bank_balance
        self.num_free_transfers = num_free_transfers
        self.horizon = horizon
        self.objective = objective
        self.decay_base = decay_base
        
    def get_data(self):
        # Pull general data from FPL API
        self.general_data = pull_general_data()
        self.elements_df = self.general_data['elements']
        self.elements_types_df = self.general_data['element_types']
        self.teams_df = self.general_data['teams']
        
        # Pull squad data from FPL API
        self.initial_squad = pull_squad(self.team_id, self.gameweek)
        
        # Merge FPL form data
        form_data = pd.read_csv("../../data/raw/fpl-form-predicted-points.csv")
        self.merged_elements_df = merge_fpl_form_data(self.elements_df, form_data)
        
        # Set index to IDs
        self.merged_elements_df.set_index('id', inplace=True)
        self.elements_types_df.set_index('id', inplace=True)
        self.teams_df.set_index('id', inplace=True)
        
        # Return data
        return {'merged_elements_df': self.merged_elements_df, 
                'elements_types_df': self.elements_types_df,
                'teams_df': self.teams_df,
                'initial_squad': self.initial_squad}
        
    
    def define_variables(self):
        self.squad = LpVariable.dicts("squad", (self.players, self.all_gameweeks), cat='Binary')
        self.lineup = LpVariable.dicts("lineup", (self.players, self.future_gameweeks), cat='Binary')
        self.captain = LpVariable.dicts("captain", (self.players, self.future_gameweeks), cat='Binary')
        self.vice_captain = LpVariable.dicts("vice_captain", (self.players, self.future_gameweeks), cat="Binary")
        self.transfer_in = LpVariable.dicts("transfer_in", (self.players, self.future_gameweeks), cat="Binary")
        self.transfer_out = LpVariable.dicts("transfer_out", (self.players, self.future_gameweeks), cat="Binary")
        self.money_in_bank = LpVariable.dicts("money_in_bank", (self.all_gameweeks), lowBound=0, cat="Continuous")
        self.free_transfers_available = LpVariable.dicts("free_transfers_available", (self.all_gameweeks), lowBound=1, upBound=2, cat="Integer")
        self.penalised_transfers = LpVariable.dicts("penalised_transfers", (self.future_gameweeks), cat="Integer", lowBound=0)
        self.aux = LpVariable.dicts("auxiliary_variable", (self.future_gameweeks), cat="Binary")
        
    def define_constraints(self):
        # Define dictionaries to use for constraints
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
        self.transfer_diff = {gw: (self.transfers_made[gw] - self.free_transfers_available[gw]) for gw in self.future_gameweeks}
        
        # Define initial conditions
  
        # Players in next gameweek squad must either be in current gameweek squad or transferred in
        # And players not in next gameweek squad must be transferred out
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += (self.squad[p][gw] == (self.squad[p][gw - 1] + self.transfer_in[p][gw] -  self.transfer_out[p][gw])), f"Player {p} squad/transfer constraint for gameweek {gw}"

        # Number of transfers made in each gameweek cannot exceed 5
        for gw in self.future_gameweeks:
            self.model += self.transfers_made[gw] <= 20, f"Transfers made constraint for gameweek {gw}"
        
        
        # Money in bank at current gameweek must be equal to bank balance
        self.model += self.money_in_bank[gameweek - 1] == self.bank_balance, f"Initial money in bank constraint"

        # Number of free transfers available in current gameweek must be equal to num_free_transfers
        self.model += self.free_transfers_available[gameweek - 1] == self.num_free_transfers, f"Initial free transfers available constraint"
        
        # ----------------------------------------
        # Squad and lineup constraints
        # ----------------------------------------

        # Total number of players in squad in each gameweek must be equal to 15
        for gw in self.future_gameweeks:
            self.model += self.squad_count[gw] == 15, f"Squad count constraint for gameweek {gw}"

        # Total number of players in lineup in each gameweek must be equal to 11
        for gw in self.future_gameweeks:
            self.model += self.lineup_count[gw] == 11, f"Lineup count constraint for gameweek {gw}"

        # Lineup player must be in squad (but reverse can not be true) in each gameweek
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += self.lineup[p][gw] <= self.squad[p][gw], f"Lineup player must be in squad constraint for player {p} in gameweek {gw}"
        
        
        # ----------------------------------------
        # Captain and vice captain constraints
        # ----------------------------------------

        # Only 1 captain in each gameweek
        for gw in self.future_gameweeks:
            self.model += lpSum([self.captain[p][gw] for p in self.players]) == 1, f"Captain count constraint for gameweek {gw}"

        # Only 1 vice captain in each gameweek
        for gw in self.future_gameweeks:
            self.model += lpSum([self.vice_captain[p][gw] for p in self.players]) == 1, f"Vice captain count constraint for gameweek {gw}"

        # Captain must be in lineup in each gameweek
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += self.captain[p][gw] <= self.lineup[p][gw], f"Captain must be in lineup constraint for player {p} in gameweek {gw}"

        # Vice captain must be in lineup in each gameweek
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += self.vice_captain[p][gw] <= self.lineup[p][gw], f"Vice captain must be in lineup constraint for player {p} in gameweek {gw}"

        # Captain and vice captain can not be the same player in each gameweek
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += self.captain[p][gw] + self.vice_captain[p][gw] <= 1, f"Captain and vice captain can not be the same player constraint for player {p} in gameweek {gw}"
                
        # ----------------------------------------
        # Position / Formation constraints
        # ----------------------------------------

        # Number of players in each position in lineup must be within the allowed range (defined in element_types_df as squad_min_play and squad_max_play) for every gameweek
        for gw in self.future_gameweeks:
            for pos in self.positions:
                self.model += (self.lineup_position_count[pos, gw] >= self.elements_types_df.loc[pos, "squad_min_play"]), f"Min lineup players in position {pos} in gameweek {gw}"
                self.model += (self.lineup_position_count[pos, gw] <= self.elements_types_df.loc[pos, "squad_max_play"]), f"Max lineup players in position {pos} in gameweek {gw}"


        # Number of players in each position in squad must be satisfied (defined in element_types_df as squad_select) for every gameweek
        for gw in self.future_gameweeks:
            for pos in self.positions:
                self.model += (self.squad_position_count[pos, gw] == self.elements_types_df.loc[pos, "squad_select"]), f"Squad players in position {pos} in gameweek {gw}"
            
            
        
        
        
        
        
        
        
        
        
        
        
        
    def solve(self):
        # Get data
        self.get_data()
        
        # List of players, positions, teams and gameweeks
        self.players = self.merged_elements_df.index.values
        self.positions = self.elements_types_df.index.values
        self.teams = self.teams_df.index.values
        
        # List of gameweeks
        self.future_gameweeks = list(range(self.gameweek, self.gameweek+self.horizon))
        self.all_gameweeks = [self.gameweek -  1] + self.future_gameweeks
        
        # Define model
        model_name = f"solve_EV_max_gw_{gameweek}_horizon_{horizon}_objective_{objective}"
        self.model = LpProblem(model_name, LpMaximize)
        
        # Define variables
        self.define_variables()
        
        # Define constraints
        self.define_constraints()
        
        
        return self.model
        
     
        


   
# Create instance of class
# Define parameters
team_id = 1
gameweek = 22
bank_balance = 4.2
num_free_transfers = 1
horizon = 3
objective = 'regular'
decay_base = 0.85

optimizer = OptimiseTeam(team_id, gameweek, bank_balance, num_free_transfers, horizon, objective, decay_base)
data = optimizer.get_data()
data['merged_elements_df'].head()

model = optimizer.solve()