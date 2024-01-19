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
# Load and process data
# ----------------------------------------

slim_elements = pd.read_csv("../../data/processed/slim_elements.csv")
slim_elements.set_index("id", inplace=True)

# ----------------------------------------
# List of player IDs and positions (for indexing)
# ----------------------------------------

player_ids = slim_elements.index.to_list()
positions = slim_elements.position.unique().tolist()
teams = slim_elements.team_name.unique().tolist()

# ----------------------------------------
# Initialising the model
# ----------------------------------------

model = LpProblem("SinglePeriod", sense=LpMaximize)

# ----------------------------------------
# Defining the decision variables
# ----------------------------------------

squad = LpVariable.dicts("squad", player_ids, cat="Binary")
lineup = LpVariable.dicts("lineup", player_ids, cat="Binary")
captain = LpVariable.dicts("captain", player_ids, cat="Binary")
vice_captain = LpVariable.dicts("vice_captain", player_ids, cat="Binary")

# ----------------------------------------
# Defining constraints
# ----------------------------------------

# Total number of players in squad must be 15
model += lpSum([squad[i] for i in player_ids]) == 15

# Total number of players in lineup must be 11
model += lpSum([lineup[i] for i in player_ids]) == 11

# Only 1 captain
model += lpSum([captain[i] for i in player_ids]) == 1

# Only 1 vice captain
model += lpSum([vice_captain[i] for i in player_ids]) == 1

# Lineup player must be in squad (but reverse is can not be true)
for i in player_ids:
    model += lineup[i] <= squad[i]

# Captain must be in lineup
for i in player_ids:
    model += captain[i] <= lineup[i]

# Vice captain must be in lineup
for i in player_ids:
    model += vice_captain[i] <= lineup[i]

# Captain and vice captain can not be the same player
for i in player_ids:
    model += captain[i] + vice_captain[i] <= 1
