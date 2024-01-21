# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys
import os

import numpy as np
import pandas as pd
import requests

sys.path.append("..")

pd.options.mode.chained_assignment = None  # default='warn'

# ------------------------------------------------------------------------------
# Defining functions to get data from FPL API
# ------------------------------------------------------------------------------

# Base URL for all FPL API endpoints
base_url = "https://fantasy.premierleague.com/api/"


def get_general_data():
    """
    Returns a dictionary of general data for the current season.
    Data includes: elements, element_types, teams, events.
    """
    r = requests.get(base_url + "bootstrap-static/").json()

    elements_df = pd.DataFrame(r["elements"])
    element_types_df = pd.DataFrame(r["element_types"])
    teams_df = pd.DataFrame(r["teams"])
    events_df = pd.DataFrame(r["events"])

    # Find current and next gameweek numbers using is_current and is_next columns in events_df
    current_gw = events_df[events_df["is_current"] == True]["id"].iloc[0]
    next_gw = events_df[events_df["is_next"] == True]["id"].iloc[0]

    # Add current_gw and next_gw to elements_df
    elements_df["gw_current"] = current_gw
    elements_df["gw_next"] = next_gw

    return {
        "elements": elements_df,
        "element_types": element_types_df,
        "teams": teams_df,
        "events": events_df,
    }


def get_fixtures_data():
    """
    Returns a dataframe of fixtures for the current season.
    """
    r = requests.get(base_url + "fixtures/").json()
    return pd.DataFrame(r)


def get_player_data(player_id):
    """
    Returns a dictionary of player data for a given player id.
    Data includes: fixtures, history, history_past.
    """
    r = requests.get(base_url + "element-summary/" + str(player_id) + "/").json()

    return r


# ------------------------------------------------------------------------------
# Using functions to get data from FPL API
# ------------------------------------------------------------------------------

# Get general data
general_data = get_general_data()

elements_df = general_data["elements"]
element_types_df = general_data["element_types"]
teams_df = general_data["teams"]
events_df = general_data["events"]

# Get fixtures data
fixtures_df = get_fixtures_data()

# ------------------------------------------------------------------------------
# Exporting dataframes to csv files in data/raw/current_gw/
# ------------------------------------------------------------------------------

# Get current gameweek number
current_gw = elements_df["gw_current"].iloc[0]

# Defining path to save dataframes to
path = "../../data/raw/gw_" + str(current_gw) + "/"

# Check if path exists, if not create it
if not os.path.exists(path):
    os.mkdir(path)

# Export dataframes to csv files
elements_df.to_csv(path + "elements.csv", index=False)
element_types_df.to_csv(path + "element_types.csv", index=False)
teams_df.to_csv(path + "teams.csv", index=False)
events_df.to_csv(path + "events.csv", index=False)
fixtures_df.to_csv(path + "fixtures.csv", index=False)

# Print message to console
print(
    "FPL data successfully downloaded and saved to data/raw/gw_" + str(current_gw) + "/"
)

# ------------------------------------------------------------------------------
