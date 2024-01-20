# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import json
import sys
import shutil
import os

import numpy as np
import pandas as pd
import requests

sys.path.append("..")

# ------------------------------------------------------------------------------
# Defining functions to get data from FPL API
# ------------------------------------------------------------------------------

# Base URL for all FPL API endpoints
base_url = "https://fantasy.premierleague.com/api/"


def get_general_data():
    """
    Returns a dictionary of general data for the current season.
    Data includes: events, game_settings, phases, teams, total_players, elements, element_stats, element_types.
    """
    r = requests.get(base_url + "bootstrap-static/").json()
    return r


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
# Creating dataframes from API data
# ------------------------------------------------------------------------------

# Create dataframes from general data
elements_df = pd.DataFrame(get_general_data()["elements"])
element_types_df = pd.DataFrame(get_general_data()["element_types"])
teams_df = pd.DataFrame(get_general_data()["teams"])
events_df = pd.DataFrame(get_general_data()["events"])

# Create dataframe from fixtures data
fixtures_df = pd.DataFrame(get_fixtures_data())

# ------------------------------------------------------------------------------
# Find current and next gameweek numbers and add to elements_df
# ------------------------------------------------------------------------------

# Find current gameweek number by getting id of row where 'is_current' is True
current_gw = events_df[events_df["is_current"] == True]["id"].iloc[0]

# Find next gameweek number by getting id of row where 'is_next' is True
next_gw = events_df[events_df["is_next"] == True]["id"].iloc[0]

# Add current_gw and next_gw to elements_df
elements_df["gw_current"] = current_gw
elements_df["gw_next"] = next_gw

# ------------------------------------------------------------------------------
# Export dataframes to csv files in data/raw/current_gw/
# ------------------------------------------------------------------------------

# Path to current_gw folder
path = "../../data/raw/gw_" + str(current_gw) + "/"

# If current_gw folder does not exist, create it
if not os.path.exists(path):
    os.mkdir(path)

# Export dataframes to csv files
elements_df.to_csv(path + "elements.csv", index=False)
element_types_df.to_csv(path + "element_types.csv", index=False)
teams_df.to_csv(path + "teams.csv", index=False)
events_df.to_csv(path + "events.csv", index=False)
fixtures_df.to_csv(path + "fixtures.csv", index=False)
