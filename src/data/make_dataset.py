import json
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append("..")


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


elements_df = pd.DataFrame(get_general_data()["elements"])
element_types_df = pd.DataFrame(get_general_data()["element_types"])
teams_df = pd.DataFrame(get_general_data()["teams"])
fixtures_df = get_fixtures_data()

# Export dataframes to csv files
elements_df.to_csv("../../data/raw/elements.csv", index=False)
element_types_df.to_csv("../../data/raw/element_types.csv", index=False)
teams_df.to_csv("../../data/raw/teams.csv", index=False)
fixtures_df.to_csv("../../data/raw/fixtures.csv", index=False)