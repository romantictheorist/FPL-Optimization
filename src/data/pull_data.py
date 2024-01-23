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
# Defining functions to pull data from FPL API
# ------------------------------------------------------------------------------

# Base URL for all FPL API endpoints
base_url = "https://fantasy.premierleague.com/api/"


def pull_general_data():
    """
    Returns a dictionary of general data for the current season.
    Data includes: elements, element_types, teams, events.
    """
    r1 = requests.get(base_url + "bootstrap-static/").json()
    elements_df = pd.DataFrame(r1["elements"])
    element_types_df = pd.DataFrame(r1["element_types"])
    teams_df = pd.DataFrame(r1["teams"])
    events_df = pd.DataFrame(r1["events"])

    r2 = requests.get(base_url + "fixtures/").json()
    fixtures_df = pd.DataFrame(r2)

    # Find current and next gameweek numbers using is_current and is_next columns in events_df
    current_gw = events_df[events_df["is_current"] == True]["id"].iloc[0]
    next_gw = events_df[events_df["is_next"] == True]["id"].iloc[0]

    # Add current_gw and next_gw to elements_df
    elements_df["current_gw"] = current_gw
    elements_df["nexy_gw"] = next_gw

    # Map 'element_type' in elements_df to 'singular_name_short' in element_types_df
    elements_df["position"] = elements_df.element_type.map(
        element_types_df.set_index("id").singular_name_short
    )

    # Map 'team' in elements_df to 'name' in teams_df
    elements_df["team_name"] = elements_df.team.map(teams_df.set_index("id").name)

    # Map 'team_a' and 'team_h' in fixtures_df to 'name' in teams_df
    fixtures_df["team_a_name"] = fixtures_df.team_a.map(teams_df.set_index("id").name)
    fixtures_df["team_h_name"] = fixtures_df.team_h.map(teams_df.set_index("id").name)

    # Define path to raw data
    raw_path = "../../data/raw/"

    # Check if folder for current gameweek exists, if not create it
    if not os.path.exists(raw_path + "gw_" + str(current_gw)):
        os.mkdir(raw_path + "gw_" + str(current_gw))

    # Export elements_df to csv file
    elements_df.to_csv(
        raw_path + "gw_" + str(current_gw) + "/elements.csv", index=False
    )
    element_types_df.to_csv(
        raw_path + "gw_" + str(current_gw) + "/element_types.csv", index=False
    )
    teams_df.to_csv(raw_path + "gw_" + str(current_gw) + "/teams.csv", index=False)
    events_df.to_csv(raw_path + "gw_" + str(current_gw) + "/events.csv", index=False)
    fixtures_df.to_csv(
        raw_path + "gw_" + str(current_gw) + "/fixtures.csv", index=False
    )

    print(
        "Data pulled from FPL API and exported to csv files in data/raw/gw_"
        + str(current_gw)
        + "/"
    )

    return {
        "elements": elements_df,
        "element_types": element_types_df,
        "teams": teams_df,
        "events": events_df,
        "fixtures": fixtures_df,
        "current_gw": current_gw,
        "next_gw": next_gw,
    }


def pull_player_data(player_id):
    """
    Returns a dictionary of player data for a given player id.
    Data includes: fixtures, history, history_past.
    """
    r = requests.get(base_url + "element-summary/" + str(player_id) + "/").json()

    return r
