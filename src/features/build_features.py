# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys
import os
import numpy as np
import pandas as pd

sys.path.append("..")

# ------------------------------------------------------------------------------
# Load raw data
# ------------------------------------------------------------------------------

gw = 21
raw_path = "../../data/raw/gw_" + str(gw)

elements_df = pd.read_csv(raw_path + "/elements.csv")
element_types_df = pd.read_csv(raw_path + "/element_types.csv")
teams_df = pd.read_csv(raw_path + "/teams.csv")
events_df = pd.read_csv(raw_path + "/events.csv")
fixtures_df = pd.read_csv(raw_path + "/fixtures.csv")

# ------------------------------------------------------------------------------
# Define functions that clean and process the raw data
# ------------------------------------------------------------------------------


def build_slim_elements(elements_df):
    """
    Cleans and processes the raw elements data.
    """

    # List of columns to keep
    slim_features = [
        "id",
        "web_name",
        "team",
        "element_type",
        "gw_current",
        "gw_next",
        "selected_by_percent",
        "now_cost",
        "minutes",
        "transfers_in",
        "value_season",
        "total_points",
        "points_per_game",
        "form",
        "ep_this",
        "ep_next",
    ]

    # Select slim_features
    slim_elements_df = elements_df[slim_features]

    # Map 'element_type' in slim_elements_df to 'singular_name_short' in element_types_df
    slim_elements_df["position"] = slim_elements_df.element_type.map(
        element_types_df.set_index("id").singular_name_short
    )

    # Map 'team' in slim_elements_df to 'name' in teams_df
    slim_elements_df["team_name"] = slim_elements_df.team.map(
        teams_df.set_index("id").name
    )

    # Create a new column called 'value' (points / cost) that takes float value of 'value_season'
    slim_elements_df["value"] = slim_elements_df.value_season.astype(float)

    return slim_elements_df


def build_slim_element_types(element_types_df):
    """
    Cleans and processes the raw element_types data.
    """

    # List of columns to keep
    slim_features = [
        "id",
        "singular_name_short",
        "squad_select",
        "squad_min_play",
        "squad_max_play",
    ]

    # Select slim_features
    slim_element_types_df = element_types_df[slim_features]

    return slim_element_types_df


def build_slim_teams(teams_df):
    """
    Cleans and processes the raw teams data.
    """

    # List of columns to keep
    slim_features = [
        "id",
        "name",
        "short_name",
        "strength",
        "strength_attack_away",
        "strength_attack_home",
        "strength_defence_away",
        "strength_defence_home",
        "strength_overall_away",
        "strength_overall_home",
    ]

    # Select slim_features
    slim_teams_df = teams_df[slim_features]

    return slim_teams_df


def build_slim_events_df(events_df):
    """
    Cleans and processes the raw events data.
    """

    # List of columns to keep
    slim_features = [
        "id",
        "name",
        "deadline_time",
        "finished",
        "data_checked",
        "is_previous",
        "is_current",
        "is_next",
        "most_selected",
        "most_transferred_in",
        "top_element",
        "top_element_info",
        "most_captained",
        "most_vice_captained",
    ]

    # Select slim_features
    slim_events_df = events_df[slim_features]

    return slim_events_df


def build_slim_fixtures_df(fixtures_df):
    """
    Cleans and processes the raw fixtures data.
    """

    # List of columns to keep
    slim_features = [
        "id",
        "code",
        "event",
        "finished",
        "finished_provisional",
        "team_a",
        "team_a_score",
        "team_h",
        "team_h_score",
        "kickoff_time",
        "minutes",
        "provisional_start_time",
        "started",
    ]

    # Select slim_features
    slim_fixtures_df = fixtures_df[slim_features]

    # Map 'team_a' in slim_fixtures_df to 'name' in teams_df
    slim_fixtures_df["team_a_name"] = slim_fixtures_df.team_a.map(
        teams_df.set_index("id").name
    )

    # Map 'team_h' in slim_fixtures_df to 'name' in teams_df
    slim_fixtures_df["team_h_name"] = slim_fixtures_df.team_h.map(
        teams_df.set_index("id").name
    )

    return slim_fixtures_df


# ------------------------------------------------------------------------------
# Export slim dataframes to csv
# ------------------------------------------------------------------------------

# Build slim dataframes
slim_elements_df = build_slim_elements(elements_df)
slim_element_types_df = build_slim_element_types(element_types_df)
slim_teams_df = build_slim_teams(teams_df)
slim_events_df = build_slim_events_df(events_df)
slim_fixtures_df = build_slim_fixtures_df(fixtures_df)

# Define processed path
processed_path = "../../data/processed/gw_" + str(gw)

# If current_gw folder does not exist, create it
if not os.path.exists(processed_path):
    os.mkdir(processed_path)

# Export slim dataframes to csv
slim_elements_df.to_csv(processed_path + "/slim_elements.csv", index=False)
slim_element_types_df.to_csv(processed_path + "/slim_element_types.csv", index=False)
slim_teams_df.to_csv(processed_path + "/slim_teams.csv", index=False)
slim_events_df.to_csv(processed_path + "/slim_events.csv", index=False)
slim_fixtures_df.to_csv(processed_path + "/slim_fixtures.csv", index=False)

# ------------------------------------------------------------------------------
