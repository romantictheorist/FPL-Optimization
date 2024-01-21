# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys
import os
import numpy as np
import pandas as pd

sys.path.append("..")
pd.options.mode.chained_assignment = None  # default='warn'

# ------------------------------------------------------------------------------
# Defining a function that cleans and process the raw data folders (gw_1, gw_2, etc.)
# ------------------------------------------------------------------------------


def build_slim_dataframes(gw):
    """
    Builds and returns a dictionary of slim dataframes for a given gameweek.
    """

    # Define raw path
    raw_path = "../../data/raw/gw_" + str(gw) + "/"

    # Check if raw_path exists: if not, throw error, else continue
    if not os.path.exists(raw_path):
        raise Exception("Raw data for gameweek " + str(gw) + " does not exist.")
    else:
        # ------------------------------------------------------------------------------
        # Load raw data for specified gameweek
        # ------------------------------------------------------------------------------

        elements_df = pd.read_csv(raw_path + "elements.csv")
        element_types_df = pd.read_csv(raw_path + "element_types.csv")
        teams_df = pd.read_csv(raw_path + "teams.csv")
        events_df = pd.read_csv(raw_path + "events.csv")
        fixtures_df = pd.read_csv(raw_path + "fixtures.csv")

        # ------------------------------------------------------------------------------
        # slim_elements_df
        # ------------------------------------------------------------------------------

        # Filter columns
        slim_elements_df = elements_df[
            [
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
                "chance_of_playing_this_round",
                "chance_of_playing_next_round",
            ]
        ]

        # Map 'element_type' in slim_elements_df to 'singular_name_short' in slim_element_types_df
        slim_elements_df["position"] = slim_elements_df.element_type.map(
            element_types_df.set_index("id").singular_name_short
        )

        # # Map 'team' in slim_elements_df to 'name' in slim_teams_df
        slim_elements_df["team_name"] = slim_elements_df.team.map(
            teams_df.set_index("id").name
        )

        # # Create a new column called 'value' (points / cost) that takes float value of 'value_season'
        slim_elements_df["value"] = slim_elements_df.value_season.astype(float)

        # ------------------------------------------------------------------------------
        # slim_element_types_df
        # ------------------------------------------------------------------------------

        # Filter columns
        slim_element_types_df = element_types_df[
            [
                "id",
                "singular_name_short",
                "squad_select",
                "squad_min_play",
                "squad_max_play",
            ]
        ]

        # ------------------------------------------------------------------------------
        # slim_teams_df
        # ------------------------------------------------------------------------------

        # Filter columns
        slim_teams_df = teams_df[
            [
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
        ]

        # ------------------------------------------------------------------------------
        # slim_events_df
        # ------------------------------------------------------------------------------

        # Filter columns
        slim_events_df = events_df[
            [
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
        ]

        # ------------------------------------------------------------------------------
        # slim_fixtures_df
        # ------------------------------------------------------------------------------

        # Filter columns
        slim_fixtures_df = fixtures_df[
            [
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
        ]

        # Map 'team_a' in slim_fixtures_df to 'name' in teams_df
        slim_fixtures_df["team_a_name"] = slim_fixtures_df.team_a.map(
            teams_df.set_index("id").name
        )

        # Map 'team_h' in slim_fixtures_df to 'name' in teams_df
        slim_fixtures_df["team_h_name"] = slim_fixtures_df.team_h.map(
            teams_df.set_index("id").name
        )

        # ------------------------------------------------------------------------------
        # Return slim dataframes
        # ------------------------------------------------------------------------------

    return {
        "slim_elements_df": slim_elements_df,
        "slim_element_types_df": slim_element_types_df,
        "slim_teams_df": slim_teams_df,
        "slim_events_df": slim_events_df,
        "slim_fixtures_df": slim_fixtures_df,
    }


# ------------------------------------------------------------------------------

# Define raw path
path = "../../data/raw/"

# Get gameweek number of newest raw data folder, i.e. current_gw
current_gw = max(
    [
        int(folder.split("_")[1])
        for folder in os.listdir(path)
        if os.path.isdir(path + folder)
    ]
)

# Build slim dataframes for current_gw
slim_elements_df = build_slim_dataframes(current_gw)["slim_elements_df"]
slim_element_types_df = build_slim_dataframes(current_gw)["slim_element_types_df"]
slim_teams_df = build_slim_dataframes(current_gw)["slim_teams_df"]
slim_events_df = build_slim_dataframes(current_gw)["slim_events_df"]
slim_fixtures_df = build_slim_dataframes(current_gw)["slim_fixtures_df"]

# Define processed path
processed_path = "../../data/processed/gw_" + str(current_gw)

# If current_gw folder does not exist, create it
if not os.path.exists(processed_path):
    os.mkdir(processed_path)

# Export slim dataframes to csv
slim_elements_df.to_csv(processed_path + "/slim_elements.csv", index=False)
slim_element_types_df.to_csv(processed_path + "/slim_element_types.csv", index=False)
slim_teams_df.to_csv(processed_path + "/slim_teams.csv", index=False)
slim_events_df.to_csv(processed_path + "/slim_events.csv", index=False)
slim_fixtures_df.to_csv(processed_path + "/slim_fixtures.csv", index=False)

# Print message to console
print(
    "Slim dataframes for gameweek "
    + str(current_gw)
    + " successfully exported to data/processed/gw_"
    + str(current_gw)
    + "/"
)

# ------------------------------------------------------------------------------
