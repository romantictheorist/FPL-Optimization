# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys

import numpy as np
import pandas as pd

sys.path.append("..")

# ------------------------------------------------------------------------------
# Load raw data
# ------------------------------------------------------------------------------

elements_df = pd.read_csv("../../data/raw/elements.csv")
element_types_df = pd.read_csv("../../data/raw/element_types.csv")
teams_df = pd.read_csv("../../data/raw/teams.csv")
fixtures_df = pd.read_csv("../../data/raw/fixtures.csv")

# ------------------------------------------------------------------------------
# Clean data
# ------------------------------------------------------------------------------

# Select only the columns we want
slim_elements_df = elements_df[
    [
        "id",
        "web_name",
        "team",
        "element_type",
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
]

# Create a new column called 'position' that maps the element_type to the position name
slim_elements_df["position"] = slim_elements_df.element_type.map(
    element_types_df.set_index("id").singular_name_short
)

# Create a new column called 'team_name' that maps the team to the team name
slim_elements_df["team_name"] = slim_elements_df.team.map(teams_df.set_index("id").name)

# Create a new column called 'value' that calculates the value of a player (points / cost)
slim_elements_df["value"] = slim_elements_df.value_season.astype(float)

# ------------------------------------------------------------------------------
# Export slim data
# ------------------------------------------------------------------------------

slim_elements_df.to_csv("../../data/processed/slim_elements.csv", index=False)
