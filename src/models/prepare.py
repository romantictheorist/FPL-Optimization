import json
import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from ..data.FantasyData import FantasyData
from ..data.FantasyTeam import FantasyTeam


@dataclass
class OptimiserDataset:
    """
    Class to represent the dataset required for the optimisation process.

    Attributes:
        team_id (int): The ID of the team.
        horizon (int): The number of future gameweeks to consider in the optimisation.

        players (List[int]): The list of player IDs.
        teams (List[int]): The list of team IDs.
        positions (List[int]): The list of position IDs.

        current_gameweek (int): The current gameweek number.
        future_gameweeks (List[int]): The list of future gameweek numbers.
        all_gameweeks (List[int]): The list of all gameweek numbers.

        initial_team (List[int]): The initial fantasy team.
        bank_balance (float): The bank balance of the team.
        free_transfers (int): The number of free transfers available.

        squad_min_play (dict): A dictionary mapping position IDs to minimum number of players to play for that position.
        squad_max_play (dict): A dictionary mapping position IDs to maximum number of players to play for that position.
        squad_select (dict): A dictionary mapping position IDs to the number of players to select for that position.

        predictions (pd.DataFrame): The predictions dataframe containing player predictions for each gameweek.

    """

    team_id: int
    horizon: int

    players: List[int]
    teams: List[int]
    positions: List[int]

    current_gameweek: int
    future_gameweeks: List[int]
    all_gameweeks: List[int]

    initial_team: List[int]
    bank_balance: float
    free_transfers: int

    squad_min_play: dict
    squad_max_play: dict
    squad_select: dict

    predictions: pd.DataFrame


@dataclass
class PrepareDatasetForOptimiser:
    """
    PrepareDatasetForOptimiser

    Class for preparing the dataset for the optimiser.

    Attributes:
        fantasy_data (FantasyData): Instance of the `FantasyData` class that fetches data from the FPL API.
        fantasy_team (FantasyTeam): Instance of the `FantasyTeam` class that fetches data for the fantasy team.
        dataset (OptimiserDataset): Instance of the `OptimiserDataset` class that holds the prepared dataset.

    Methods:
        __post_init__: Initializes the attributes of the class.
        prepare_data: Prepares the dataset by fetching required data from the API and creating an instance of `OptimiserDataset`.
        check_valid_horizon: Checks if the given horizon is valid based on the number of remaining gameweeks.
        get_future_gameweeks: Returns a list of future gameweeks based on the current gameweek and horizon.
        get_all_gameweeks: Returns a list of all gameweeks based on the current gameweek and horizon.
        get_squad_min_play: Returns a dictionary mapping position IDs to the minimum number of players to play in the position.
        get_squad_max_play: Returns a dictionary mapping position IDs to the maximum number of players to play in the position.
        get_squad_select: Returns a dictionary mapping position IDs to the number of positions to select in the squad.
        build_predictions_df: Fetches and preprocesses players data from the FPL API and FPLForm data.

    """

    # Placeholders
    fantasy_data: FantasyData = field(init=False)
    fantasy_team: FantasyTeam = field(init=False)
    dataset: OptimiserDataset = field(init=False)

    def __post_init__(self):
        # Load settings from JSON file
        with open("settings.json", "r") as f:
            settings = json.load(f)

        # Set the parameters
        self.team_id = settings["team_id"]
        self.horizon = settings["horizon"]

        # Create instances of custom classes that fetch data from FPL API
        self.fantasy_data = FantasyData()
        self.fantasy_team = FantasyTeam(self.team_id)

        # Check horizon: if pass, update dataset placeholder
        if self.check_valid_horizon():
            self.dataset = self.prepare_data()

    def prepare_data(self):
        team_id = self.team_id
        horizon = self.horizon

        players = self.fantasy_data.get_player_list()
        teams = self.fantasy_data.get_team_list()
        positions = self.fantasy_data.get_position_list()

        current_gameweek = self.fantasy_data.get_current_event()
        future_gameweeks = self.get_future_gameweeks()
        all_gameweeks = self.get_all_gameweeks()

        initial_team = self.fantasy_team.get_current_picks()
        bank_balance = self.fantasy_team.get_bank_balance()
        free_transfers = self.fantasy_team.get_num_next_free_transfers()

        squad_min_play = self.get_squad_min_play()
        squad_max_play = self.get_squad_max_play()
        squad_select = self.get_squad_select()

        predictions = self.build_predictions_df()

        return OptimiserDataset(
            team_id,
            horizon,
            players,
            teams,
            positions,
            current_gameweek,
            future_gameweeks,
            all_gameweeks,
            initial_team,
            bank_balance,
            free_transfers,
            squad_min_play,
            squad_max_play,
            squad_select,
            predictions,
        )

    def check_valid_horizon(self) -> bool:
        current_gameweek = self.fantasy_data.get_current_event()
        remaining_gameweeks = 38 - current_gameweek
        if current_gameweek + self.horizon <= 38:
            return True
        else:
            raise ValueError(
                f"Invalid horizon ({self.horizon}): Cannot be greater than the number of"
                f" remaining gameweeks ({remaining_gameweeks})."
            )

    def get_future_gameweeks(self) -> list:
        current_gw = self.fantasy_data.get_current_event()
        next_gw = current_gw + 1
        return list(range(next_gw, next_gw + self.horizon))

    def get_all_gameweeks(self) -> list:
        current_gw = self.fantasy_data.get_current_event()
        future_gameweeks = self.get_future_gameweeks()
        return [current_gw] + future_gameweeks

    def get_squad_min_play(self) -> dict:
        positions_df = self.fantasy_data.get_positions_df().set_index("id")
        return positions_df["squad_min_play"].to_dict()

    def get_squad_max_play(self) -> dict:
        positions_df = self.fantasy_data.get_positions_df().set_index("id")
        return positions_df["squad_max_play"].to_dict()

    def get_squad_select(self) -> dict:
        positions_df = self.fantasy_data.get_positions_df().set_index("id")
        return positions_df["squad_select"].to_dict()

    def build_predictions_df(self) -> pd.DataFrame:
        # Fetch and preprocess players data
        players_df = self.fantasy_data.get_players_df()
        players_df = players_df.drop_duplicates()
        players_df = players_df[
            [
                "id",
                "web_name",
                "element_type",
                "team",
                "total_points",
                "now_cost",
                "form",
                "points_per_game",
                "selected_by_percent",
                "status",
            ]
        ]

        status_dict = {
            "a": "Available",
            "d": "Doubtful",
            "i": "Injured",
            "s": "Suspended",
            "u": "Unavailable",
        }

        players_df["status"] = players_df["status"].map(status_dict)

        players_df.rename(
            columns={
                "web_name": "name",
                "element_type": "position",
                "now_cost": "price",
            },
            inplace=True,
        )

        players_df["price"] = players_df["price"] / 10

        # Read and preprocess FPLForm data
        location = "data/external/fpl-form-predicted-points.csv"
        if not os.path.isfile(location):
            raise FileNotFoundError(
                f"FPLForm Predicted Points file not found at location: {location}"
            )
        else:
            fplform_df = pd.read_csv(location)
            fplform_df = fplform_df.drop_duplicates()
            cols_to_keep = ["ID"] + [col for col in fplform_df if col[0].isnumeric()]
            fplform_df = fplform_df[cols_to_keep]
            beyond_horizon_gws = set(
                range(max(self.get_future_gameweeks()) + 1, 38 + 1)
            )
            beyond_horizon_cols = [
                col
                for col in cols_to_keep
                for gw in beyond_horizon_gws
                if col.startswith(str(gw))
            ]
            fplform_df.drop(columns=beyond_horizon_cols, inplace=True)
            fplform_df.drop(
                columns=fplform_df.filter(like="with_prob").columns, inplace=True
            )
            fplform_df.drop(columns=fplform_df.filter(like="tba").columns, inplace=True)
            fplform_df.columns = [
                col.replace("pts_no_prob", "xp") for col in fplform_df.columns
            ]

        # Merge if sizes are equal
        if len(fplform_df) == len(players_df):
            merged_df = players_df.merge(
                fplform_df, left_on="id", right_on="ID", how="left"
            ).drop("ID", axis=1)
        else:
            raise ValueError(
                f"Cannot merge FPL API data with FPL Form data because differing sizes: "
                f"{len(players_df), len(fplform_df)}"
            )

        # Fill NaN values with zeroes
        merged_df = merged_df.fillna(0)

        return merged_df
