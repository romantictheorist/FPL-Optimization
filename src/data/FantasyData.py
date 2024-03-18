from dataclasses import dataclass

import pandas as pd
import requests


@dataclass
class FantasyData:
    """

    The `FantasyData` class provides methods to fetch and retrieve data from the Fantasy Premier League API.

    Attributes:
        fantasy_url (str): The URL for the Fantasy Premier League API.

    Methods:
        __post_init__(self) -> None:
            Initializes the class object by fetching the fantasy data from the API.

        get_players_df(self) -> pd.DataFrame:
            Retrieves and returns a pandas DataFrame containing information about all the players.

        get_positions_df(self) -> pd.DataFrame:
            Retrieves and returns a pandas DataFrame containing information about all the player positions.

        get_teams_df(self) -> pd.DataFrame:
            Retrieves and returns a pandas DataFrame containing information about all the teams.

        get_events_df(self) -> pd.DataFrame:
            Retrieves and returns a pandas DataFrame containing information about all the events.

        get_current_event(self) -> int:
            Retrieves and returns the ID of the current event.

        get_player_list(self) -> list:
            Retrieves and returns a list of player IDs.

        get_team_list(self) -> list:
            Retrieves and returns a list of team IDs.

        get_position_list(self) -> list:
            Retrieves and returns a list of position IDs.

        get_player_costs(self) -> dict:
            Retrieves and returns a dictionary mapping player IDs to their current costs.


    """

    fantasy_url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    def __post_init__(self) -> None:
        self.fantasy_data = requests.get(FantasyData.fantasy_url).json()

    def get_players_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.fantasy_data["elements"])

    def get_positions_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.fantasy_data["element_types"])

    def get_teams_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.fantasy_data["teams"])

    def get_events_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.fantasy_data["events"])

    def get_current_event(self) -> int:
        events = self.get_events_df()
        current_event = events.loc[events["is_current"], "id"].item()
        return current_event

    def get_player_list(self) -> list:
        players = self.get_players_df()
        return players["id"].to_list()

    def get_team_list(self) -> list:
        teams = self.get_teams_df()
        return teams["id"].to_list()

    def get_position_list(self) -> list:
        positions = self.get_positions_df()
        return positions["id"].to_list()

    def get_player_costs(self) -> dict:
        players = self.get_players_df()
        costs = players.set_index("id")["now_cost"]
        return costs.to_dict
