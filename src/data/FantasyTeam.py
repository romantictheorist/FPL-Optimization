from dataclasses import dataclass, field

import pandas as pd
import requests


@dataclass
class FantasyTeam:
    """
    A class representing a fantasy team in the Premier League Fantasy Football game.

    Attributes:
        team_url (str): The URL for the API endpoint to fetch team data.
        team_id (int): The ID of the team.

    Notes:
        - The team_data attribute acts as a placeholder for fetched team data.
        - The __post_init__ method is automatically called after the instance is initialized.
        - The check_team method verifies the validity of the team ID and raises errors if necessary.
        - The get_team_data method fetches and returns the data for the team.
        - The remaining methods retrieve specific information from the team's data.

    Methods:
        check_team() -> bool:
            Checks the validity of the team ID. Returns True if valid, otherwise raises an error.

        get_team_data() -> dict:
            Fetches and returns the data for the team based on the team ID.

        get_started_gw() -> int:
            Returns the gameweek when the team started playing.

        get_current_gw() -> int:
            Returns the current gameweek.

        get_bank_balance() -> float:
            Returns the bank balance of the team.

        get_team_value() -> float:
            Returns the value of the team.

        get_last_num_transfers() -> int:
            Returns the number of transfers made in the last deadline.

        get_current_picks() -> list:
            Returns the ID of the players in the team's current lineup.

        get_team_transfers() -> pd.DataFrame:
            Returns a DataFrame containing the team's transfer history.

        get_team_history() -> pd.DataFrame:
            Returns a DataFrame containing the team's performance history.

        get_num_next_free_transfers() -> int:
            Returns the number of remaining free transfers for the team.
    """

    team_url = "https://fantasy.premierleague.com/api/entry/"
    team_id: int
    team_data: dict = field(init=False, default_factory=dict)

    # If team ID passes checks, fetch team data and update placeholder
    def __post_init__(self):
        if self.check_team():
            self.team_data = self.get_team_data()

    def check_team(self) -> bool:
        if not isinstance(self.team_id, int):
            raise ValueError("Team ID must be an integer.")
        else:
            pass

        team_data = self.get_team_data()

        if "detail" in team_data and team_data["detail"] == "Not found.":
            raise ValueError("Team ID does not exist.")
        else:
            return True

    def get_team_data(self) -> dict:
        response = requests.get(f"{FantasyTeam.team_url}{self.team_id}/")
        return response.json()

    def get_started_gw(self) -> int:
        return int(self.team_data["started_event"])

    def get_current_gw(self) -> int:
        return int(self.team_data["current_event"])

    def get_bank_balance(self) -> float:
        return float(self.team_data["last_deadline_bank"] / 10)

    def get_team_value(self) -> float:
        return float(self.team_data["last_deadline_value"] / 10)

    def get_last_num_transfers(self) -> int:
        return int(self.team_data["last_deadline_total_transfers"])

    def get_current_picks(self) -> list:
        current_gw = self.get_current_gw()
        response = requests.get(
            f"{FantasyTeam.team_url}{self.team_id}/event/{current_gw}/picks"
        )
        picks = response.json()["picks"]
        picks_id = [p["element"] for p in picks]
        return picks_id

    def get_team_transfers(self) -> pd.DataFrame:
        response = requests.get(f"{FantasyTeam.team_url}{self.team_id}/transfers")
        transfers = response.json()
        transfers_df = pd.DataFrame(transfers).sort_values(by="time", ascending=True)
        return transfers_df.reset_index(drop=True)

    def get_team_history(self) -> pd.DataFrame:
        response = requests.get(f"{FantasyTeam.team_url}{self.team_id}/history")
        history = response.json()["current"]
        return pd.DataFrame(history)

    def get_num_next_free_transfers(self) -> int:
        started_gw = self.get_started_gw()
        transfer_history = self.get_team_history()[["event", "event_transfers"]]

        rolling_next_ft = []

        for index, row in transfer_history.iterrows():
            if row["event"] < started_gw:
                rolling_next_ft.append(float("nan"))
            elif row["event"] == started_gw:
                rolling_next_ft.append(1)
            else:
                transfer_diff = rolling_next_ft[-1] + 1 - int(row["event_transfers"])

                if transfer_diff > 2:
                    next_ft = 2
                elif 0 < transfer_diff <= 2:
                    next_ft = transfer_diff
                else:
                    next_ft = 1

                rolling_next_ft.append(next_ft)

        return rolling_next_ft[-1]
