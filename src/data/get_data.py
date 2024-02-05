import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

sys.path.append("..")

pd.options.mode.chained_assignment = None  # default='warn'


class FPLDataPuller:
    """
    Class for pulling data from FPL API.
    """

    base_url = "https://fantasy.premierleague.com/api/"

    def __init__(self):
        pass

    def _get_general_data(self) -> dict:
        """
        Summary:
        --------
        Pulls gameweek data from FPL API. Data includes: elements, element_types, teams, events.

        Returns:
        --------
        Dictionary of data for the current gameweek.
        """

        print("Pulling gameweek data from FPL API...")

        r = requests.get(self.base_url + "bootstrap-static/").json()

        # Create dataframes for each endpoint
        gameweek_df = pd.DataFrame(r["elements"])
        positions_df = pd.DataFrame(r["element_types"])
        teams_df = pd.DataFrame(r["teams"])
        events_df = pd.DataFrame(r["events"])

        # Get current gameweek from events_df
        current_gw = events_df[events_df["is_current"] == True]["id"].values[0]

        return {
            "gameweek": gameweek_df,
            "positions": positions_df,
            "teams": teams_df,
            "events": events_df,
            "current_gw": current_gw,
        }

    def _get_team_ids(self, team_id: int, gameweek: int) -> dict:
        """
        Summary:
        --------
        Pulls team ids for a given team id and gameweek from FPL API.

        Args:
        -----
        team_id: int
            Team id for which data is to be pulled.
        gameweek: int
            Gameweek for which data is to be pulled.

        Returns:
        --------
        List of player ids for the squad.
        """

        print(
            f"Pulling team {team_id} player IDs for gameweek {gameweek} from FPL API..."
        )

        r = requests.get(
            self.base_url
            + "entry/"
            + str(team_id)
            + "/event/"
            + str(gameweek)
            + "/picks/"
        ).json()

        # Raise warning if details not found, otherwise return list of player ids for the squad
        if "detail" in r.keys():
            raise ValueError("Details not found.")
            return None

        else:
            picks = r["picks"]
            ids = [p["element"] for p in picks]
            return ids

    def _get_team_data(self, team_id: int) -> dict:
        """
        Summary:
        --------
        Pulls team data for a given team id from FPL API.
        Data includes: money_in_bank, team_value, total_points, gameweek_points, current_gw.

        Parameters:
        -----------
        team_id: int
            Team id for which data is to be pulled.

        Returns:
        --------
        Dictionary of team data for the current season.
        """

        print(f"Pulling team {team_id} data from FPL API...")

        r = requests.get(self.base_url + "entry/" + str(team_id) + "/").json()

        bank_balance = r["last_deadline_bank"] / 10
        team_value = r["last_deadline_value"] / 10
        total_points = r["summary_overall_points"]
        gameweek_points = r["summary_event_points"]
        current_gw = r["current_event"]

        return {
            "bank_balance": bank_balance,
            "team_value": team_value,
            "total_points": total_points,
            "gameweek_points": gameweek_points,
            "current_gw": current_gw,
        }


class FPLFormScraper:
    """
    Class for scraping data from FPLForm.com
    """

    url = "https://fplform.com/export-fpl-form-data"

    def __init__(self):
        pass

    def _get_predicted_points(self) -> pd.DataFrame:
        """
        Summary:
        --------
        Scrapes predicted points from FPLForm.com.

        Returns:
        --------
        Dataframe of predicted points.
        """

        print("Scraping predicted points from FPLForm...")

        # Open browser and navigate to FPLForm.com
        driver = webdriver.Chrome()
        driver.get(self.url)

        # Move slider to the right to select all gameweeks
        slider = driver.find_element(
            By.XPATH, "//div[contains(@class, 'handle-upper')]"
        )
        ActionChains(driver).drag_and_drop_by_offset(slider, 500, 0).perform()

        # Find and click the "With Extra Columns" button
        with_extra_columns_button = driver.find_element(By.ID, "extra")
        with_extra_columns_button.click()

        # Find and click the "Generate CSV file" button
        generate_button = driver.find_element(
            By.XPATH, "//button[contains(text(), 'Generate')]"
        )
        generate_button.click()

        # Wait for the download to complete
        time.sleep(1)

        # Close the browser
        driver.quit()

        # Get path to downloaded file by finding the most recent file in the downloads folder
        downloads_folder = os.path.expanduser("~") + "/Downloads"
        files = os.listdir(downloads_folder)
        files.sort(key=lambda x: os.path.getctime(os.path.join(downloads_folder, x)))
        most_recent_file = files[-1]

        # Read the csv file into a dataframe
        df = pd.read_csv(downloads_folder + "/" + most_recent_file)

        # Delete the file from the downloads folder
        os.remove(downloads_folder + "/" + most_recent_file)

        return df
