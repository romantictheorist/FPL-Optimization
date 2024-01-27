# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys
import os

import pandas as pd
import requests

sys.path.append("..")

pd.options.mode.chained_assignment = None  # default='warn'

# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------

class FPLDataPuller:
    """
    Class for pulling data from FPL API.
    """
    
    base_url = "https://fantasy.premierleague.com/api/"
    raw_path = "../../data/raw/"
    
    def __init__(self):
        pass
    
    def get_general_data(self) -> dict:
        """
        Summary:
        --------
        Pulls general data from FPL API. Data includes: elements, element_types, teams, events.
        Data is exported to csv files in data/raw/gw_{current_gw}/
        Note: Some preprocessing is done to the data before exporting.
        
        Returns:
        --------
        Dictionary of general data for the current season.
        """
        
        r = requests.get(self.base_url + "bootstrap-static/").json()
        
        # Create dataframes for each endpoint
        elements_df = pd.DataFrame(r["elements"])
        element_types_df = pd.DataFrame(r["element_types"])
        teams_df = pd.DataFrame(r["teams"])
        events_df = pd.DataFrame(r["events"])
        
        # Find previous, current and next gameweek numbers from events_df
        previous_gw = events_df[events_df["is_previous"] == True]["id"].iloc[0]
        current_gw = events_df[events_df["is_current"] == True]["id"].iloc[0]
        next_gw = events_df[events_df["is_next"] == True]["id"].iloc[0]
        
        # Add previous, current and next gameweek numbers to elements_df as columns
        elements_df["previous_gw"] = previous_gw
        elements_df["current_gw"] = current_gw
        elements_df["next_gw"] = next_gw
        
        # Divide 'now_cost' by 10 to get cost in millions
        elements_df["now_cost"] = elements_df.now_cost / 10
        
        # Map 'element_type' in elements_df to 'singular_name_short' in element_types_df
        elements_df["position"] = elements_df.element_type.map(
            element_types_df.set_index("id").singular_name_short
        )
        
        # Map 'team' in elements_df to 'name' in teams_df
        elements_df["team_name"] = elements_df.team.map(teams_df.set_index("id").name)
        
        # Create folder for current gameweek if it doesn't exist
        if not os.path.exists(self.raw_path + "gw_" + str(current_gw)):
            os.mkdir(self.raw_path + "gw_" + str(current_gw))
        
        # Export dataframes to csv files
        elements_df.to_csv(
            self.raw_path + "gw_" + str(current_gw) + "/elements.csv", index=False
        )
        element_types_df.to_csv(
            self.raw_path + "gw_" + str(current_gw) + "/element_types.csv", index=False
        )
        teams_df.to_csv(self.raw_path + "gw_" + str(current_gw) + "/teams.csv", index=False)
        events_df.to_csv(self.raw_path + "gw_" + str(current_gw) + "/events.csv", index=False)
        
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
            "previous_gw": previous_gw,
            "current_gw": current_gw,
            "next_gw": next_gw,
        }
    
    
    def get_player_data(self, player_id: int) -> dict:
        """
        Summary:
        --------
        Pulls player data for a given player id from FPL API.
        Data includes: fixtures, history, history_past.
        Data is exported to csv files in data/raw/players/{player_id}/
        
        Parameters:
        -----------
        player_id: int
            Player id for which data is to be pulled.
        
        Returns:
        --------
        Dictionary of player data for the current season.
        """
        
        r = requests.get(self.base_url + "element-summary/" + str(player_id) + "/").json()
        
        # Create dataframes for each endpoint
        fixtures_df = pd.DataFrame(r["fixtures"])
        history_df = pd.DataFrame(r["history"])
        history_past_df = pd.DataFrame(r["history_past"])
        
        # Create folder for player if it doesn't exist
        if not os.path.exists(self.raw_path + "players/" + str(player_id)):
            os.mkdir(self.raw_path + "players/" + str(player_id))
        
        # Export dataframes to csv files
        fixtures_df.to_csv(
            self.raw_path + "players/" + str(player_id) + "/fixtures.csv", index=False
        )
        history_df.to_csv(
            self.raw_path + "players/" + str(player_id) + "/history.csv", index=False
        )
        history_past_df.to_csv(
            self.raw_path + "players/" + str(player_id) + "/history_past.csv", index=False
        )
        
        print(
            "Data pulled from FPL API and exported to csv files in data/raw/players/" + str(player_id) + "/"
        )
        
        return {
            "fixtures": fixtures_df,
            "history": history_df,
            "history_past": history_past_df,
        }
    
        
    def get_squad(self, team_id: int, gameweek: int) -> dict:
        """
        Summary:
        --------
        Pulls squad data for a given team id from FPL API.

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
        
        r = requests.get(
            self.base_url + "entry/" + str(team_id) + "/event/" + str(gameweek) + "/picks/"
        ).json()    
        
        # Raise warning if details not found, otherwise return list of player ids for the squad
        if "detail" in r.keys():
            raise ValueError("Details not found.")
            return None
        
        else:
            picks = r["picks"]
            squad = [p["element"] for p in picks]
            return squad
        
        
        
        





# my_team_id = 10599528
#pull_squad(team_id=10599528, gw=21)
