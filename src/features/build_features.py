# Description: Functions to build features for the model.

# Import packages
import pandas as pd
import sys
from typing import TypedDict, Optional
from datetime import datetime

sys.path.append("..")

from data.get_data import FPLFormScraper, FPLDataPuller


class ProcessData:
    def __init__(self):
        pass
    
    def _process_gameweek(self, gameweek_df) -> pd.DataFrame:
        """
        Summary:
        --------
        Process gameweek data.
        
        Arguments:
        ----------
        gameweek_df: pd.DataFrame
        
        Returns:
        --------
        gameweek_df: pd.DataFrame
            Dataframe containing processed gameweek data.
        """

        gameweek_df = gameweek_df.copy()
        
        # Keep only relevant columns
        cols_to_keep = [
            "id",
            "web_name",
            "team",
            "element_type",
            "now_cost",
            "form",
            "points_per_game",
            "total_points",
            "chance_of_playing_next_round",
            "status",
            "selected_by_percent",
            "transfers_in_event",
            "transfers_out_event",
        ]
        
        gameweek_df = gameweek_df[cols_to_keep]
        
        # Rename columns
        gameweek_df.rename(columns={'web_name': 'name',
                                    'team': 'team_id',
                                    'now_cost': 'cost',
                                    'chance_of_playing_next_round': 'official_chance',
                                    'status': 'official_availablity'}, 
                           inplace=True)
        
        # Divide cost by 10 to get cost in millions
        gameweek_df["cost"] = gameweek_df["cost"] / 10
        
        # Create dictionary for status
        status_dict = {
            "a": "Available",
            "d": "Doubtful",
            "i": "Injured",
            "s": "Suspended",
            "u": "Unavailable",
        }
        
        # Map status_dict to official_availablity
        gameweek_df["official_availablity"] = gameweek_df["official_availablity"].map(status_dict)
        
        # Create dictionary for position
        position_dict = {
            1: "GKP",
            2: "DEF",
            3: "MID",
            4: "FWD",
        }
        
        # Map position_dict to element_type
        position_column = gameweek_df["element_type"].map(position_dict)
        
        # Move position column to after element_type column
        gameweek_df.insert(4, "position", position_column)
        
        return gameweek_df
    
    def _map_teams_to_gameweek(self, teams_df, gameweek_df) -> pd.DataFrame:
        """
        Summary:
        --------
        Map team names to gameweek data using team IDs.
        
        Arguments:
        ----------
        teams_df: pd.DataFrame
            Dataframe containing team data.
        gameweek_df: pd.DataFrame
            Dataframe containing gameweek data.
            
        Returns:
        --------
        gameweek_df: pd.DataFrame
            Dataframe containing gameweek data with team names.
        """
       
        gameweek_df = gameweek_df.copy()
        teams_df = teams_df.copy()
        
        # Map 'short_name' in teams_df to 'team_name' in gameweek_df using 'team_id'
        team_column = gameweek_df["team_id"].map(teams_df.set_index("id").short_name)
        
        # Insert team_name column after team_id column
        gameweek_df.insert(3, "team", team_column)
    
        return gameweek_df
    
    
    def _process_predicted_points(self, predicted_points_df) -> pd.DataFrame:
        """
        Summary:
        --------
        Process predicted points data.
        
        Arguments:
        ----------
        predicted_points_df: pd.DataFrame
            Dataframe containing predicted points for each player.
            
        Returns:
        --------
        predicted_points_df: pd.DataFrame
            Dataframe containing processed predicted points data.
        """
        
        predicted_points_df = predicted_points_df.copy()
        
        # Replace GK with GKP in predicted_points_df
        predicted_points_df["Pos"] = predicted_points_df["Pos"].replace("GK", "GKP")
        
        # Drop last three columns
        predicted_points_df = predicted_points_df.iloc[:, :-3]
        
        # Drop all columns with 'with_prob' in column name (these are predicted points x probability of playing)
        predicted_points_df = predicted_points_df.loc[:, ~predicted_points_df.columns.str.contains("with_prob")]
        
        # Rename columns
        # If column name number and contains 'pts', rename to 'GW1_Pts' 'GW2_Pts' etc.
        predicted_points_df.rename(columns=lambda x: f"gw_{x.split('_')[0]}_xp" if x.split('_')[0].isdigit() and "pts" in x else x, inplace=True)
        predicted_points_df.rename(columns=lambda x: f"gw_{x.split('_')[0]}_prob_of_appearing" if x.split('_')[0].isdigit() and x.split('_')[1].__eq__('prob') else x, inplace=True)
        
        return predicted_points_df
    
    def _merge_gameweek_and_predicted_points(self, gameweek_df, predicted_points_df) -> pd.DataFrame:
        """
        Summary:
        --------
        Merge gameweek data with predicted points data.
        
        Arguments:
        ----------
        gameweek_df: pd.DataFrame
            Dataframe containing gameweek data.
        predicted_points_df: pd.DataFrame
            Dataframe containing predicted points for each player.
        
        Returns:
        --------
        merged_df: pd.DataFrame
            Dataframe containing merged data.
        """
   
        gameweek_df = gameweek_df.copy()
        predicted_points_df = predicted_points_df.copy()
        
        # Merged on player_id and name
        merged_df = pd.merge(gameweek_df, predicted_points_df, 
                            left_on=["id", "name"], 
                            right_on=["ID", "Name"])
        
        # Drop duplicate columns
        merged_df.drop(columns=["ID", "Name", "Pos", "Team", "Price"], inplace=True)
        
        # Check if merged_df has same number of rows as gameweek_df
        if merged_df.shape[0] != gameweek_df.shape[0]:
            gameweek_df_ids = gameweek_df["id"].unique()
            merged_df_ids = merged_df["id"].unique()
            gameweek_df_ids_not_in_merged_df_ids = set(gameweek_df_ids) - set(merged_df_ids)
            print(f"Players left out of merge: {gameweek_df_ids_not_in_merged_df_ids}")
            
        return merged_df
    




















