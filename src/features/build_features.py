# Description: Functions to build features for the model.

# Import packages
import pandas as pd
import sys

sys.path.append("..")

from data.get_data import FPLFormScraper

def merge_predicted_points(predicted_points_df, other_df) -> pd.DataFrame:
    """
    Summary:
    --------
    Merge predicted_points_df with other_df.
    
    Parameters:
    -----------
    predicted_points_df: pd.DataFrame
        Dataframe containing predicted points for each player.
    other_df: pd.DataFrame
        Dataframe containing other data.
        
    Returns:
    --------
    merged_df: pd.DataFrame
        Dataframe containing merged data.
    """
    
    print("Merging predicted points with data...")
    
    predicted_points_df = predicted_points_df.copy()
    other_df = other_df.copy()
    
    # Check if dataframe has columns id, web_name, position
    if "id" not in other_df.columns:
        print("Dataframe does not have column 'id'.")
        sys.exit(1)
        return None
    if "web_name" not in other_df.columns:
        print("Dataframe does not have column 'web_name'.")
        sys.exit(1)
        return None
    if "position" not in other_df.columns:
        print("Dataframe does not have column 'position'.")
        sys.exit(1)
        return None
    
    # Replace GK with GKP in predicted_points_df
    predicted_points_df["Pos"] = predicted_points_df["Pos"].replace("GK", "GKP")
    
    # Merge
    merged_df = pd.merge(
        other_df,
        predicted_points_df,
        left_on=[
            "id",
            "web_name",
            "position",
        ],
        right_on=["ID", "Name", "Pos"]
    )
    
    # Drop duplicate columns
    merged_df.drop(
        columns=["ID", "Name", "Pos", "Team", "Price"],
        inplace=True,
    )
    
    # Check if merged_df has same number of rows as other_df
    if merged_df.shape[0] != other_df.shape[0]:
        other_df_ids = other_df["id"].unique()
        merged_df_ids = merged_df["id"].unique()
        other_df_ids_not_in_merged_df_ids = set(other_df_ids) - set(merged_df_ids)
        print(f"Players left out of merged_df: {other_df_ids_not_in_merged_df_ids}")
    else:
        pass
    
    return merged_df









