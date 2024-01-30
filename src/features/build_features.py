# Description: Functions to build features for the model.

# Import packages
import pandas as pd
import sys


def merge_fpl_form_data(dataframe):
    """
    Merge fpl_form_data with given dataframe.
    """
    fpl_form_data = pd.read_csv("../../data/raw/fpl-form-predicted-points.csv")

    # Replace GK with GKP 
    fpl_form_data["Pos"] = fpl_form_data["Pos"].replace("GK", "GKP")

    # Merge fpl_form_data with dataframe
    
    # Check if dataframe has columns id, web_name, position
    if "id" not in dataframe.columns:
        print("Dataframe does not have column 'id'.")
        sys.exit(1)
        return None
    if "web_name" not in dataframe.columns:
        print("Dataframe does not have column 'web_name'.")
        sys.exit(1)
        return None
    if "position" not in dataframe.columns:
        print("Dataframe does not have column 'position'.")
        sys.exit(1)
        return None
    
    # Merge
    merged_df = pd.merge(
        dataframe,
        fpl_form_data,
        left_on=[
            "id",
            "web_name",
            "position",
        ],
        right_on=["ID", "Name", "Pos"],
    )
    
    # Drop duplicate columns
    merged_df.drop(
        columns=["ID", "Name", "Pos", "Team", "Price"],
        inplace=True,
    )
    

    # If merge was successful, return merged_elements_df
    if len(merged_df) == len(merged_df):
        print("FPL form data successfully merged with elements_df.")
        return merged_df
    else:
        print("Merge was unsuccessful.")
        return None
