# Description: Functions to build features for the model.

# Import packages
import pandas as pd


def merge_fpl_form_data(elements_df, fpl_form_data):
    """
    Merge fpl_form_data with elements_df.
    """

    # Replace GK with GKP to match elements_df
    fpl_form_data["Pos"] = fpl_form_data["Pos"].replace("GK", "GKP")

    # Merge fpl_form_data with elements_df
    merged_elements_df = pd.merge(
        elements_df,
        fpl_form_data,
        left_on=[
            "id",
            "web_name",
            "position",
        ],
        right_on=["ID", "Name", "Pos"],
    )

    # Drop duplicate columns
    merged_elements_df.drop(
        columns=["ID", "Name", "Pos", "Team", "Price"],
        inplace=True,
    )

    # If merge was successful, return merged_elements_df
    if len(merged_elements_df) == len(elements_df):
        print("FPL form data successfully merged with elements_df.")
        return merged_elements_df
    else:
        print("Merge was unsuccessful.")
        return None
