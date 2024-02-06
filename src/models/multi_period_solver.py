import os
import time
from datetime import datetime

import pandas as pd
import pulp
from pulp import LpProblem, LpVariable, lpSum, LpMaximize
from pulp import value as pulp_value

from data.get_data import FPLDataPuller, FPLFormScraper
from features.build_features import ProcessData


class OptimizeMultiPeriod(FPLDataPuller, FPLFormScraper, ProcessData):
    # Class variable to store the data pulled from the FPL API and FPLForm.com
    _cached_data = None

    def __init__(
        self,
        team_id: int,
        gameweek: int,
        horizon: int,
        objective="regular",
        decay_base=0.85,
    ):
        # Step 1: Set the instance variables
        super().__init__()
        self.team_id = team_id
        self.gameweek = gameweek
        self.horizon = horizon
        self.objective = objective
        self.decay_base = decay_base

        self.future_gameweeks = list(range(self.gameweek, self.gameweek + self.horizon))
        self.all_gameweeks = [self.gameweek - 1] + list(
            range(self.gameweek, self.gameweek + self.horizon)
        )

        self.results = None
        self.gw_xp = None
        self.total_xp = None

        # Step 2: Check if the gameweek is valid
        if self._check_gameweek(gameweek=self.gameweek):
            self.current_gw = self.gameweek - 1

        # Step 3: Get the data needed for the optimization problem
        if OptimizeMultiPeriod._cached_data is None:
            print("Fetching data...")
            data = self._get_data(team_id=self.team_id)

            # List of keys to save
            data_to_save = [
                "gameweek_df",
                "positions_df",
                "teams_df",
                "predicted_points_df",
            ]

            # Save raw data
            self._save_data(
                data={key: data[key] for key in data_to_save if key in data},
                destination=f"../../data/raw/gw_{self.current_gw}/",
            )

            # Process the raw data
            processed_data = self._process_data(data=data)

            # Save the processed data
            self._save_data(
                data={"merged_df": processed_data["merged_df"]},
                destination=f"../../data/processed/gw_{self.current_gw}/",
            )
            # Set the class variable to the processed data
            OptimizeMultiPeriod._cached_data = processed_data

        # Step 5: Unpack the needed (processed) data from the class variable into instance variables
        self.merged_df = self._cached_data["merged_df"]
        self.positions_df = self._cached_data["positions_df"]
        self.teams_df = self._cached_data["teams_df"]
        self.initial_squad = self._cached_data["initial_squad"]
        self.bank_balance = self._cached_data["bank_balance"]
        self.num_free_transfers = self._cached_data["num_free_transfers"]
        self.players = self._cached_data["players"]
        self.positions = self._cached_data["positions"]
        self.teams = self._cached_data["teams"]

    def solve_problem(self):
        """
        Summary:
        --------
        Function to solve the optimization problem. The function follows the following steps:
            1. Check if the gameweek is valid
            2. Set the optimization problem
            3. Set the optimization variables for the problem
            4. Set the optimization dictionaries (i.e. player cost, expected points, probability of appearance, etc.)
            5. Set the initial conditions for the optimization problem
            6. Set the constraints for the optimization problem
            7. Set the objective function for the optimization problem
            8. Solve the optimization problem
            9. Get the results of the optimization problem
            10. Check the results of the optimization problem

        Returns:
        --------
        None
        """
        # Step 1: Set the optimization problem
        self._set_problem()

        # Step 2: Set the optimization variables for the problem
        self._set_variables()

        # Step 3: Set the optimization dictionaries (i.e. player cost, expected points, probability of appearance, etc.)
        self._set_dictionaries()

        # Step 4: Set the initial conditions for the optimization problem
        self._set_initial_conditions()

        # Step 5: Set the constraints for the optimization problem
        self._set_constraints()

        # Step 6: Set the objective function for the optimization problem
        self._set_objective()

        # Step 7: Solve the optimization problem
        print("Solving model...")
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))

        if self.model.status != 1:
            print("Model could not solved.")
            print("Status:", self.model.status)
            return None
        else:
            print("Model solved.")
            print("Time:", round(self.model.solutionTime, 2))
            print("Objective:", round(self.model.objective.value(), 2))

        # Step 8: Get the results of the optimization problem
        self.results = self.get_results()

        # Step 9: Check the results of the optimization problem
        if self._check_results(self.results):
            print("Results passed checks.")

    def _get_data(self, team_id: int):
        """
        Summary:
        --------
        Function to get the data needed for the optimization problem. The function follows the following steps:
            1. Get the general data (i.e. gameweek, positions, teams, etc.)
            2. Get the team data (i.e. bank balance, current squad, etc.)
            3. Get the predicted points for each

        Arguments:
        ----------
        team_id (int): Team ID

        Returns:
        --------
        Dictionary with the following keys:
            - gameweek_df: Dataframe with gameweek data
            - positions_df: Dataframe with position data
            - teams_df: Dataframe with team data
            - predicted_points_df: Dataframe with predicted points for each player
            - current_gw: Current gameweek
            - initial_squad: Initial squad
            - bank_balance: Bank balance
        """

        # Only pull data from the API and FPLForm.com if it hasn't been pulled yet
        raw_path = f"../../data/raw/gw_{self.current_gw}/"

        if (
            os.path.exists(raw_path + "gameweek_df.csv")
            and os.path.exists(raw_path + "positions_df.csv")
            and os.path.exists(raw_path + "teams_df.csv")
        ):
            print("Data already pulled from FPL API.")
            gameweek_df = pd.read_csv(raw_path + "gameweek_df.csv")
            positions_df = pd.read_csv(raw_path + "positions_df.csv")
            teams_df = pd.read_csv(raw_path + "teams_df.csv")
        else:
            print("Pulling data from FPL API...")
            general_data = self._get_general_data()
            gameweek_df = general_data["gameweek"]
            positions_df = general_data["positions"]
            teams_df = general_data["teams"]

        # Only pull data from FPLForm.com if today's data hasn't been pulled yet
        today = datetime.today().strftime("%Y_%m_%d")
        if os.path.exists(raw_path + f"predicted_points_df_{today}.csv"):
            print("Predicted points already pulled from FPLForm.com.")
            predicted_points_df = pd.read_csv(
                raw_path + f"predicted_points_df_{today}.csv"
            )
        else:
            print("Pulling predicted points from FPLForm.com...")
            predicted_points_df = self._get_predicted_points()

        # Pull team data from the FPL API
        initial_squad = self._get_team_ids(team_id=team_id, gameweek=self.current_gw)
        bank_balance = self._get_team_data(team_id=team_id)["bank_balance"]

        # Pull number of free transfers available
        num_free_transfers = self._get_num_free_transfers(team_id=team_id)

        # Store the data in a dictionary
        data = {
            "gameweek_df": gameweek_df,
            "positions_df": positions_df,
            "teams_df": teams_df,
            "predicted_points_df": predicted_points_df,
            "initial_squad": initial_squad,
            "bank_balance": bank_balance,
            "num_free_transfers": num_free_transfers,
        }

        return data

    def _process_data(self, data: dict):
        gameweek_df = self._process_gameweek(gameweek_df=data["gameweek_df"])
        gameweek_df = self._map_teams_to_gameweek(
            gameweek_df=gameweek_df, teams_df=data["teams_df"]
        )
        predicted_points_df = self._process_predicted_points(
            predicted_points_df=data["predicted_points_df"]
        )
        merged_df = self._merge_gameweek_and_predicted_points(
            gameweek_df=gameweek_df, predicted_points_df=predicted_points_df
        )

        # Set index as IDs for needed dataframes
        for df in [merged_df, data["positions_df"], data["teams_df"]]:
            df.set_index("id", inplace=True)

        players = merged_df.index.tolist()
        positions = data["positions_df"].index.tolist()
        teams = data["teams_df"].index.tolist()

        # Update data dictionary
        data["merged_df"] = merged_df
        data["gameweek_df"] = gameweek_df
        data["predicted_points_df"] = predicted_points_df
        data["players"] = players
        data["positions"] = positions
        data["teams"] = teams

        return data

    @staticmethod
    def _save_data(data: dict, destination: str):
        """
        Summary:
        --------
        Function to save the data to a specified destination. The function follows the following steps:
            1. Check if the destination exists. If not, create it.
            2. Save the data to the destination.

        Arguments:
        ----------
        data (dict): Dictionary with the data to save
        destination (str): Destination to save the data

        Returns:
        --------
        None
        """
        if not os.path.exists(destination):
            os.makedirs(destination)

        for key, value in data.items():
            # If data is predicted points, save with current date
            if key != "predicted_points_df":
                value.to_csv(f"{destination}{key}.csv", index=False)
            else:
                current_date = datetime.today().strftime("%Y_%m_%d")
                value.to_csv(f"{destination}{key}_{current_date}.csv", index=False)

    def _check_gameweek(self, gameweek: int):
        """
        Summary:
        --------
        Function to if the gameweek is valid.
        If the gameweek given is not valid, i.e. if it is not the next gameweek, raise a ValueError.

        Arguments:
        ----------
        gameweek (int): Gameweek to check

        Returns:
        --------
        True if the gameweek is valid, otherwise raises a ValueError.
        """

        # Get the current gameweek
        current_gw = self._get_current_gameweek()

        if gameweek < current_gw + 1:
            raise ValueError(
                f"Cannot optimize for GW{gameweek} since it has already passed. Optimization can only start from the "
                f"next gameweek (i.e. GW{current_gw + 1})."
            )

        elif gameweek > current_gw + 1:
            raise ValueError(
                f"Cannot optimize for GW{gameweek} since it is ahead of the next gameweek. Optimization can only "
                f"start from the next gameweek (i.e. GW{current_gw + 1})."
            )
        else:
            return True

    def _set_problem(self):
        """
        Summary:
        --------
        Function to define the optimization problem.

        Returns:
        --------
        None
        """
        problem_name = (
            f"multiperiod_optimizer_{self.team_id}"
            f"_gw_{self.gameweek}_horizon_{self.horizon}_objective_{self.objective}"
        )
        self.model = LpProblem(problem_name, LpMaximize)

    def _set_variables(self):
        """
        Summary:
        --------
        Function to set the optimization variables for the problem.

        Returns:
        --------
        None
        """

        self.squad = LpVariable.dicts(
            "squad", (self.players, self.all_gameweeks), cat="Binary"
        )

        self.lineup = LpVariable.dicts(
            "lineup", (self.players, self.future_gameweeks), cat="Binary"
        )
        self.captain = LpVariable.dicts(
            "captain", (self.players, self.future_gameweeks), cat="Binary"
        )
        self.vice_captain = LpVariable.dicts(
            "vice_captain", (self.players, self.future_gameweeks), cat="Binary"
        )
        self.transfer_in = LpVariable.dicts(
            "transfer_in", (self.players, self.future_gameweeks), cat="Binary"
        )
        self.transfer_out = LpVariable.dicts(
            "transfer_out", (self.players, self.future_gameweeks), cat="Binary"
        )
        self.money_in_bank = LpVariable.dicts(
            "money_in_bank", self.all_gameweeks, lowBound=0, cat="Continuous"
        )
        self.free_transfers_available = LpVariable.dicts(
            "free_transfers_available",
            self.all_gameweeks,
            lowBound=1,
            upBound=2,
            cat="Integer",
        )
        self.penalised_transfers = LpVariable.dicts(
            "penalised_transfers", self.future_gameweeks, cat="Integer", lowBound=0
        )
        self.aux = LpVariable.dicts(
            "auxiliary_variable", self.future_gameweeks, cat="Binary"
        )

    def _set_dictionaries(self):
        """
        Summary:
        --------
        Function to define the optimization dictionaries.
        These are used to help define the constraints and objective function.

        Returns:
        --------
        None
        """

        self.player_cost = self.merged_df["cost"].to_dict()
        self.player_xp_gw = {
            (p, gw): self.merged_df.loc[p, f"gw_{gw}_xp"]
            for p in self.players
            for gw in self.future_gameweeks
        }
        self.player_prob_gw = {
            (p, gw): self.merged_df.loc[p, f"gw_{gw}_prob_of_appearing"]
            for p in self.players
            for gw in self.future_gameweeks
        }
        self.squad_count = {
            gw: lpSum([self.squad[p][gw] for p in self.players])
            for gw in self.future_gameweeks
        }
        self.lineup_count = {
            gw: lpSum([self.lineup[p][gw] for p in self.players])
            for gw in self.future_gameweeks
        }
        self.lineup_position_count = {
            (pos, gw): lpSum(
                [
                    self.lineup[p][gw]
                    for p in self.players
                    if self.merged_df.loc[p, "element_type"] == pos
                ]
            )
            for pos in self.positions
            for gw in self.future_gameweeks
        }
        self.squad_position_count = {
            (pos, gw): lpSum(
                [
                    self.squad[p][gw]
                    for p in self.players
                    if self.merged_df.loc[p, "element_type"] == pos
                ]
            )
            for pos in self.positions
            for gw in self.future_gameweeks
        }
        self.squad_team_count = {
            (team, gw): lpSum(
                [
                    self.squad[p][gw]
                    for p in self.players
                    if self.merged_df.loc[p, "team_id"] == team
                ]
            )
            for team in self.teams
            for gw in self.future_gameweeks
        }
        self.revenue = {
            gw: lpSum(
                [self.player_cost[p] * self.transfer_out[p][gw] for p in self.players]
            )
            for gw in self.future_gameweeks
        }
        self.expenditure = {
            gw: lpSum(
                [self.player_cost[p] * self.transfer_in[p][gw] for p in self.players]
            )
            for gw in self.future_gameweeks
        }
        self.transfers_made = {
            gw: lpSum([self.transfer_in[p][gw] for p in self.players])
            for gw in self.future_gameweeks
        }
        self.transfers_made[self.gameweek - 1] = 1
        self.transfer_diff = {
            gw: (self.transfers_made[gw] - self.free_transfers_available[gw])
            for gw in self.future_gameweeks
        }

        # Position constraints dictionary
        self.squad_min_play = self.positions_df["squad_min_play"].to_dict()
        self.squad_max_play = self.positions_df["squad_max_play"].to_dict()
        self.squad_select = self.positions_df["squad_select"].to_dict()

    def _set_initial_conditions(self):
        """
        Summary: -------- Function to define the initial conditions for the optimization problem. These are the
        conditions that must be satisfied at the start of the gameweek. Initial conditions are added to model as
        constraints by using the += operator. Note that constraints, and therefore initial conditions, must include
        conditional operators (<=, >=, ==).

        Returns:
        --------
        None
        """

        # Players in initial squad must be in squad in current gameweek
        for p in [player for player in self.players if player in self.initial_squad]:
            self.model += (
                self.squad[p][self.gameweek - 1] == 1,
                f"In initial squad constraint for player {p}",
            )

        # Players not in initial squad must not be in squad in current gameweek
        for p in [
            player for player in self.players if player not in self.initial_squad
        ]:
            self.model += (
                self.squad[p][self.gameweek - 1] == 0,
                f"Not initial squad constraint for player {p}",
            )

        # Money in bank at current gameweek must be equal to bank balance
        self.model += (
            self.money_in_bank[self.gameweek - 1] == self.bank_balance,
            f"Initial money in bank constraint",
        )

        # Number of free transfers available in current gameweek must be equal to num_free_transfers
        self.model += (
            self.free_transfers_available[self.gameweek - 1] == self.num_free_transfers,
            f"Initial free transfers available constraint",
        )

    def _set_constraints(self):
        """
        Summary:
        --------
        Function to define the constraints for the optimization problem.
        Constraints must include conditional operators (<=, >=, ==), and are added to model by using the += operator.

        Constraints are added to model by using the += operator

        Constraints include:
            - Squad and lineup constraints
            - Captain and vice captain constraints
            - Position / Formation constraints
            - Team played for constraints
            - Probability of appearance constraints
            - Budgeting / Financial constraints
            - General transfer constraints
            - Free transfer constraints

        Returns:
        --------
        None
        """

        # Squad and lineup constraints
        for gw in self.future_gameweeks:
            self.model += (
                self.squad_count[gw] == 15,
                f"Squad count constraint for gameweek {gw}",
            )
            self.model += (
                self.lineup_count[gw] == 11,
                f"Lineup count constraint for gameweek {gw}",
            )

            for p in self.players:
                self.model += (
                    self.lineup[p][gw] <= self.squad[p][gw],
                    f"Lineup player must be in squad constraint for player {p} in gameweek {gw}",
                )

        # Captain and vice captain constraints
        for gw in self.future_gameweeks:
            self.model += (
                lpSum([self.captain[p][gw] for p in self.players]) == 1,
                f"Captain count constraint for gameweek {gw}",
            )
            self.model += (
                lpSum([self.vice_captain[p][gw] for p in self.players]) == 1,
                f"Vice captain count constraint for gameweek {gw}",
            )

            for p in self.players:
                self.model += (
                    self.captain[p][gw] <= self.lineup[p][gw],
                    f"Captain must be in lineup constraint for player {p} in gameweek {gw}",
                )
                self.model += (
                    self.vice_captain[p][gw] <= self.lineup[p][gw],
                    f"Vice captain must be in lineup constraint for player {p} in gameweek {gw}",
                )
                self.model += (
                    self.captain[p][gw] + self.vice_captain[p][gw] <= 1,
                    f"Captain and vice captain can not be the same player constraint for player {p} in gameweek {gw}",
                )

        # Position / Formation constraints
        for gw in self.future_gameweeks:
            for pos in self.positions:
                self.model += (
                    self.lineup_position_count[pos, gw]
                    >= self.positions_df.loc[pos, "squad_min_play"]
                ), f"Min lineup players in position {pos} in gameweek {gw}"
                self.model += (
                    self.lineup_position_count[pos, gw]
                    <= self.positions_df.loc[pos, "squad_max_play"]
                ), f"Max lineup players in position {pos} in gameweek {gw}"
                self.model += (
                    self.squad_position_count[pos, gw]
                    == self.positions_df.loc[pos, "squad_select"]
                ), f"Squad players in position {pos} in gameweek {gw}"

        # Team played for constraints
        for gw in self.future_gameweeks:
            for team in self.teams:
                self.model += (
                    self.squad_team_count[team, gw] <= 3
                ), f"Max players from team {team} in gameweek {gw}"

        # Probability of appearance constraints
        for gw in self.future_gameweeks:
            for p in self.players:
                self.model += (
                    self.squad[p][gw] <= (self.player_prob_gw[p, gw] >= 0.75),
                    f"Probability of appearance for squad player {p} for gameweek {gw}",
                )
                self.model += (
                    self.lineup[p][gw] <= (self.player_prob_gw[p, gw] >= 0.90),
                    f"Probability of appearance for lineup player {p} for gameweek {gw}",
                )

        # Budgeting / Financial constraints
        for gw in self.future_gameweeks:
            self.model += (
                self.money_in_bank[gw]
                == (
                    self.money_in_bank[gw - 1] + self.revenue[gw] - self.expenditure[gw]
                )
            ), f"Money in bank constraint for gameweek {gw}"

        # General transfer constraints
        for gw in self.future_gameweeks:
            self.model += (
                self.transfers_made[gw] <= 20,
                f"Transfers made constraint for gameweek {gw}",
            )

            for p in self.players:
                self.model += (
                    self.squad[p][gw]
                    == (
                        self.squad[p][gw - 1]
                        + self.transfer_in[p][gw]
                        - self.transfer_out[p][gw]
                    )
                ), f"Player {p} squad/transfer constraint for gameweek {gw}"

        # Free transfer constraints
        for gw in self.future_gameweeks:
            self.model += (
                self.free_transfers_available[gw] == (self.aux[gw] + 1)
            ), f"FTA and Aux constraint for gameweek {gw}"
            self.model += (
                self.free_transfers_available[gw - 1] - self.transfers_made[gw - 1]
                <= 2 * self.aux[gw],
                f"FTA and TM Equality 1 constraint for gameweek {gw}",
            )
            self.model += (
                self.free_transfers_available[gw - 1] - self.transfers_made[gw - 1]
                >= self.aux[gw] + (-14) * (1 - self.aux[gw]),
                f"FTA and TM Equality 2 constraint for gameweek {gw}",
            )
            self.model += (
                self.penalised_transfers[gw] >= self.transfer_diff[gw],
                f"Penalised transfers constraint for gameweek {gw}",
            )

    def _set_objective(self):
        """
        Summary:
        --------
        Function to define the objective function for the optimization problem.
        Objective function is added to model by using the += operator.

        Objective functions include:
            - Regular: Maximize total expected points over all gameweeks
            - Decay: Maximize total expected points in each gameweek, with decay factor

        Returns:
        --------
        None
        """

        gw_xp_before_pen = {
            gw: lpSum(
                [
                    self.player_xp_gw[p, gw]
                    * (
                        self.lineup[p][gw]
                        + self.captain[p][gw]
                        + 0.1 * self.vice_captain[p][gw]
                    )
                    for p in self.players
                ]
            )
            for gw in self.future_gameweeks
        }
        gw_xp_after_pen = {
            gw: (gw_xp_before_pen[gw] - 4 * self.penalised_transfers[gw])
            for gw in self.future_gameweeks
        }

        if self.objective == "regular":
            obj_func = lpSum([gw_xp_after_pen[gw] for gw in self.future_gameweeks])
            self.model += obj_func

        elif self.objective == "decay":
            obj_func = lpSum(
                [
                    gw_xp_after_pen[gw] * pow(self.decay_base, gw - self.gameweek)
                    for gw in self.future_gameweeks
                ]
            )
            self.model += obj_func
            base_str = "_base_" + str(self.decay_base)
            self.model.name += base_str

    def get_results(self) -> pd.DataFrame:
        """
        Summary:
        --------
        Function to get the results for the optimization problem.

        Returns:
        --------
        Dataframe with results from the optimization problem.
        """

        # Only return results if model is solved
        if self.model.status != 1:
            raise ValueError("Results unavailable since model is not solved.")
        else:
            results_list = []
            for gw in self.future_gameweeks:
                for p in self.players:
                    if (
                        self.squad[p][gw].varValue == 1
                        or self.transfer_out[p][gw].varValue == 1
                    ):
                        results_list.append(
                            {
                                "gw": gw,
                                "player_id": p,
                                "name": self.merged_df.loc[p, "name"],
                                "team": self.merged_df.loc[p, "team"],
                                "position": self.merged_df.loc[p, "position"],
                                "position_id": self.merged_df.loc[p, "element_type"],
                                "cost": self.player_cost[p],
                                "prob_of_appearing": self.player_prob_gw[p, gw],
                                "xp": self.player_xp_gw[p, gw],
                                "squad": self.squad[p][gw].varValue,
                                "lineup": self.lineup[p][gw].varValue,
                                "captain": self.captain[p][gw].varValue,
                                "vice_captain": self.vice_captain[p][gw].varValue,
                                "transfer_in": self.transfer_in[p][gw].varValue,
                                "transfer_out": self.transfer_out[p][gw].varValue,
                            }
                        )

            results_df = pd.DataFrame(results_list).round(2)
            results_df.sort_values(
                by=["gw", "squad", "lineup", "position_id", "xp"],
                ascending=[True, False, False, True, False],
                inplace=True,
            )
            results_df.reset_index(drop=True, inplace=True)

            return results_df

    def get_gw_xp(self):
        """
        Summary:
        --------
        Function to get the team's expected points for each gameweek.

        Returns:
        --------
        Dictionary with expected points for each gameweek.
        """
        gw_xp = {
            gw: round(
                pulp_value(
                    lpSum(
                        [
                            self.player_xp_gw[p, gw]
                            * (self.lineup[p][gw] + self.captain[p][gw])
                            for p in self.players
                        ]
                    )
                ),
                2,
            )
            for gw in self.future_gameweeks
        }

        return gw_xp

    def get_total_xp(self):
        """
        Summary:
        --------
        Function to get the team's total expected points for the horizon.

        Returns:
        --------
        Float with total expected points for the horizon.
        """
        total_xp = sum(self.gw_xp.values())
        return float(round(total_xp, 2))

    def _check_results(self, results_df: pd.DataFrame):
        """
        Summary:
        --------
        Function to check results of the optimization problem.
        If all checks are passed, prints "Results passed checks".

        Arguments:
        --------
        results_df (pd.DataFrame): Dataframe with results from optimization problem, generated by get_results() method.

        Returns:
        --------
        True if all checks are passed, otherwise raises a Warning.
        """

        if results_df is None:
            raise ValueError("No results available to check.")
        else:
            for gw in self.future_gameweeks:
                # Get results for gameweek
                gw_results = results_df[results_df["gw"] == gw]

                # Compute key metrics for gameweek
                num_players_squad = gw_results.squad.sum()
                num_players_lineup = gw_results.lineup.sum()
                num_transfers_in = gw_results.transfer_in.sum()
                num_transfers_out = gw_results.transfer_out.sum()
                num_captains = gw_results.captain.sum()
                num_vice_captains = gw_results.vice_captain.sum()
                num_captains_lineup = gw_results[
                    gw_results["captain"] == 1
                ].lineup.sum()
                num_players_team = gw_results[
                    gw_results["squad"] == 1
                ].team.value_counts()

                # Dictionary of position count for players in squad
                position_count_squad = (
                    gw_results[gw_results["squad"] == 1]
                    .position_id.value_counts()
                    .to_dict()
                )

                # Dictionary of position count for players in lineup
                position_count_lineup = (
                    gw_results[gw_results["lineup"] == 1]
                    .position_id.value_counts()
                    .to_dict()
                )

                # Conditions to check and their respective warning messages
                condition_1 = num_players_squad == 15
                condition_1_str = f"Number of players in squad for GW{gw} is not 15."

                condition_2 = num_players_lineup == 11
                condition_2_str = f"Number of players in lineup for GW{gw} is not 11."

                condition_3 = num_transfers_in == num_transfers_out
                condition_3_str = f"Number of transfers in is not equal to number of transfer out for GW{gw}."

                condition_4 = num_players_team.max() <= 3
                condition_4_str = f"Number of players from each team in not within the limit of 3 for GW{gw}."

                condition_5 = position_count_squad == self.squad_select
                condition_5_str = f"Number of players in each position in squad does not meet requirements for GW{gw}."

                condition_6a = all(
                    [
                        position_count_lineup[pos] >= self.squad_min_play[pos]
                        for pos in self.positions
                    ]
                )
                condition_6a_str = (
                    f"Number of players in each position in lineup is greater than "
                    f"the allowed range for GW{gw}."
                )

                condition_6b = all(
                    [
                        position_count_lineup[pos] <= self.squad_max_play[pos]
                        for pos in self.positions
                    ]
                )
                condition_6b_str = (
                    f"Number of players in each position in lineup is less than the allowed "
                    f"range for GW{gw}."
                )

                condition_7 = all(
                    gw_results[gw_results["squad"] == 1].prob_of_appearing >= 0.75
                )
                condition_7_str = (
                    f"Probability of appearance for each player in squad is not greater "
                    f"than 75% for GW{gw}."
                )

                condition_8 = all(
                    gw_results[gw_results["lineup"] == 1].prob_of_appearing >= 0.90
                )
                condition_8_str = (
                    f"Probability of appearance for each player in lineup is not greater"
                    f" than 90% for GW{gw}."
                )

                condition_9 = num_captains == 1
                condition_9_str = f"Number of captains is not equal to 1 for GW{gw}."

                condition_10 = num_vice_captains == 1
                condition_10_str = (
                    f"Number of vice captains is not equal to 1 for GW{gw}."
                )

                condition_11 = num_captains_lineup == 1
                condition_11_str = f"Captain is not in lineup for GW{gw}."

                # Create a nested list of conditions and their respective warning messages
                conditions = [
                    [condition_1, condition_1_str],
                    [condition_2, condition_2_str],
                    [condition_3, condition_3_str],
                    [condition_4, condition_4_str],
                    [condition_5, condition_5_str],
                    [condition_6a, condition_6a_str],
                    [condition_6b, condition_6b_str],
                    [condition_7, condition_7_str],
                    [condition_8, condition_8_str],
                    [condition_9, condition_9_str],
                    [condition_10, condition_10_str],
                    [condition_11, condition_11_str],
                ]

                # Loop through conditions and if any condition is False, add the warning message to the warnings list
                warnings = []
                for condition in conditions:
                    if not condition[0]:
                        warnings.append(condition[1])

                # If there are no warnings, return True, otherwise raise a warning
                if len(warnings) == 0:
                    return True
                else:
                    warnings_str = "\n".join(warnings)
                    raise Warning(
                        f"Results for GW{gw} did not pass checks:\n{warnings_str}"
                    )

    def get_summary(self):
        """
        Summary:
        --------
        Function to extract summary of actions over all gameweeks from the optimization problem.

        Returns:
        --------
        String with summary of actions over all gameweeks.
        """

        # Only write summary if model is solved
        if self.model.status != 1:
            print("Summary unavailable since model is not solved.")
            return None
        else:
            pass

        # Initialize summary list
        summary_list = []

        # Add summary of actions for each gameweek
        for gw in self.future_gameweeks:
            summary_list.append("-" * 50)
            summary_list.append(f"Gameweek {gw} summary:")
            summary_list.append("-" * 50)
            summary_list.append(f"Expected points: {self.get_gw_xp()[gw]}")
            summary_list.append(f"Money in bank: {self.money_in_bank[gw].varValue}")
            summary_list.append(
                f"Free transfers available: {int(self.free_transfers_available[gw].varValue)}"
            )
            summary_list.append(
                f"Transfers made: {int(pulp_value(self.transfers_made[gw]))}"
            )
            summary_list.append(
                f"Penalised transfers: {int(self.penalised_transfers[gw].varValue)}"
            )

            for p in self.players:
                if self.transfer_in[p][gw].varValue == 1:
                    player_name = self.merged_df.loc[p, "name"]
                    team_name = self.merged_df.loc[p, "team"]
                    summary_list.append(
                        f"Player {p} ({player_name} @ {team_name}) transferred in."
                    )
                if self.transfer_out[p][gw].varValue == 1:
                    player_name = self.merged_df.loc[p, "name"]
                    team_name = self.merged_df.loc[p, "team"]
                    summary_list.append(
                        f"Player {p} ({player_name} @ {team_name}) transferred out."
                    )

        # Join the summary list into a single string
        full_summary = "\n".join(summary_list)

        return full_summary


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Parameters to optimize over
    team_id = 10599528
    horizons = [1, 2, 3]
    objectives = ["regular"]
    gameweek = 24

    # Create and solve optimization problem for each combination of horizon and objective
    optimizers = [
        OptimizeMultiPeriod(
            team_id=team_id,
            gameweek=gameweek,
            horizon=h,
            objective=objective,
        )
        for h in horizons
        for objective in objectives
    ]

    for optimizer in optimizers:
        print("-" * 100)
        print(
            f"Optimizing for team {optimizer.team_id}, gameweek {optimizer.gameweek}, "
            f"horizon {optimizer.horizon}, objective {optimizer.objective}"
        )
        print("-" * 100)
        optimizer.solve_problem()
        solved_model = optimizer.model
        results = optimizer.get_results()
        summary = optimizer.get_summary()
        name = optimizer.model.name

        if results is not None:
            solved_model.writeLP(f"../../models/multi_period/{name}_model.lp")
            results.to_csv(f"../../data/results/{name}_results.csv", index=False)

            with open(f"../../reports/{name}_summary.txt", "w") as f:
                f.write(summary)
                f.close()

    print("*" * 100)
    print("Total seconds in loop:", round(time.time() - t0, 2))
    print("*" * 100)
