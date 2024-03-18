import pandas as pd
import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

from src.models.PrepareOptimiser import PrepareDatasetForOptimiser


class FantasyOptimiser:
    """

    The `FantasyOptimiser` class is responsible for solving the optimization problem for fantasy football team selection.

    Attributes:
    - `dataset` (object): The dataset object containing all the necessary data for optimization.
    - `team_id` (int): The team ID for which the optimization is being performed.
    - `horizon` (int): The number of future gameweeks to be considered for optimization.
    - `model` (object): The optimization model.
    - `results` (DataFrame): The results dataframe containing the optimal team selection.
    - `summary` (str): A summary of the optimization results.

    Methods:
    - `solve()`: Solves the optimization problem and generates the optimal team selection and transfers. -
    - `get_results()`: Returns a dataframe containing the optimal team selection, captain and
                     vice-captain choices and transfers
    - `get_summary()`: Returns a summary of actions for each gameweek.
    - `get_model()`: Returns the solved model of the optimization problem.

    """
    def __init__(self):
        # Prepare and store dataset for optimization
        self.dataset = PrepareDatasetForOptimiser().dataset

        # Unpack problem parameters
        self.team_id = self.dataset.team_id
        self.horizon = self.dataset.horizon

        # Placeholders
        self.model = None
        self.results = pd.DataFrame()
        self.summary = ""

    def solve(self):
        # Access the prepared dataset
        print("Accessing Fantasy Dataset...")
        players = self.dataset.players
        teams = self.dataset.teams
        positions = self.dataset.positions
        current_gameweek = self.dataset.current_gameweek
        future_gameweeks = self.dataset.future_gameweeks
        all_gameweeks = self.dataset.all_gameweeks
        initial_team = self.dataset.initial_team
        bank_balance = self.dataset.bank_balance
        next_free_transfers = self.dataset.free_transfers
        squad_min_play = self.dataset.squad_min_play
        squad_max_play = self.dataset.squad_max_play
        squad_select = self.dataset.squad_select

        # Defining the optimisation problem
        print("Starting optimisation...")
        prob = LpProblem(name="FantasyOptimiser", sense=LpMaximize)

        # Problem variables
        print("Defining model variables...")
        squad = LpVariable.dicts("Squad", (players, all_gameweeks), cat="Binary")
        lineup = LpVariable.dicts("Lineup", (players, future_gameweeks), cat="Binary")
        captain = LpVariable.dicts("Captain", (players, future_gameweeks), cat="Binary")
        vice_captain = LpVariable.dicts(
            "Vice Captain", (players, future_gameweeks), cat="Binary"
        )
        transfer_in = LpVariable.dicts(
            "Transferred In", (players, future_gameweeks), cat="Binary"
        )
        transfer_out = LpVariable.dicts(
            "Transferred Out", (players, future_gameweeks), cat="Binary"
        )
        money_in_bank = LpVariable.dicts(
            "Bank Balance", all_gameweeks, cat="Continuous", lowBound=0
        )
        free_transfers = LpVariable.dicts(
            "Free Transfers Available",
            all_gameweeks,
            cat="Integer",
            lowBound=1,
            upBound=2,
        )
        penalised_transfers = LpVariable.dicts(
            "Penalised Transfers", future_gameweeks, cat="Integer", lowBound=0
        )
        auxiliary_variable = LpVariable.dicts(
            "Auxiliary Variable", future_gameweeks, cat="Binary"
        )

        # Counters and trackers to use for defining constraints and initial conditions
        print("Defining counters...")
        squad_counter = {
            gw: lpSum([squad[p][gw] for p in players]) for gw in future_gameweeks
        }
        lineup_counter = {
            gw: lpSum([lineup[p][gw] for p in players]) for gw in future_gameweeks
        }
        squad_pos_counter = {
            (pos, gw): lpSum(
                [squad[p][gw] for p in players if self._get_player_position(p) == pos]
            )
            for pos in positions
            for gw in future_gameweeks
        }
        lineup_pos_counter = {
            (pos, gw): lpSum(
                [lineup[p][gw] for p in players if self._get_player_position(p) == pos]
            )
            for pos in positions
            for gw in future_gameweeks
        }
        team_counter = {
            (team, gw): lpSum(
                [squad[p][gw] for p in players if self._get_player_team(p) == team]
            )
            for team in teams
            for gw in future_gameweeks
        }
        revenue = {
            gw: lpSum([transfer_out[p][gw] * self._get_player_price(p) for p in players])
            for gw in future_gameweeks
        }
        expenditure = {
            gw: lpSum([transfer_in[p][gw] * self._get_player_price(p) for p in players])
            for gw in future_gameweeks
        }
        transfer_counter = {
            gw: lpSum([transfer_in[p][gw] for p in players]) for gw in future_gameweeks
        }
        transfer_diff = {
            gw: (transfer_counter[gw] - free_transfers[gw]) for gw in future_gameweeks
        }

        # Initial conditions
        print("Defining initial conditions...")
        transfer_counter[current_gameweek] = 1
        prob += (
            money_in_bank[current_gameweek] == bank_balance,
            f"Initial money in bank is {bank_balance}",
        )
        prob += (
            free_transfers[current_gameweek] == next_free_transfers,
            f"Initial number of free transfers is {next_free_transfers}",
        )
        for p in players:
            if p in initial_team:
                prob += (
                    squad[p][current_gameweek] == 1,
                    f"Player in the initial squad constraint for player {p}",
                )
            else:
                prob += (
                    squad[p][current_gameweek] == 0,
                    f"Player not in the initial squad constraint for player {p}",
                )

        print("Defining model constraints...")
        for gw in future_gameweeks:
            # Squad and lineup constraints:
            prob += (
                squad_counter[gw] == 15,
                f"Squad count must be 15 constraint for gameweek {gw}",
            )
            prob += (
                lineup_counter[gw] == 11,
                f"Lineup count must be 11 constraint for gameweek {gw}",
            )
            for p in players:
                prob += (
                    lineup[p][gw] <= squad[p][gw],
                    f"Lineup player must also be squad player constraint for player {p} in gameweek {gw}",
                )

            # Captain and vice captain constraints:
            prob += (
                lpSum([captain[p][gw] for p in players]) == 1,
                f"Only 1 captain constraint for gameweek {gw}",
            )
            prob += (
                lpSum([vice_captain[p][gw] for p in players]) == 1,
                f"Only 1 vice captain count constraint for gameweek {gw}",
            )

            for p in players:
                prob += (
                    captain[p][gw] + vice_captain[p][gw] <= 1,
                    f"Captain and vice captain mutual exclusion constraint for player {p} in gameweek {gw}",
                )
                prob += (
                    captain[p][gw] <= lineup[p][gw],
                    f"Captain must be in lineup constraint for player {p} in gameweek {gw}",
                )
                prob += (
                    vice_captain[p][gw] <= lineup[p][gw],
                    f"Vice captain must be in lineup constraint for player {p} in gameweek {gw}",
                )

            # Position and formation constraints:
            for pos in positions:
                prob += (
                    lineup_pos_counter[pos, gw] >= squad_min_play[pos],
                    f"Minimum number of lineup players in position {pos} in gameweek constraint {gw}",
                )
                prob += (
                    lineup_pos_counter[pos, gw] <= squad_max_play[pos],
                    f"Maximum number of lineup players in position {pos} in gameweek constraint {gw}",
                )
                prob += (
                    squad_pos_counter[pos, gw] == squad_select[pos],
                    f"Number of squad players in position {pos} in gameweek {gw} constraint",
                )

            # Team constraints:
            for team in teams:
                prob += (
                    team_counter[team, gw] <= 3,
                    f"Only a maximum of 3 players from team {team} in gameweek {gw} constraint",
                )

            # Probability of appearance constraints:
            for p in players:
                prob += (
                    lineup[p][gw] <= (self._get_player_prob(p, gw) >= 0.90),
                    f"Probability of lineup player {p} in gameweek {gw} must be at least 90%",
                )
                prob += (
                    squad[p][gw] <= (self._get_player_prob(p, gw) >= 0.75),
                    f"Probability of squad player {p} in gameweek {gw} must be at least 75%",
                )

            # Budgeting constraints:
            prob += (
                money_in_bank[gw]
                == (money_in_bank[gw - 1] + revenue[gw] - expenditure[gw]),
                f"Bank balance constraint for gameweek {gw}",
            )

            # Transfer constraints:
            prob += (
                transfer_counter[gw] <= 20,
                f"Maximum of 20 transfers allowed for gameweek {gw}",
            )
            for p in players:
                prob += (
                    squad[p][gw]
                    == (squad[p][gw - 1] + transfer_in[p][gw] - transfer_out[p][gw]),
                    f"Squad player {p} in gameweek {gw} must be in previous GW squad or transferred in this GW",
                )

            prob += (
                free_transfers[gw] == (auxiliary_variable[gw] + 1),
                f"First Free transfers allowed and Auxiliary variable relationship for gameweek {gw}",
            )
            prob += (
                free_transfers[gw - 1] - transfer_counter[gw - 1]
                <= 2 * (auxiliary_variable[gw]),
                f"Second Free transfers allowed and Auxiliary variable relationship for gameweek {gw}",
            )
            prob += (
                penalised_transfers[gw] >= transfer_diff[gw],
                f"Penalised transfers constraint for gameweek {gw}",
            )

        # Dictionary of team xp for each gameweek (x2 for captain and x1.1 for vice captain)
        team_gameweek_xp = {
            gw: lpSum(
                [
                    (
                        self._get_player_xp(p, gw)
                        * (lineup[p][gw] + captain[p][gw] + 0.1 * vice_captain[p][gw])
                    )
                    - (4 * penalised_transfers[gw])
                    for p in players
                ]
            )
            for gw in future_gameweeks
        }

        # Defining the objective function (team's total xp across all gameweeks in the horizon)
        objective_func = lpSum([team_gameweek_xp[gw] for gw in future_gameweeks])
        prob += objective_func

        # Solve the optimisation problem
        print("Solving model...")
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # If optimal solution is found, get the results
        print("Getting results...")
        if LpStatus[status] != "Optimal":
            raise ValueError(f"Solution could not be found: {LpStatus[status]}")
        else:
            pass

        optimized_results_list = []
        for gw in future_gameweeks:
            for p in players:
                if squad[p][gw].varValue == 1 or transfer_out[p][gw].varValue == 1:
                    optimized_results_list.append(
                        {
                            "GW": gw,
                            "ID": p,
                            "Name": self._get_player_name(p),
                            "Team": self._get_player_team(p),
                            "Pos": self._get_player_position(p),
                            "Price": self._get_player_price(p),
                            "Prob of Appearing": self._get_player_prob(p, gw),
                            "xP": self._get_player_xp(p, gw),
                            "In Squad": squad[p][gw].varValue,
                            "In Lineup": lineup[p][gw].varValue,
                            "Is Captain": captain[p][gw].varValue,
                            "Is Vice Captain": vice_captain[p][gw].varValue,
                            "Transferred In": transfer_in[p][gw].varValue,
                            "Transferred Out": transfer_out[p][gw].varValue,
                        }
                    )

        optimized_results_df = pd.DataFrame(optimized_results_list).round(2)
        optimized_results_df.sort_values(
            by=["GW", "In Squad", "In Lineup", "Pos", "xP"],
            ascending=[True, False, False, True, False],
            inplace=True,
        )
        optimized_results_df.reset_index(drop=True, inplace=True)

        # Check results: if pass, update placeholders and generate summary of actions
        print("Checking results...")
        if self._check_results(optimized_results_df):
            self.results = optimized_results_df
            self.model = prob

            summary_list: list[str] = []

            for gw in self.dataset.future_gameweeks:
                gw_results = self.results[self.results["GW"] == gw]
                summary_list.append("=" * 20)
                summary_list.append(f"Gameweek: {gw}")
                summary_list.append("=" * 20)
                summary_list.append(
                    f"Total Expected Points: {round(gw_results[gw_results['In Lineup'] == 1]['xP'].sum() +
                                                    gw_results[gw_results['Is Captain'] == 1]['xP'].item(), 2)}"
                )
                summary_list.append(
                    f"Total Cost: {round(gw_results[gw_results['In Squad'] == 1]['Price'].sum(), 2)}"
                )
                summary_list.append(
                    f"Captain: {gw_results[gw_results['Is Captain'] == 1]['Name'].item()}"
                )
                summary_list.append(
                    f"Vice Captain: {gw_results[gw_results['Is Vice Captain'] == 1]['Name'].item()}"
                )
                summary_list.append(
                    f"Bank Balance: {round(money_in_bank[gw].varValue, 2)}"
                )
                summary_list.append(
                    f"Free Transfers Available: {free_transfers[gw].varValue}"
                )
                summary_list.append(f"Transfers Made: {value(transfer_counter[gw])}")
                summary_list.append(
                    f"Penalised Transfers: {penalised_transfers[gw].varValue}"
                )
                summary_list.append(
                    f"Players Transferred In: {gw_results[gw_results['Transferred In'] == 1]['Name'].values}"
                )
                summary_list.append(
                    f"Players Transferred Out: {gw_results[gw_results['Transferred Out'] == 1]['Name'].values}"
                )

            summary_str = "\n".join(summary_list)

            self.summary = summary_str

            print("Optimisation complete.")

    def get_model(self) -> LpProblem:
        if self.model is None:
            raise Exception("No optimal model found.")
        else:
            return self.model

    def get_results(self) -> pd.DataFrame:
        if len(self.results) == 0:
            raise Exception("No results found.")
        else:
            return self.results

    def get_summary(self) -> str:
        if len(self.summary) == 0:
            raise Exception("No summary found.")
        else:
            return self.summary

    def _get_player_name(self, player_id: int) -> str:
        df = self.dataset.predictions
        return df[df["id"] == player_id]["name"].values[0]

    def _get_player_position(self, player_id: int) -> int:
        df = self.dataset.predictions
        return df[df["id"] == player_id]["position"].values[0]

    def _get_player_team(self, player_id: int) -> int:
        df = self.dataset.predictions
        return df[df["id"] == player_id]["team"].values[0]

    def _get_player_price(self, player_id: int) -> float:
        df = self.dataset.predictions
        return df[df["id"] == player_id]["price"].values[0]

    def _get_player_prob(self, player_id: int, gameweek: int) -> float:
        df = self.dataset.predictions
        return df[(df["id"] == player_id)][f"{gameweek}_prob"].values[0]

    def _get_player_xp(self, player_id: int, gameweek: int) -> float:
        df = self.dataset.predictions
        return df[(df["id"] == player_id)][f"{gameweek}_xp"].values[0]

    def _check_results(self, results: pd.DataFrame) -> bool:
        for gw in self.dataset.future_gameweeks:
            gw_results = results[results["GW"] == gw]
            num_squad_players = gw_results["In Squad"].sum()
            num_lineup_players = gw_results["In Lineup"].sum()
            num_transfers_in = gw_results["Transferred In"].sum()
            num_transfers_out = gw_results["Transferred Out"].sum()
            num_captains = gw_results["Is Captain"].sum()
            num_vice_captains = gw_results["Is Vice Captain"].sum()
            num_teams = gw_results[gw_results["In Squad"] == 1]["Team"].value_counts()
            num_positions_squad = (
                gw_results[gw_results["In Squad"] == 1]["Pos"].value_counts().to_dict()
            )
            num_positions_lineup = (
                gw_results[gw_results["In Lineup"] == 1]["Pos"].value_counts().to_dict()
            )

            # Conditions to check and their respective warning messages
            condition_1 = num_squad_players == 15
            condition_1_warning = f"Number of players in squad for GW{gw} is not 15."

            condition_2 = num_lineup_players == 11
            condition_2_warning = f"Number of players in lineup for GW{gw} is not 11."

            condition_3 = num_transfers_in == num_transfers_out
            condition_3_warning = f"Number of transfers in is not equal to number of transfer out for GW{gw}."

            condition_4 = num_teams.max() <= 3
            condition_4_warning = f"Number of players from each team in not within the limit of 3 for GW{gw}."

            condition_5 = num_positions_squad == self.dataset.squad_select
            condition_5_warning = f"Number of players in each position in squad does not meet requirements for GW{gw}."

            condition_6a = all(
                [
                    num_positions_lineup[pos] >= self.dataset.squad_min_play[pos]
                    for pos in self.dataset.positions
                ]
            )
            condition_6a_warning = (
                f"Number of players in each position in lineup is greater than "
                f"the allowed range for GW{gw}."
            )

            condition_6b = all(
                [
                    num_positions_lineup[pos] <= self.dataset.squad_max_play[pos]
                    for pos in self.dataset.positions
                ]
            )
            condition_6b_warning = (
                f"Number of players in each position in lineup is less than the allowed "
                f"range for GW{gw}."
            )

            condition_7 = all(
                gw_results[gw_results["In Squad"] == 1]["Prob of Appearing"] >= 0.75
            )
            condition_7_warning = (
                f"Probability of appearance for each player in squad is not greater "
                f"than 75% for GW{gw}."
            )

            condition_8 = all(
                gw_results[gw_results["In Lineup"] == 1]["Prob of Appearing"] >= 0.90
            )
            condition_8_warning = (
                f"Probability of appearance for each player in lineup is not greater"
                f" than 90% for GW{gw}."
            )

            condition_9 = num_captains == 1
            condition_9_warning = f"Number of captains is not equal to 1 for GW{gw}."

            condition_10 = num_vice_captains == 1
            condition_10_warning = (
                f"Number of vice captains is not equal to 1 for GW{gw}."
            )

            # Create a nested list of conditions and their respective warning messages
            conditions = [
                [condition_1, condition_1_warning],
                [condition_2, condition_2_warning],
                [condition_3, condition_3_warning],
                [condition_4, condition_4_warning],
                [condition_5, condition_5_warning],
                [condition_6a, condition_6a_warning],
                [condition_6b, condition_6b_warning],
                [condition_7, condition_7_warning],
                [condition_8, condition_8_warning],
                [condition_9, condition_9_warning],
                [condition_10, condition_10_warning],
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


if __name__ == "__main__":
    # Initialise optimiser
    opt = FantasyOptimiser()

    # Solve the optimisation problem
    opt.solve()

    # Export solved model, results and summary of actions
    model = opt.get_model()
    results = opt.get_results()
    summary = opt.get_summary()
    print(summary)

    model.writeLP(f"../../models/fpl_optimiser_{opt.team_id}_horizon_{opt.horizon}")
    results.to_csv(f"../../data/results/{opt.team_id}_horizon_{opt.horizon}")
    with open(f"../../reports/summary_{opt.team_id}_horizon_{opt.horizon}", "w") as f:
        f.write(summary)
