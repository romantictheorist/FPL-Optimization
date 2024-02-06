<br/>
<p align="center">
  <a href="https://github.com/DannyBozbay/CERN-Gradient-Boosting">
    <img src="https://e0.365dm.com/22/08/2048x1152/skysports-fantasy-premier-league_5864666.jpg?20220814171817" alt="Logo">
  </a>

  <h3 align="center">Optimizing Fantasy Premier League</h3>

  <p align="center">
    Using linear programming via PuLP to solve for optimal lineups, captains and transfers.
    <br/>
    <br/>
    <a href="https://github.com/DannyBozbay/CERN-Gradient-Boosting"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/DannyBozbay/CERN-Gradient-Boosting/total) ![Contributors](https://img.shields.io/github/contributors/DannyBozbay/CERN-Gradient-Boosting?color=dark-green) ![Issues](https://img.shields.io/github/issues/DannyBozbay/CERN-Gradient-Boosting)

## Table Of Contents

* [About the Project](#about-the-project)
* [Installation](#installation)
* [License](#license)
* [Authors](#authors)

## About The Project

The goals of this project are:

- **Optimizing Fantasy Premier League Team Performance**: The primary goal of this project is to use linear programming techniques, implemented through PuLP in Python, to maximize the expected points of a Fantasy Premier League (FPL) team. By formulating constraints and objectives, the algorithm aims to select the most effective lineup, captain, and potential transfers for each gameweek.

- **Incorporating FPL Rules and Custom Constraints**: This project successfully integrates FPL rules, such as limitations on the number of players from each team, into the optimization process. Additionally, custom constraints, like ensuring a minimum probability of player appearances, are incorporated to enhance the realism and effectiveness of the optimized team selections.

- **Enhancing Decision-Making and Performance Prediction**: By leveraging linear programming and probability calculations, this project aids FPL managers in making informed decisions about their team composition and strategy. By optimizing expected points while adhering to FPL regulations and strategic preferences, the project aims to improve team performance over the course of the season.





## Installation

1. Clone the repository

```bash
git clone https://github.com/dannybozbay/FPL-Optimization.git
```

2. Create a Python (3.11 or higher) environment with required dependencies:


```bash
cd fpl-optimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## License

Distributed under the MIT License. See [LICENSE](https://github.com/DannyBozbay/CERN-Gradient-Boosting/blob/main/LICENSE.md) for more information.

## Authors

* **Danny Bozbay** - *Theoretical Physics Graduate* - [Danny Bozbay](https://github.com/DannyBozbay/)


