<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/dannybozbay/fpl-optimization">
    <img src="https://cf-images.eu-west-1.prod.boltdns.net/v1/static/2508689025001/001172a9-87d5-4723-b991-59d47e89d7bb/87ae00c7-6916-47f4-abca-7f6fe10a5684/1280x720/match/image.jpg" alt="Logo" width="500" height="275">
  </a>

<h3 align="center">Winning FPL With Linear Optimisation</h3>

  <p align="center">
    Harnessing the power of linear programming to optimise your team selection, captain choice and transfers!
    <br />
    <a href="https://github.com/dannybozbay/fpl-optimization"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/dannybozbay/fpl-optimization">View Demo</a>
    ·
    <a href="https://github.com/dannybozbay/fpl-optimization/issues">Report Bug</a>
    ·
    <a href="https://github.com/dannybozbay/fpl-optimization/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites-and-installation">Prerequisites and Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>




<!-- ABOUT THE PROJECT -->

## About The Project

Winning your fantasy football league can be a challenging and sometimes frustrating journey—but when you get it right, it’s incredibly rewarding. The path to victory often requires nailing multiple decisions, consistently. The key elements? Your initial team selection, weekly captain choice, and timely transfers. Yet, how often do we rely purely on instinct for these decisions? Why guess when we have a wealth of data at our fingertips?

This project leverages that data with PuLP, an open-source linear programming package in Python, to maximize your Fantasy Premier League team's expected points. By setting up constraints and objectives, the algorithm optimizes your lineup, captain, and transfers for each gameweek. Say goodbye to blind guessing and hello to data-driven success!


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->

## Installation

To get started with the optimiser, follow these steps to set up your development environment:

**1. Clone the Repository**

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/dannybozbay/FPL-Optimization.git
```

**2. Navigate to the Project Directory**

Change to the project directory:

```bash
cd FPL-Optimization
```

**3. Create a Python Virtual Environment**

It's recommended to use a virtual environment to manage dependencies. Create a virtual environment with venv:

```bash
python3 -m venv venv
```

Activate the virtual environment:

* On **Windows**:

    ```bash
    venv\Scripts\activate
    ```

* On **macOS/Linux**:

    ```bash
   source venv/bin/activate
    ```

**4. Install Dependencies**

Install the required packages listed in *requirements.txt*:

```bash
pip install -r requirements.txt
```

**5. Download Points Projection Data**

Download the latest points projection from [FPLForm](https://fplform.com/export-fpl-form-data) and save it inside `data/external` with the name `fpl-form-predicted-points.csv`. Be sure to set the upper range of gameweeks to *GW3* and select *With Extra Columns* before generating the CSV file.

**6. Start Using the Project**

You are now ready to use the optimiser. Follow the documentation and examples provided in the repository to get started.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

After installing the optimiser, you can run it by following these steps:

1. Set your FPL Team ID (`team_id`) and the number of gameweeks you wish to optimise for (`horizon`) in
   the `settings.json` file and save it.

2. Run  the script `run_optimiser.py`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

Running the optimiser will provide the following:

- CSV table located under `data/results` with information about your optimal lineup, captain and transfers (both in and
  out) for each gameweek.

- TXT file located under `reports` containing a summary of actions for each gameweek, including the teams total expected
  points, its cost, its captain and its transfers.
- Solved LP model located under `models`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Email: dannybozbay@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

A special thanks to Sertalpbila for inspiring this project with his own work and for providing invaluable video tutorials on how to approach the problem. If you're looking to tackle this challenge yourself, I highly recommend checking them out!

* [https://github.com/sertalpbilal/FPL-Optimization-Tools](sertal)
* [https://www.youtube.com/playlist?list=PLrIyJJU8_viOags1yudB_wyafRuTNs1Ed]()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/dannybozbay/fpl-optimization.svg?style=for-the-badge

[contributors-url]: https://github.com/dannybozbay/fpl-optimization/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/dannybozbay/fpl-optimization.svg?style=for-the-badge

[forks-url]: https://github.com/dannybozbay/fpl-optimization/ƒƒƒƒnetwork/members

[stars-shield]: https://img.shields.io/github/stars/dannybozbay/fpl-optimization.svg?style=for-the-badge

[stars-url]: https://github.com/dannybozbay/fpl-optimization/stargazers

[issues-shield]: https://img.shields.io/github/issues/dannybozbay/fpl-optimization.svg?style=for-the-badge

[issues-url]: https://github.com/dannybozbay/fpl-optimization/issues

[license-shield]: https://img.shields.io/github/license/dannybozbay/fpl-optimization.svg?style=for-the-badge

[license-url]: https://github.com/dannybozbay/fpl-optimization/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/dannybozbay

[product-screenshot]: images/screenshot.png

[Python.js]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54

[Python-url]: https://www.python.org

[Pandas.js]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white

[Pandas-url]: https://pandas.pydata.org

