# Special Course in Space Physics
Repository for the MSc Special Course in Space Physics held within the SOLER project at the University of Turku in May 2025

## Usage

You can run the Jupyter Notebooks either on the project's JupyterHub server or locally on your computer. If you are new to Jupyter Notebooks, the official documentation will give you more info about [What is the Jupyter Notebook?](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.html) and [Running Code](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html) with it.

### On SOLER's JupyterHub

You can run the Notebooks online on SOLER's JupyterHub (more info at [soler-horizon.eu/hub](https://soler-horizon.eu/hub)). For this you only need a [free GitHub account](https://github.com/signup) for verification. You can directly access the special course folder by [following this link](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/MSc-course-2025/). There, open the *.ipynb* files within the separate folders for the specific tools.

### Locally

#### Install locally

1. These tools require a recent Python (>=3.10) installation. [Following SunPy's approach, we recommend installing Python via miniforge (click for instructions).](https://docs.sunpy.org/en/stable/tutorial/installation.html#installing-python)
2. [Download this file](https://github.com/soler-he/MSc-course-2025/archive/refs/heads/main.zip) and extract to a folder of your choice (or clone the repository [https://github.com/soler-he/MSc-course-2025](https://github.com/soler-he/MSc-course-2025) if you know how to use `git`).
3. Open a terminal or the miniforge prompt and move to the directory where the code is.
4. Create a new virtual environment (e.g., `conda create --name msc_course_2025 python=3.12`).
5. Activate the just created virtual environment (e.g., `conda activate msc_course_2025`).
6. If you **don't** have `git` installed (try executing it), install it with `conda install conda-forge::git`.
7. Install the Python dependencies from the *requirements.txt* file with `pip install -r requirements.txt`

#### Use locally

Activate the created virtual environment in the terminal (step 5. of [Installation](#installation)), go to the folder where the tools have been extracted to, and run `jupyter-lab`. This will open the default web-browser. There, open the *.ipynb* files within the separate folders for the specific tools.


