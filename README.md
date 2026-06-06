# Yukawa Potential — Numerical Solvers and Analysis

This repository contains code, libraries, and Jupyter notebooks used to explore numerical solutions for the quantum-mechanical Yukawa (screened Coulomb) potential. The work is organized as an academic computational physics project: research scripts, numerical solvers, plotting utilities, and writeups for analysis and figures.

**Project status:** exploratory research code. See the "Work in progress" section below for limitations and TODOs.

**Main goals**
- Provide numerical methods and experiments for bound states and scattering in the Yukawa potential.
- Collect reusable numerical utilities (finite-element routines, ODE/root solvers, variable-phase methods, FFT helpers) for physics problems.
- Produce figures and notebooks used in an academic report and presentations.

Contents and main features
- Jupyter notebooks that explore solver behavior and generate figures.
- A small Python library of numerical tools (FEM, root finding, ODE wrappers, VPM helpers, FFT utilities).
- Example scripts that run or demonstrate solvers and plotting routines.

Repository structure
- code/: primary analysis and notebooks
  - ai/: experimental AI-assisted notebooks and helper scripts
    - ipynb/: exploration notebooks (chatgpt-* and claude-*)
    - py/: small experiment scripts (e.g. chatgpt-solver.py)
  - ipynb/: organized solver notebooks and auxiliary notebooks used for figures
  - libraries/: local reusable libraries (jwanglibs) used across scripts
  - py/: runnable Python modules and solver scripts used for experiments and plotting
- srccode/: standalone source utilities (e.g. hydrogen.py)
- documents/: LaTeX report and bibliography used to build the written report
- presentations/: Beamer/LaTeX presentation sources and plots

Key files (examples)
- code/ai/py/chatgpt-solver.py — experimental solver script
- code/libraries/jwanglibs/ — local numerical libraries (fem.py, vpm.py, rootfinder.py, etc.)
- code/ipynb/solver*/ — collections of Jupyter notebooks that run experiments and produce plots
- documents/report/ — LaTeX project for the written report

Requirements / dependencies
The project uses standard scientific Python libraries. At minimum, install:

- Python 3.8 or newer
- numpy
- scipy
- matplotlib
- jupyter or jupyterlab

Optional (used in parts of the repo):
- pandas (if you prefer tabular data manipulation)
- numba (optional performance speedups)
- LaTeX (to build the report in `documents/`)

How to set up
1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install core dependencies:

```bash
pip install --upgrade pip
pip install numpy scipy matplotlib jupyter
```

3. (Optional) Install other tools used in the repo:

```bash
pip install pandas numba
# To build the LaTeX report, install a TeX distribution (TeX Live / MacTeX)
```

How to run the code
- Notebooks: start Jupyter Lab or Notebook in the repository root and open notebooks under `code/ipynb/` or `code/ai/ipynb/`:

```bash
jupyter lab
```

- Scripts: many analysis scripts live under `code/py/` and `code/ai/py/`. Run a script with Python, for example:

```bash
python code/ai/py/chatgpt-solver.py
```

Example usage
- Open `code/ipynb/solver1/solver1.0.ipynb` (or other solver notebooks) and run the cells to reproduce experiments and plots.
- Use the utilities in `code/libraries/jwanglibs/` from other scripts; e.g.:

```python
from jwanglibs import fem, rootfinder
# construct a problem, call solver routines, plot results with matplotlib
```

Notes, limitations, and work in progress
- The repository is research-oriented; scripts and notebooks are exploratory and not packaged as a polished library.
- There is no pinned `requirements.txt` — consider generating one from your environment (`pip freeze > requirements.txt`) if you need reproducible installs.
- Some scripts rely on local paths or ad-hoc data files. Before running, check notebook top cells and script headers for path assumptions.
- TODO: add a consolidated `examples/` folder with small, runnable demos and a `requirements.txt`.

Author / attribution
- Charlie M. (author/maintainer)
- For questions or to contribute, open an issue or submit a pull request.

License
- No license provided. Add a license file if you wish to make this project open-source.

Contact
- See repository owner and GitHub profile for contact details.

--
Generated from the repository contents; please review and adjust any script names or dependency versions before using in production. TODO: add `requirements.txt` and example runner scripts.
