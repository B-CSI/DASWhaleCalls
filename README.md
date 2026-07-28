# DAS Whale Calls - Workflow
This project provides scripts to test the End-to-End Workflow for Fin Whale Song Detection, Note Characterization, and Localization described in:

Diego-Tortosa, D.; Romagosa, M.; Ugalde, A.; Latorre, H.; Ventosa, S.; García, J.E.; Villaseñor, A. (2026) **An End-to-End Workflow for Fin Whale Song Detection, Note Characterization, and Localization with Distributed Acoustic Sensing**. ArXiv: [URL]

If you use this repository in your research, a citation to the manuscript would be appreciated.

## Requirements
After cloning the repository:
```
git clone https://github.com/B-CSI/DASWhaleCalls.git
cd DASWhaleCalls
```

Python >= 3.11 is required. It is recommended to use a virtual environment:
```
# Create virtual environment:
python -m venv venv

# Activate environment (Linux / macOS):
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Main notebook

An example workflow is implemented in `DASWhaleCalls_example.ipyn`. 
This notebook serves as the main entry point of the project and some functionalities in a reproducible step-by-step manner.