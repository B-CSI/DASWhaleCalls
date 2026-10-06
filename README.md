# DAS Whale Calls - Workflow
This project provides scripts to test the End-to-End Workflow for Fin Whale Song Detection, Note Characterization, and Localization described in:

Diego-Tortosa, D.; Romagosa, M.; Ugalde, A.; Latorre, H.; Ventosa, S.; García, J.E.; Villaseñor, A. (2026) **An End-to-End Workflow for Fin Whale Song Detection, Note Characterization, and Localization with Distributed Acoustic Sensing**. [arXiv:2608.02387](https://doi.org/10.48550/arXiv.2608.02387)


If you use this repository in your research, a citation to the manuscript would be appreciated.

# 🚧 Work in progress — example workflow under development

This repository is currently under development. The example workflow and accompanying scripts are being progressively refined, and therefore the code, file structure, parameters, and analysis steps may change substantially over the coming months(end of 2026).

These changes concern the implementation and presentation of the example workflow and should not be interpreted as changes to the underlying method itself.

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



## Contributing & License Expectations

Licensed under **AGPL-3.0**.

**✅ Feel free to** fork, use, modify, redistribute, and deploy this project.

**🚫 If you distribute or deploy modified versions**, you must comply with the AGPL-3.0 by making the corresponding source code available and preserving the license and copyright notices.

Thank you for helping keep this project open, transparent, and beneficial for everyone.
