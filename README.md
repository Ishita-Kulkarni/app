# Decon Simulator

A Flask web app for agent selection, dermal penetration modeling, and decontamination visualization. It can optionally read a CSV `CWA_with_SMILES_using_CAS2.csv` for agent properties and falls back to PubChem for missing data.

## Setup

```bash
# From repo root
cd decon_simulator

# Create and activate a virtual env (Linux)
python3 -m venv .venv
. .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
# From decon_simulator/
FLASK_APP=app.py flask run  # or python app.py
```

Then open http://127.0.0.1:5000/ in your browser.

## Notes
- If `CWA_with_SMILES_using_CAS2.csv` is present in the working directory, it will be used to pre-populate agent data. Otherwise defaults and PubChem lookups are used.
- Matplotlib uses the non-interactive backend (`Agg`) and plots are embedded as base64 images.
- If PubChem requests fail (e.g., offline), simulations still run with sensible fallbacks.
