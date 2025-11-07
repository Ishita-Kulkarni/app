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

## Run Locally

```bash
# From decon_simulator/
python app.py
```

Then open http://127.0.0.1:5000/ in your browser.

## Deploy to Render

### Option 1: Using render.yaml (Recommended)
1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Blueprint"
4. Connect your GitHub repository
5. Render will automatically detect `render.yaml` and configure the service

### Option 2: Manual Setup
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: decon-simulator
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.11.0 (or latest)
5. Click "Create Web Service"

### Environment Variables (Optional)
If you want to use a custom CSV file location:
- Key: `DECON_CSV_PATH`
- Value: `/path/to/your/CWA_with_SMILES_using_CAS2.csv`

## Notes
- If `CWA_with_SMILES_using_CAS2.csv` is present in the working directory, it will be used to pre-populate agent data. Otherwise defaults and PubChem lookups are used.
- Matplotlib uses the non-interactive backend (`Agg`) and plots are embedded as base64 images.
- If PubChem requests fail (e.g., offline), simulations still run with sensible fallbacks.
- The app automatically detects the PORT environment variable set by Render.
