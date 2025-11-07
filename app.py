import os
import io
import base64
import logging
from functools import lru_cache
import warnings

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import pubchempy as pcp
from pubchempy import PubChemPyDeprecationWarning
warnings.filterwarnings("ignore", category=PubChemPyDeprecationWarning)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# -------------------------
# Global plotting / time configuration (UNIFORM)
# -------------------------
GLOBAL_T_COMMON = np.linspace(0.0, 30.0, 400)  # minutes for webpage plots
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 120
PLOT_LINEWIDTH = 1.8
PLOT_GRID_STYLE = {'linestyle': '--', 'alpha': 0.6}
PLOT_TITLE_FS = 14
PLOT_LABEL_FS = 12
PLOT_LEGEND_FS = 'small'


def apply_uniform_style(ax):
    ax.grid(**PLOT_GRID_STYLE)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_LABEL_FS)


# -------------------------
# Load CSV helper (supports env var, upload, and defaults)
# -------------------------
df_agents = None
df_agents_colmap = {}
df_agents_canonmap = {}


def _norm_colname(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())


def _detect_agent_columns(df: pd.DataFrame) -> dict:
    """Detect likely columns for key properties by fuzzy header matching.
    Returns mapping from canonical key -> actual column name.
    Canonical keys: name, mw, logp, sw, cas, smiles, formula, pvap
    """
    norm_map = {_norm_colname(c): c for c in df.columns}

    def find(keys: list[str]) -> str | None:
        # exact normalized match first
        for k in keys:
            kn = _norm_colname(k)
            if kn in norm_map:
                return norm_map[kn]
        # contains/alias
        for knorm, actual in norm_map.items():
            for k in keys:
                if _norm_colname(k) in knorm:
                    return actual
        return None

    syn = {
        'name': ['name', 'chemical', 'compound', 'agent', 'chemicalname', 'agentname'],
        'mw': ['mw', 'molecularweight', 'molwt', 'molecularmass', 'formulaweight', 'molecular_weight'],
        'logp': ['logp', 'xlogp', 'xlogp3', 'logkow', 'log_kow', 'logpoctanolwater'],
        'sw': ['sw', 'solubility', 'aqueoussolubility', 'watersolubility', 'solubilitymgml', 'solubilitymgl'],
        'cas': ['cas', 'casrn', 'casno', 'casnumber'],
        'smiles': ['smiles', 'isomericsmiles', 'canonicalsmiles'],
        'formula': ['formula', 'molecularformula', 'mf', 'molecular_formula'],
        'pvap': ['vaporpressure', 'vapourpressure', 'vp', 'vapor_pressure', 'pvap']
    }

    out = {}
    for key, keys in syn.items():
        col = find(keys)
        if col:
            out[key] = col
    return out


def load_agents_csv(custom_path: str | None = None):
    """Try to load the agents CSV.
    Search order:
      1) custom_path (explicit)
      2) env var DECON_CSV_PATH or AGENTS_CSV_PATH
      3) app directory CWA_with_SMILES_using_CAS2.csv
      4) current working directory CWA_with_SMILES_using_CAS2.csv
    Updates globals df_agents and df_agents_colmap.
    Returns: (loaded: bool, path_used: str | None)
    """
    global df_agents, df_agents_colmap

    candidates = []
    if custom_path:
        candidates.append(custom_path)
    env_path = os.getenv('DECON_CSV_PATH') or os.getenv('AGENTS_CSV_PATH')
    if env_path:
        candidates.append(env_path)
    default_name = "CWA_with_SMILES_using_CAS2.csv"
    app_dir = os.path.dirname(__file__)
    candidates.append(os.path.join(app_dir, default_name))
    candidates.append(os.path.abspath(default_name))

    for path in candidates:
        try:
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                df_map = {col.lower(): col for col in df.columns}
                df_agents = df
                df_agents_colmap = df_map
                # Build canonical mapping for flexible lookup
                globals()['df_agents_canonmap'] = _detect_agent_columns(df)
                app.logger.info("Loaded CSV from %s with %d rows", path, len(df))
                return True, path
        except Exception as e:
            app.logger.warning("Failed to load CSV from %s: %s", path, e)

    df_agents = None
    df_agents_colmap = {}
    globals()['df_agents_canonmap'] = {}
    app.logger.info("CSV not found; will rely on PubChem fallback or defaults")
    return False, None


# Attempt initial load at startup
_loaded, _csv_used = load_agents_csv()


# -------------------------
# PubChem helper (best-effort)
# -------------------------
_pubchem_cache = {}


def pubchem_lookup(name):
    key = name.strip().lower()
    if key in _pubchem_cache:
        return _pubchem_cache[key]
    try:
        compounds = pcp.get_compounds(name, 'name')
        c = compounds[0] if compounds else None
    except Exception as e:
        app.logger.warning("PubChem lookup failed for %s: %s", name, e)
        c = None
    _pubchem_cache[key] = c
    return c


def safe_pubchem_prop(c, prop_name):
    """Safely extract a property (by name) from a PubChem compound object."""
    try:
        for p in getattr(c, 'properties', []) or []:
            if p.get('Name') == prop_name:
                return p.get('Value', {}).get('Fvalue')
    except Exception:
        pass
    return None


# -------------------------
# Agent data fetcher (MW, logP, Sw, CAS, SMILES, formula)
# -------------------------
def safe_get_from_row(row, key_variants):
    # Legacy: map explicit lower-case variants to original columns
    for key in key_variants:
        key_l = key.lower()
        if key_l in df_agents_colmap:
            try:
                return row[df_agents_colmap[key_l]]
            except Exception:
                continue
    return None


def get_canonical_value(row, canon_key):
    """Fetch a value from a row using detected canonical column mapping."""
    try:
        col = df_agents_canonmap.get(canon_key)
        if col is None:
            return None
        return row[col]
    except Exception:
        return None


def get_agent_data(agent_name):
    props = {'MW': None, 'logP': None, 'Sw': None, 'CAS': None, 'SMILES': None, 'formula': None}

    if df_agents is not None:
        try:
            name_col = df_agents_canonmap.get('name', df_agents.columns[0])
            mask = df_agents[name_col].astype(str).str.strip().str.lower() == agent_name.strip().lower()
            row = df_agents[mask]
            if not row.empty:
                row0 = row.iloc[0]
                # Prefer canonical detection; fall back to legacy variant lookup
                mw = get_canonical_value(row0, 'mw')
                if mw is None:
                    mw = safe_get_from_row(row0, ['MW', 'mw', 'MolecularWeight', 'molecular_weight'])
                logp = get_canonical_value(row0, 'logp')
                if logp is None:
                    logp = safe_get_from_row(row0, ['LogP', 'logP', 'xlogp'])
                sw = get_canonical_value(row0, 'sw')
                if sw is None:
                    sw = safe_get_from_row(row0, ['Sw', 'sw', 'solubility', 'Solubility'])
                cas = get_canonical_value(row0, 'cas')
                if cas is None:
                    cas = safe_get_from_row(row0, ['CAS', 'cas'])
                smiles = get_canonical_value(row0, 'smiles')
                if smiles is None:
                    smiles = safe_get_from_row(row0, ['SMILES', 'smiles'])
                formula = get_canonical_value(row0, 'formula')
                if formula is None:
                    formula = safe_get_from_row(row0, ['formula', 'Formula', 'MolecularFormula'])
                pvap = get_canonical_value(row0, 'pvap')

                try:
                    if mw is not None and str(mw).strip().lower() not in ('none', 'nan', ''):
                        props['MW'] = float(mw)
                except:
                    pass
                try:
                    if logp is not None and str(logp).strip().lower() not in ('none', 'nan', ''):
                        props['logP'] = float(logp)
                except:
                    pass
                try:
                    if sw is not None and str(sw).strip().lower() not in ('none', 'nan', ''):
                        props['Sw'] = float(sw)
                except:
                    pass
                try:
                    if pvap is not None and str(pvap).strip().lower() not in ('none', 'nan', ''):
                        props['Pvap'] = float(pvap)
                except:
                    pass
                if cas is not None:
                    props['CAS'] = str(cas)
                if smiles is not None:
                    props['SMILES'] = str(smiles)
                if formula is not None:
                    props['formula'] = str(formula)
        except Exception as e:
            app.logger.debug("CSV lookup exception: %s", e)

    # PubChem fallback for missing values
    need = any(props[k] is None for k in ('MW', 'logP', 'SMILES', 'formula'))
    if need:
        c = pubchem_lookup(agent_name)
        if c:
            try:
                if props['MW'] is None and getattr(c, 'molecular_weight', None) not in (None, ''):
                    props['MW'] = float(c.molecular_weight)
            except:
                pass
            try:
                if props['logP'] is None and getattr(c, 'xlogp', None) not in (None, 'None'):
                    props['logP'] = float(c.xlogp)
            except:
                pass
            if props['SMILES'] is None:
                smiles_val = getattr(c, 'smiles', None) or getattr(c, 'isomeric_smiles', None)
                if smiles_val:
                    props['SMILES'] = str(smiles_val)
            if props['formula'] is None and getattr(c, 'molecular_formula', None):
                props['formula'] = c.molecular_formula

    # Try to fetch aqueous solubility if not present
    if props['Sw'] is None:
        try:
            c2 = pcp.get_properties('AqueousSolubility', agent_name, 'name')
            if c2 and isinstance(c2, list) and len(c2) > 0:
                sol = c2[0].get('AqueousSolubility', None)
                if sol is not None:
                    try:
                        sfloat = float(sol)
                        props['Sw'] = sfloat / 1000.0 if sfloat > 100 else sfloat
                    except:
                        pass
        except Exception:
            pass

    return props


# -------------------------
# Helper: compute Msat and related values
# -------------------------
def compute_Msat_from_logP_sw(logP, Sw, fdep=0.1, hsc=13.4e-4):
    Kow = 10.0 ** logP
    Kscw = 0.040 * Kow**0.81 + 4.06 * Kow**0.27 + 0.359
    Csat = Kscw * Sw
    Msat = fdep * hsc * Csat
    return Msat, Kscw, Csat

KBDO = 1.25  # retained for compatibility with existing parameter calculations

agent_properties = [
    ("Sarin (GB)", 140.09, 1.1, 0.30, 9.0),
    ("Cyclosarin (GF)", 180.17, 1.3, 1.67, 4.5),
    ("VX", 267.37, 1.008, 0.675, 1.15),
    ("VR", 267.37, 1.1, 0.32, 1.0),
    ("Soman (GD)", 182.18, 1.022, 1.78, 0.75),
    ("Tabun (GA)", 162.12, 1.08, 0.38, 0.6),
    ("HD (Mustard)", 159.08, 1.27, 2.14, 0.4),
    ("Lewisite (L)", 207.35, 1.89, 2.56, 0.3),
    ("T-2 Toxin", 466.6, 1.15, 2.27, 0.15),
]

agentAbsorptionRates = {
    "Sarin (GB)": 3.0e-6,
    "Tabun (GA)": 2.0e-6,
    "Soman (GD)": 4.0e-6,
    "Cyclosarin (GF)": 2.5e-6,
    "VX": 150e-6,
    "VR": 120e-6,
    "HD (Mustard)": 50e-6,
    "Lewisite (L)": 30e-6,
    "T-2 Toxin": 10e-6,
}


def get_ka(agent):
    return agentAbsorptionRates.get(agent, 1e-6)


def calculate_values(name, mw, rho, logP, kObs):
    Kow = 10 ** logP
    BF = 1.37 * 4.2 * Kow ** 0.31
    phiaq = 0.6
    Kscw = (1 - 0.613) * BF + phiaq
    logPscw = -2.8 + (0.66 * logP) - (0.0056 * mw)
    Pscw = 10 ** logPscw
    Dsc = (Pscw * L * 100) / Kscw
    Dskin = Dsc / 3600.0
    DskinM = Dskin * 1e-4
    kp = (DskinM * Kow) / L
    kr = kObs / KBDO
    return Dskin, Kow, kp, kr


############################################################
# Dermal absorption model (Original ODE-based implementation)
############################################################

def solve_dermal_absorption_original(MW, logKow, Pvap, Sw, Mo=1e-3, tf_hours=25.0):
    """
    Solve the dermal absorption model from the original script for a single agent.
    Returns t_hours, Qt (absorbed), Qet (evaporated).
    """
    # Structural parameters (can be extended to infer from SMILES; using mild defaults)
    nc = 3
    nh = 8
    no = 1
    nn = 0
    nring = 0

    # System parameters
    hsc = 13.4e-4  # cm
    h1 = hsc
    fdep = 0.1
    tf = float(tf_hours) * 3600.0

    # Environment for evaporation
    u = 16.5  # cm/s
    L_air = 13.4  # cm
    R = 62.37  # mL·Torr/K·mmol
    T = 298.15

    # Calculated transport properties
    Kow = 10.0 ** logKow
    Kscw = 0.040 * Kow ** 0.81 + 4.06 * Kow ** 0.27 + 0.359
    Csat = Kscw * Sw
    logPscw = -2.8 + 0.66 * logKow - 0.0056 * MW
    Pscw = 10 ** logPscw
    kp = Pscw
    D1 = (Pscw * h1 / Kscw) / 3600.0

    # Gas phase transport
    Vp = Pvap  # torr
    S = 16.5 * nc + 1.98 * nh + 5.69 * nn + 5.48 * no - 20.42 * nring
    Dg = (10 ** (-3) * T ** 1.75 * (1 / 29 + 1 / MW) ** 0.5) / (S ** (1 / 3) + (20.1) ** (1 / 3)) ** 2
    kg = (3260 / 3600) * Dg ** (2 / 3) * np.sqrt(u / L_air)
    K = (kg * Vp * MW) / (R * T) * 1.0 / (kp * Sw)
    chi = K

    Msat = fdep * hsc * Csat
    Msurfo = Mo - Msat
    kevaprho = chi * D1 * Csat / hsc

    # Decide regime
    use_above = bool(Mo > Msat)

    if use_above:
        # Above saturation ODEs (phase 1) - derived system
        def differential_system(t, y):
            Ts2, Ts3, Ts4, Ts5, Ts6, Ts7, Ts8, Ts9, Ts10, Ts11, Qt, Qst = y
            denom = (h1 - fdep * hsc) ** 2
            # Precomputed coefficients as per original derivation
            dTs2_dt = (1 / denom) * D1 * (0.0 + 3699.34 * Csat - 4857.68 * Ts2 + 1398.55 * Ts3 -
                                          339.574 * Ts4 + 155.38 * Ts5 - 94.1732 * Ts6 + 68.0747 * Ts7 -
                                          56.3546 * Ts8 + 52.8474 * Ts9 - 57.2106 * Ts10 + 79.6158 * Ts11)
            dTs3_dt = (1 / denom) * D1 * (0.0 - 216.102 * Csat + 623.883 * Ts2 - 666.927 * Ts3 +
                                          314.69 * Ts4 - 79.6341 * Ts5 + 38.4789 * Ts6 - 24.8625 * Ts7 +
                                          19.3336 * Ts8 - 17.4769 * Ts9 + 18.5229 * Ts10 - 25.5064 * Ts11)
            dTs4_dt = (1 / denom) * D1 * (0.0 + 51.3576 * Csat - 103.349 * Ts2 + 214.683 * Ts3 -
                                          290.794 * Ts4 + 156.338 * Ts5 - 40.8813 * Ts6 + 20.5419 * Ts7 -
                                          14.0087 * Ts8 + 11.7504 * Ts9 - 11.934 * Ts10 + 16.0906 * Ts11)
            dTs5_dt = (1 / denom) * D1 * (0.0 - 21.3451 * Csat + 38.4943 * Ts2 - 44.2193 * Ts3 +
                                          127.211 * Ts4 - 188.669 * Ts5 + 108.676 * Ts6 - 29.5263 * Ts7 +
                                          15.5706 * Ts8 - 11.4047 * Ts9 + 10.7477 * Ts10 - 13.9713 * Ts11)
            dTs6_dt = (1 / denom) * D1 * (0.0 + 12.362 * Csat - 21.2652 * Ts2 + 19.4771 * Ts3 -
                                          30.3147 * Ts4 + 99.0212 * Ts5 - 155.568 * Ts6 + 94.3278 * Ts7 -
                                          26.904 * Ts8 + 15.2359 * Ts9 - 12.5905 * Ts10 + 15.3766 * Ts11)
            dTs7_dt = (1 / denom) * D1 * (0.0 - 9.15831 * Csat + 15.3765 * Ts2 - 12.5905 * Ts3 +
                                          15.2361 * Ts4 - 26.9043 * Ts5 + 94.3281 * Ts6 - 155.568 * Ts7 +
                                          99.0218 * Ts8 - 30.3155 * Ts9 + 19.4782 * Ts10 - 21.267 * Ts11)
            dTs8_dt = (1 / denom) * D1 * (0.0 + 8.43803 * Csat - 13.9738 * Ts2 + 10.7495 * Ts3 -
                                          11.4063 * Ts4 + 15.5722 * Ts5 - 29.5281 * Ts6 + 108.678 * Ts7 -
                                          188.671 * Ts8 + 127.214 * Ts9 - 44.2228 * Ts10 + 38.4997 * Ts11)
            dTs9_dt = (1 / denom) * D1 * (0.0 - 9.80631 * Csat + 16.1085 * Ts2 - 11.9465 * Ts3 +
                                          11.7612 * Ts4 - 14.0191 * Ts5 + 20.5525 * Ts6 - 40.8927 * Ts7 +
                                          156.351 * Ts8 - 290.809 * Ts9 + 214.702 * Ts10 - 103.378 * Ts11)
            dTs10_dt = (1 / denom) * D1 * (0.0 + 15.6376 * Csat - 25.5668 * Ts2 + 18.5648 * Ts3 -
                                           17.5131 * Ts4 + 19.3681 * Ts5 - 24.8978 * Ts6 + 38.5167 * Ts7 -
                                           79.6766 * Ts8 + 314.74 * Ts9 - 666.991 * Ts10 + 623.98 * Ts11)
            dTs11_dt = (1 / denom) * D1 * (0.0 - 48.899 * Csat + 79.7604 * Ts2 - 57.3114 * Ts3 +
                                           52.935 * Ts4 - 56.4392 * Ts5 + 68.1623 * Ts6 - 94.2691 * Ts7 +
                                           155.491 * Ts8 - 339.708 * Ts9 + 1398.72 * Ts10 - 4857.95 * Ts11)
            dQt_dt = -(1 / (h1 - fdep * hsc)) * D1 * (0.0 - 0.999885 * Csat + 1.63008 * Ts2 -
                                                      1.16854 * Ts3 + 1.07425 * Ts4 - 1.1361 * Ts5 +
                                                      1.35336 * Ts6 - 1.82683 * Ts7 + 2.87424 * Ts8 -
                                                      5.62786 * Ts9 + 16.1529 * Ts10 - 123.326 * Ts11)
            dQst_dt = -kevaprho + (1 / (h1 - fdep * hsc)) * D1 * (0.0 - 110.997 * Csat +
                                                                   123.321 * Ts2 - 16.1502 * Ts3 +
                                                                   5.62579 * Ts4 - 2.87258 * Ts5 +
                                                                   1.82542 * Ts6 - 1.3521 * Ts7 +
                                                                   1.13491 * Ts8 - 1.07303 * Ts9 +
                                                                   1.16715 * Ts10 - 1.6281 * Ts11)
            return [dTs2_dt, dTs3_dt, dTs4_dt, dTs5_dt, dTs6_dt, dTs7_dt, dTs8_dt, dTs9_dt, dTs10_dt, dTs11_dt, dQt_dt, dQst_dt]

        def qst_negative(t, y):
            return y[11]
        qst_negative.terminal = True
        qst_negative.direction = -1

        y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Msurfo]
        t_span = (0, tf)
        t_eval = np.linspace(0, tf, 5000)
        sol = solve_ivp(differential_system, t_span, y0, t_eval=t_eval, events=qst_negative, method='RK45', rtol=1e-7, atol=1e-9)

        t = sol.t
        Qt = sol.y[10]
        Qst = sol.y[11]
        Qet = kevaprho * t

        # Phase 2 if needed
        if sol.t_events[0].size > 0:
            ttrans = sol.t_events[0][0]
            ROOT = np.array([1e-20, 0.0130467, 0.0674683, 0.160295, 0.283302,
                             0.425563, 0.574437, 0.716698, 0.839705, 0.932532,
                             0.986953, 1.0])
            # Interpolation for transition values
            zTop = ROOT * fdep * hsc
            zSkin = ROOT * (1 - fdep) * hsc + fdep * hsc
            zAll = np.concatenate([zTop[1:-1], zSkin[1:-1]])
            vTop = np.full(len(zTop[1:-1]), Csat)
            # Prepare vector from Ts2..Ts11 at ttrans
            Ts_list = [np.interp(ttrans, t, sol.y[i]) for i in range(0, 10)]
            vSkin = np.array(Ts_list[1:-1])  # Ts3..Ts10 (match lengths)
            if vSkin.size != zSkin[1:-1].size:
                vSkin = np.interp(np.linspace(0, 1, zSkin[1:-1].size), np.linspace(0, 1, vSkin.size), vSkin)
            zAll = np.concatenate([zTop[1:-1], zSkin[1:-1]])
            vAll = np.concatenate([vTop, vSkin])
            uniq_idx = np.unique(zAll, return_index=True)[1]
            interpCombined = interp1d(zAll[uniq_idx], vAll[uniq_idx], kind='linear', bounds_error=False, fill_value='extrapolate')

            Vr_init = []
            ROOT_positions = ROOT[1:-1]
            for pos in ROOT_positions:
                Vr_init.append(interpCombined(pos * hsc))
            Vr_init = np.array(Vr_init[:10])
            qtinitp2 = float(np.interp(ttrans, t, Qt))
            qevapinitp2 = float(kevaprho * ttrans)
            y0_p2 = np.concatenate([Vr_init, [qtinitp2, qevapinitp2]])

            def differential_system_phase2(t, y):
                Vr2, Vr3, Vr4, Vr5, Vr6, Vr7, Vr8, Vr9, Vr10, Vr11, Q1t, Qet = y
                common_denom = -(110.997 / hsc) - (1.0 * chi) / hsc
                flux_term = (0.0 - (123.321 * Vr2) / hsc + (16.1502 * Vr3) / hsc -
                             (5.62579 * Vr4) / hsc + (2.87258 * Vr5) / hsc -
                             (1.82542 * Vr6) / hsc + (1.3521 * Vr7) / hsc -
                             (1.13491 * Vr8) / hsc + (1.07303 * Vr9) / hsc -
                             (1.16715 * Vr10) / hsc + (1.6281 * Vr11) / hsc)

                dVr2_dt = (1 / hsc ** 2) * D1 * (0.0 - 4857.68 * Vr2 + 1398.55 * Vr3 - 339.574 * Vr4 + 155.38 * Vr5 - 94.1732 * Vr6 + 68.0747 * Vr7 - 56.3546 * Vr8 + 52.8474 * Vr9 - 57.2106 * Vr10 + 79.6158 * Vr11 + (1 / common_denom) * 3699.34 * flux_term)
                dVr3_dt = (1 / hsc ** 2) * D1 * (0.0 + 623.883 * Vr2 - 666.927 * Vr3 + 314.69 * Vr4 - 79.6341 * Vr5 + 38.4789 * Vr6 - 24.8625 * Vr7 + 19.3336 * Vr8 - 17.4769 * Vr9 + 18.5229 * Vr10 - 25.5064 * Vr11 - (1 / common_denom) * 216.102 * flux_term)
                dVr4_dt = (1 / hsc ** 2) * D1 * (0.0 - 103.349 * Vr2 + 214.683 * Vr3 - 290.794 * Vr4 + 156.338 * Vr5 - 40.8813 * Vr6 + 20.5419 * Vr7 - 14.0087 * Vr8 + 11.7504 * Vr9 - 11.934 * Vr10 + 16.0906 * Vr11 + (1 / common_denom) * 51.3576 * flux_term)
                dVr5_dt = (1 / hsc ** 2) * D1 * (0.0 + 38.4943 * Vr2 - 44.2193 * Vr3 + 127.211 * Vr4 - 188.669 * Vr5 + 108.676 * Vr6 - 29.5263 * Vr7 + 15.5706 * Vr8 - 11.4047 * Vr9 + 10.7477 * Vr10 - 13.9713 * Vr11 - (1 / common_denom) * 21.3451 * flux_term)
                dVr6_dt = (1 / hsc ** 2) * D1 * (0.0 - 21.2652 * Vr2 + 19.4771 * Vr3 - 30.3147 * Vr4 + 99.0212 * Vr5 - 155.568 * Vr6 + 94.3278 * Vr7 - 26.904 * Vr8 + 15.2359 * Vr9 - 12.5905 * Vr10 + 15.3766 * Vr11 + (1 / common_denom) * 12.362 * flux_term)
                dVr7_dt = (1 / hsc ** 2) * D1 * (0.0 + 15.3765 * Vr2 - 12.5905 * Vr3 + 15.2361 * Vr4 - 26.9043 * Vr5 + 94.3281 * Vr6 - 155.568 * Vr7 + 99.0218 * Vr8 - 30.3155 * Vr9 + 19.4782 * Vr10 - 21.267 * Vr11 - (1 / common_denom) * 9.15831 * flux_term)
                dVr8_dt = (1 / hsc ** 2) * D1 * (0.0 - 13.9738 * Vr2 + 10.7495 * Vr3 - 11.4063 * Vr4 + 15.5722 * Vr5 - 29.5281 * Vr6 + 108.678 * Vr7 - 188.671 * Vr8 + 127.214 * Vr9 - 44.2228 * Vr10 + 38.4997 * Vr11 + (1 / common_denom) * 8.43803 * flux_term)
                dVr9_dt = (1 / hsc ** 2) * D1 * (0.0 + 16.1085 * Vr2 - 11.9465 * Vr3 + 11.7612 * Vr4 - 14.0191 * Vr5 + 20.5525 * Vr6 - 40.8927 * Vr7 + 156.351 * Vr8 - 290.809 * Vr9 + 214.702 * Vr10 - 103.378 * Vr11 - (1 / common_denom) * 9.80631 * flux_term)
                dVr10_dt = (1 / hsc ** 2) * D1 * (0.0 - 25.5668 * Vr2 + 18.5648 * Vr3 - 17.5131 * Vr4 + 19.3681 * Vr5 - 24.8978 * Vr6 + 38.5167 * Vr7 - 79.6766 * Vr8 + 314.74 * Vr9 - 666.991 * Vr10 + 623.98 * Vr11 + (1 / common_denom) * 15.6376 * flux_term)
                dVr11_dt = (1 / hsc ** 2) * D1 * (0.0 + 79.7604 * Vr2 - 57.3114 * Vr3 + 52.935 * Vr4 - 56.4392 * Vr5 + 68.1623 * Vr6 - 94.2691 * Vr7 + 155.491 * Vr8 - 339.708 * Vr9 + 1398.72 * Vr10 - 4857.95 * Vr11 - (1 / common_denom) * 48.899 * flux_term)

                dQ1t_dt = (-(1 / hsc) * D1 * (0.0 + 1.63008 * Vr2 - 1.16854 * Vr3 + 1.07425 * Vr4 - 1.1361 * Vr5 + 1.35336 * Vr6 - 1.82683 * Vr7 + 2.87424 * Vr8 - 5.62786 * Vr9 + 16.1529 * Vr10 - 123.326 * Vr11 - (1 / common_denom) * 0.999885 * flux_term))
                dQet_dt = (1 / hsc) * D1 * (0.0 + 123.321 * Vr2 - 16.1502 * Vr3 + 5.62579 * Vr4 - 2.87258 * Vr5 + 1.82542 * Vr6 - 1.3521 * Vr7 + 1.13491 * Vr8 - 1.07303 * Vr9 + 1.16715 * Vr10 - 1.6281 * Vr11 - (1 / common_denom) * 110.997 * flux_term)
                return np.array([dVr2_dt, dVr3_dt, dVr4_dt, dVr5_dt, dVr6_dt, dVr7_dt, dVr8_dt, dVr9_dt, dVr10_dt, dVr11_dt, dQ1t_dt, dQet_dt])

            phase2_duration = tf - ttrans
            if phase2_duration > 1.0:
                t_span_phase2 = (0, phase2_duration)
                t_eval_phase2 = np.linspace(0, phase2_duration, 4000)
                sol2 = solve_ivp(differential_system_phase2, t_span_phase2, y0_p2, t_eval=t_eval_phase2, method='RK45', rtol=1e-7, atol=1e-9)
                if sol2.success and sol2.y.size > 0:
                    t_combined = np.concatenate([t, sol2.t + ttrans])
                    Qt_combined = np.concatenate([Qt, sol2.y[-2]])
                    Qet_combined = np.concatenate([Qet, sol2.y[-1]])
                else:
                    t_combined, Qt_combined, Qet_combined = t, Qt, Qet
            else:
                t_combined, Qt_combined, Qet_combined = t, Qt, Qet
        else:
            t_combined, Qt_combined, Qet_combined = t, Qt, Qet

        return t_combined / 3600.0, Qt_combined, Qet_combined

    else:
        # Below saturation model ODE
        h = h1
        Tso = Mo / (fdep * h)

        def ode_system(t, y):
            Ts = y[0:10]
            Tv = y[10:20]
            Qt = y[20]
            Qet = y[21]
            denom1 = ((111.0 * D1) / (fdep * h) + (110.997 * D1) / (h - 1.0 * fdep * h) + (0.998546 * D1) / (fdep ** 2 * h ** 2 * (-(110.997 / (fdep * h)) - (1.0 * chi) / h)))
            chi_term_denom = -(110.997 / (fdep * h)) - (1.0 * chi) / h
            complex_flux = (0.0 - (1.63008 * D1 * Ts[0]) / (fdep * h) - (123.307 * D1 * Ts[0]) / (fdep ** 2 * h ** 2 * chi_term_denom) + (1.16854 * D1 * Ts[1]) / (fdep * h) + (16.1483 * D1 * Ts[1]) / (fdep ** 2 * h ** 2 * chi_term_denom) - (1.07425 * D1 * Ts[2]) / (fdep * h) - (5.62515 * D1 * Ts[2]) / (fdep ** 2 * h ** 2 * chi_term_denom) + (1.1361 * D1 * Ts[3]) / (fdep * h) + (2.87225 * D1 * Ts[3]) / (fdep ** 2 * h ** 2 * chi_term_denom) - (1.35336 * D1 * Ts[4]) / (fdep * h) - (1.82521 * D1 * Ts[4]) / (fdep ** 2 * h ** 2 * chi_term_denom) + (1.82683 * D1 * Ts[5]) / (fdep * h) + (1.35194 * D1 * Ts[5]) / (fdep ** 2 * h ** 2 * chi_term_denom) - (2.87424 * D1 * Ts[6]) / (fdep * h) - (1.13478 * D1 * Ts[6]) / (fdep ** 2 * h ** 2 * chi_term_denom) + (5.62786 * D1 * Ts[7]) / (fdep * h) + (1.0729 * D1 * Ts[7]) / (fdep ** 2 * h ** 2 * chi_term_denom) - (16.1529 * D1 * Ts[8]) / (fdep * h) - (1.16702 * D1 * Ts[8]) / (fdep ** 2 * h ** 2 * chi_term_denom) + (123.326 * D1 * Ts[9]) / (fdep * h) + (1.62791 * D1 * Ts[9]) / (fdep ** 2 * h ** 2 * chi_term_denom) + (123.321 * D1 * Tv[0]) / (h - 1.0 * fdep * h) - (16.1502 * D1 * Tv[1]) / (h - 1.0 * fdep * h) + (5.62579 * D1 * Tv[2]) / (h - 1.0 * fdep * h) - (2.87258 * D1 * Tv[3]) / (h - 1.0 * fdep * h) + (1.82542 * D1 * Tv[4]) / (h - 1.0 * fdep * h) - (1.3521 * D1 * Tv[5]) / (h - 1.0 * fdep * h) + (1.13491 * D1 * Tv[6]) / (h - 1.0 * fdep * h) - (1.07303 * D1 * Tv[7]) / (h - 1.0 * fdep * h) + (1.16715 * D1 * Tv[8]) / (h - 1.0 * fdep * h) - (1.6281 * D1 * Tv[9]) / (h - 1.0 * fdep * h))
            surface_term = (-123.321 * Ts[0] / (fdep * h) + 16.1502 * Ts[1] / (fdep * h) - 5.62579 * Ts[2] / (fdep * h) + 2.87258 * Ts[3] / (fdep * h) - 1.82542 * Ts[4] / (fdep * h) + 1.3521 * Ts[5] / (fdep * h) - 1.13491 * Ts[6] / (fdep * h) + 1.07303 * Ts[7] / (fdep * h) - 1.16715 * Ts[8] / (fdep * h) + 1.6281 * Ts[9] / (fdep * h) - (0.99866 * complex_flux) / (fdep * h * denom1))
            dydt = np.zeros(22)
            # Ts derivatives
            dydt[0] = (1 / (fdep ** 2 * h ** 2) * D1 * (-4857.68 * Ts[0] + 1398.55 * Ts[1] - 339.574 * Ts[2] + 155.38 * Ts[3] - 94.1732 * Ts[4] + 68.0747 * Ts[5] - 56.3546 * Ts[6] + 52.8474 * Ts[7] - 57.2106 * Ts[8] + 79.6158 * Ts[9] - (1 / denom1) * 48.8097 * complex_flux + (1 / chi_term_denom) * 3699.34 * surface_term))
            dydt[1] = (1 / (fdep ** 2 * h ** 2) * D1 * (623.883 * Ts[0] - 666.927 * Ts[1] + 314.69 * Ts[2] - 79.6341 * Ts[3] + 38.4789 * Ts[4] - 24.8625 * Ts[5] + 19.3336 * Ts[6] - 17.4769 * Ts[7] + 18.5229 * Ts[8] - 25.5064 * Ts[9] + (15.6003 * complex_flux) / denom1 - (216.102 / chi_term_denom) * surface_term))
            dydt[2] = (1 / (fdep ** 2 * h ** 2) * D1 * (-103.349 * Ts[0] + 214.683 * Ts[1] - 290.794 * Ts[2] + 156.338 * Ts[3] - 40.8813 * Ts[4] + 20.5419 * Ts[5] - 14.0087 * Ts[6] + 11.7504 * Ts[7] - 11.934 * Ts[8] + 16.0906 * Ts[9] - (9.79523 * complex_flux) / denom1 + (51.3576 / chi_term_denom) * surface_term))
            dydt[3] = (1 / (fdep ** 2 * h ** 2) * D1 * (38.4943 * Ts[0] - 44.2193 * Ts[1] + 127.211 * Ts[2] - 188.669 * Ts[3] + 108.676 * Ts[4] - 29.5263 * Ts[5] + 15.5706 * Ts[6] - 11.4047 * Ts[7] + 10.7477 * Ts[8] - 13.9713 * Ts[9] + (8.43649 * complex_flux) / denom1 - (21.3451 / chi_term_denom) * surface_term))
            dydt[4] = (1 / (fdep ** 2 * h ** 2) * D1 * (-21.2652 * Ts[0] + 19.4771 * Ts[1] - 30.3147 * Ts[2] + 99.0212 * Ts[3] - 155.568 * Ts[4] + 94.3278 * Ts[5] - 26.904 * Ts[6] + 15.2359 * Ts[7] - 12.5905 * Ts[8] + 15.3766 * Ts[9] - (9.15837 * complex_flux) / denom1 + (12.362 / chi_term_denom) * surface_term))
            dydt[5] = (1 / (fdep ** 2 * h ** 2) * D1 * (15.3765 * Ts[0] - 12.5905 * Ts[1] + 15.2361 * Ts[2] - 26.9043 * Ts[3] + 94.3281 * Ts[4] - 155.568 * Ts[5] + 99.0218 * Ts[6] - 30.3155 * Ts[7] + 19.4782 * Ts[8] - 21.267 * Ts[9] + (12.363 * complex_flux) / denom1 - (9.15831 / chi_term_denom) * surface_term))
            dydt[6] = (1 / (fdep ** 2 * h ** 2) * D1 * (-13.9738 * Ts[0] + 10.7495 * Ts[1] - 11.4063 * Ts[2] + 15.5722 * Ts[3] - 29.5281 * Ts[4] + 108.678 * Ts[5] - 188.671 * Ts[6] + 127.214 * Ts[7] - 44.2228 * Ts[8] + 38.4997 * Ts[9] - (21.3485 * complex_flux) / denom1 + (8.43803 / chi_term_denom) * surface_term))
            dydt[7] = (1 / (fdep ** 2 * h ** 2) * D1 * (16.1085 * Ts[0] - 11.9465 * Ts[1] + 11.7612 * Ts[2] - 14.0191 * Ts[3] + 20.5525 * Ts[4] - 40.8927 * Ts[5] + 156.351 * Ts[6] - 290.809 * Ts[7] + 214.702 * Ts[8] - 103.378 * Ts[9] + (51.3756 * complex_flux) / denom1 - (9.80631 / chi_term_denom) * surface_term))
            dydt[8] = (1 / (fdep ** 2 * h ** 2) * D1 * (-25.5668 * Ts[0] + 18.5648 * Ts[1] - 17.5131 * Ts[2] + 19.3681 * Ts[3] - 24.8978 * Ts[4] + 38.5167 * Ts[5] - 79.6766 * Ts[6] + 314.74 * Ts[7] - 666.991 * Ts[8] + 623.98 * Ts[9] - (216.163 * complex_flux) / denom1 + (15.6376 / chi_term_denom) * surface_term))
            dydt[9] = (1 / (fdep ** 2 * h ** 2) * D1 * (79.7604 * Ts[0] - 57.3114 * Ts[1] + 52.935 * Ts[2] - 56.4392 * Ts[3] + 68.1623 * Ts[4] - 94.2691 * Ts[5] + 155.491 * Ts[6] - 339.708 * Ts[7] + 1398.72 * Ts[8] - 4857.95 * Ts[9] + (3699.51 * complex_flux) / denom1 - (48.899 / chi_term_denom) * surface_term))
            # Tv derivatives
            dydt[10] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 - 4857.68 * Tv[0] + 1398.55 * Tv[1] - 339.574 * Tv[2] + 155.38 * Tv[3] - 94.1732 * Tv[4] + 68.0747 * Tv[5] - 56.3546 * Tv[6] + 52.8474 * Tv[7] - 57.2106 * Tv[8] + 79.6158 * Tv[9] + (3699.34 * complex_flux) / denom1))
            dydt[11] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 + 623.883 * Tv[0] - 666.927 * Tv[1] + 314.69 * Tv[2] - 79.6341 * Tv[3] + 38.4789 * Tv[4] - 24.8625 * Tv[5] + 19.3336 * Tv[6] - 17.4769 * Tv[7] + 18.5229 * Tv[8] - 25.5064 * Tv[9] - (216.102 * complex_flux) / denom1))
            dydt[12] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 - 103.349 * Tv[0] + 214.683 * Tv[1] - 290.794 * Tv[2] + 156.338 * Tv[3] - 40.8813 * Tv[4] + 20.5419 * Tv[5] - 14.0087 * Tv[6] + 11.7504 * Tv[7] - 11.934 * Tv[8] + 16.0906 * Tv[9] + (51.3576 * complex_flux) / denom1))
            dydt[13] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 + 38.4943 * Tv[0] - 44.2193 * Tv[1] + 127.211 * Tv[2] - 188.669 * Tv[3] + 108.676 * Tv[4] - 29.5263 * Tv[5] + 15.5706 * Tv[6] - 11.4047 * Tv[7] + 10.7477 * Tv[8] - 13.9713 * Tv[9] - (21.3451 * complex_flux) / denom1))
            dydt[14] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 - 21.2652 * Tv[0] + 19.4771 * Tv[1] - 30.3147 * Tv[2] + 99.0212 * Tv[3] - 155.568 * Tv[4] + 94.3278 * Tv[5] - 26.904 * Tv[6] + 15.2359 * Tv[7] - 12.5905 * Tv[8] + 15.3766 * Tv[9] + (12.362 * complex_flux) / denom1))
            dydt[15] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 + 15.3765 * Tv[0] - 12.5905 * Tv[1] + 15.2361 * Tv[2] - 26.9043 * Tv[3] + 94.3281 * Tv[4] - 155.568 * Tv[5] + 99.0218 * Tv[6] - 30.3155 * Tv[7] + 19.4782 * Tv[8] - 21.267 * Tv[9] - (9.15831 * complex_flux) / denom1))
            dydt[16] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 - 13.9738 * Tv[0] + 10.7495 * Tv[1] - 11.4063 * Tv[2] + 15.5722 * Tv[3] - 29.5281 * Tv[4] + 108.678 * Tv[5] - 188.671 * Tv[6] + 127.214 * Tv[7] - 44.2228 * Tv[8] + 38.4997 * Tv[9] + (8.43803 * complex_flux) / denom1))
            dydt[17] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 + 16.1085 * Tv[0] - 11.9465 * Tv[1] + 11.7612 * Tv[2] - 14.0191 * Tv[3] + 20.5525 * Tv[4] - 40.8927 * Tv[5] + 156.351 * Tv[6] - 290.809 * Tv[7] + 214.702 * Tv[8] - 103.378 * Tv[9] - (9.80631 * complex_flux) / denom1))
            dydt[18] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 - 25.5668 * Tv[0] + 18.5648 * Tv[1] - 17.5131 * Tv[2] + 19.3681 * Tv[3] - 24.8978 * Tv[4] + 38.5167 * Tv[5] - 79.6766 * Tv[6] + 314.74 * Tv[7] - 666.991 * Tv[8] + 623.98 * Tv[9] + (15.6376 * complex_flux) / denom1))
            dydt[19] = (1 / (h - fdep * h) ** 2 * D1 * (0.0 + 79.7604 * Tv[0] - 57.3114 * Tv[1] + 52.935 * Tv[2] - 56.4392 * Tv[3] + 68.1623 * Tv[4] - 94.2691 * Tv[5] + 155.491 * Tv[6] - 339.708 * Tv[7] + 1398.72 * Tv[8] - 4857.95 * Tv[9] - (48.899 * complex_flux) / denom1))
            dydt[20] = (-(1 / (h - fdep * h)) * D1 * (0.0 + 1.63008 * Tv[0] - 1.16854 * Tv[1] + 1.07425 * Tv[2] - 1.1361 * Tv[3] + 1.35336 * Tv[4] - 1.82683 * Tv[5] + 2.87424 * Tv[6] - 5.62786 * Tv[7] + 16.1529 * Tv[8] - 123.326 * Tv[9] - (0.999885 * complex_flux) / denom1))
            dydt[21] = ((1 / (fdep * h)) * D1 * (123.321 * Ts[0] - 16.1502 * Ts[1] + 5.62579 * Ts[2] - 2.87258 * Ts[3] + 1.82542 * Ts[4] - 1.3521 * Ts[5] + 1.13491 * Ts[6] - 1.07303 * Ts[7] + 1.16715 * Ts[8] - 1.6281 * Ts[9] + (0.99866 * complex_flux) / denom1 - (110.997 / chi_term_denom) * surface_term))
            return dydt

        y0 = np.zeros(22)
        y0[0:10] = Tso
        y0[10:20] = 0.0
        y0[20] = 0.0
        y0[21] = 0.0
        t_span = (0, tf)
        t_eval = np.linspace(0, tf, 2000)
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-7, atol=1e-9)
        if not sol.success:
            # Fallback: simple exponentials
            t = t_eval
            tau_abs = max(0.05 * tf, 1.0)
            tau_evap = max(0.02 * tf, 1.0)
            Qt = Mo * 0.3 * (1 - np.exp(-t / tau_abs))
            Qet = Mo * 0.6 * (1 - np.exp(-t / tau_evap))
        else:
            Qt = sol.y[20]
            Qet = sol.y[21]
            t = sol.t
        return t / 3600.0, Qt, Qet


def combined_plot_dermal_absorption_original(agents, Mo=1e-3, tf_hours=25.0):
    """Plot all selected agents with clean styling:
    - Each agent gets one color; styles indicate quantity: Qabs (-), Qevap (--), Qtotal (:)
    - Legend shows agent names once plus a compact style key for the three quantities
    - Right axis shows percentage of initial dose linked to the left axis scale
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    cmap = plt.get_cmap('tab10')

    if not agents:
        agents = agent_properties[:1]

    ax.grid(linestyle='--', alpha=0.35)

    # Plot all agents
    for i, agent in enumerate(agents):
        try:
            name = str(agent[0])
            MW = float(agent[1])
            logP = float(agent[3])

            props = get_agent_data(name)
            Sw = float(props.get('Sw', 1.0) or 1.0)
            # Try CSV first, then PubChem, then default
            Pvap = props.get('Pvap')
            if Pvap is None:
                Pvap = 21.0
                pc = pubchem_lookup(name)
                if pc:
                    try:
                        vap_prop = safe_pubchem_prop(pc, 'Vapor Pressure')
                        if vap_prop:
                            Pvap = float(vap_prop)
                    except Exception:
                        pass
            else:
                Pvap = float(Pvap)

            t_hr, Qt, Qet = solve_dermal_absorption_original(MW, logP, Pvap, Sw, Mo=Mo, tf_hours=tf_hours)
            Qtot = Qt + Qet

            color = cmap(i % cmap.N)
            ax.plot(t_hr, Qt, color=color, linestyle='-', linewidth=2.0, label=f"{name} — Qabs")
            ax.plot(t_hr, Qet, color=color, linestyle='--', linewidth=2.0, label=f"{name} — Qevap")
            ax.plot(t_hr, Qtot, color=color, linestyle=':', linewidth=2.2, label=f"{name} — Qtotal")
        except Exception as e:
            app.logger.exception("Original dermal solver failed for %r: %s", agent, e)
            continue

    # Axes labels
    ax.set_xlabel("Time (hours)", fontsize=PLOT_LABEL_FS)
    ax.set_ylabel("Mass (mg/cm²)", fontsize=PLOT_LABEL_FS)

    # Right axis as percentage of initial dose using a linked scale (no extra plotted line)
    try:
        def m_to_pct(m):
            return 100.0 * m / max(Mo, 1e-20)

        def pct_to_m(p):
            return (p / 100.0) * max(Mo, 1e-20)

        secax = ax.secondary_yaxis('right', functions=(m_to_pct, pct_to_m))
        secax.set_ylabel('Percentage of initial dose (%)', fontsize=PLOT_LABEL_FS)
        secax.set_ylim(0, 100)
        apply_uniform_style(secax)
    except Exception:
        ax2 = ax.twinx()
        ax2.set_ylabel('Percentage of initial dose (%)', fontsize=PLOT_LABEL_FS)
        ax2.set_ylim(0, 100)
        apply_uniform_style(ax2)

    # Compact, two-part legend: agents + style key
    from matplotlib.lines import Line2D
    handles_agents = []
    labels_agents = []
    for i, agent in enumerate(agents):
        color = cmap(i % cmap.N)
        handles_agents.append(Line2D([0], [0], color=color, lw=2.5))
        labels_agents.append(str(agent[0]))

    style_handles = [
        Line2D([0], [0], color='gray', lw=2.0, linestyle='-'),
        Line2D([0], [0], color='gray', lw=2.0, linestyle='--'),
        Line2D([0], [0], color='gray', lw=2.0, linestyle=':')
    ]
    style_labels = ['Qabs', 'Qevap', 'Qtotal']

    ax.legend(handles_agents + style_handles, labels_agents + style_labels,
              fontsize=PLOT_LEGEND_FS, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.0))

    # Limits and title
    ax.set_xlim(left=0.0)
    # Choose y max from all plotted lines on left axis
    try:
        ymax = 0.0
        for line in ax.get_lines():
            if line.get_ydata().size:
                ymax = max(ymax, np.nanmax(line.get_ydata()))
        ymax = max(1e-6, ymax * 1.05)
    except Exception:
        ymax = None
    if ymax:
        ax.set_ylim(bottom=0.0, top=ymax)

    ax.set_title("Dermal absorption model", fontsize=PLOT_TITLE_FS + 2)
    plt.tight_layout(rect=[0, 0, 0.78, 1.0])
    return fig_to_base64(fig)


# -------------------------
# Chebyshev plotting helpers (base64)
# -------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=PLOT_DPI)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img


# -------------------------
# Fallback simulate_dermal_absorption_agent
# -------------------------
def simulate_dermal_absorption_agent_fallback(MW, logP, Pvap, Sw, Mo=1e-3, tf_hours=0.5):
    t_hr = np.linspace(0.0, float(tf_hours), max(50, int(40 * tf_hours + 1)))
    abs_scale = max(1e-6, 0.02 * (10.0 ** (0.5 * logP)) / (1.0 + MW / 300.0))
    evap_scale = max(1e-6, 0.01 * (10.0 ** (-0.2 * logP)) * (Pvap / 20.0 if Pvap > 0 else 1.0))
    asymp_abs_frac = np.clip(0.15 + 0.1 * np.tanh(logP), 0.01, 0.9)
    asymp_evap_frac = np.clip(0.5 - 0.2 * np.tanh(logP), 0.01, 0.99)
    Qt_asymp = Mo * asymp_abs_frac
    Qet_asymp = Mo * asymp_evap_frac
    tau_abs = max(0.01, 0.2 / (abs_scale * 1000.0))
    tau_evap = max(0.005, 0.05 / (evap_scale * 1000.0))
    Qt = Qt_asymp * (1.0 - np.exp(-t_hr / tau_abs))
    Qet = Qet_asymp * (1.0 - np.exp(-t_hr / tau_evap))
    Qt[0] = 0.0
    Qet[0] = 0.0
    return t_hr, Qt, Qet


# -------------------------
# Safe wrapper for simulate_dermal_absorption_agent
# -------------------------
def safe_simulate_dermal_absorption_agent(MW, logP, Pvap, Sw, Mo=1e-3, tf_hours=0.5):
    try:
        fn = globals().get('simulate_dermal_absorption_agent', None)
        if fn is None:
            raise NameError("simulate_dermal_absorption_agent not found")
        t, Qt, Qet = fn(MW, logP, Pvap, Sw, Mo=Mo, tf_hours=tf_hours)
        t = np.asarray(t, dtype=float)
        Qt = np.asarray(Qt, dtype=float)
        Qet = np.asarray(Qet, dtype=float)
        if np.max(t) > 2.0 and np.max(t) <= 60.0 * 24.0 and np.mean(np.diff(t)) > 0.01:
            if np.max(t) > 24.0:
                app.logger.info("simulate_dermal_absorption_agent returned times in minutes; converting to hours.")
                t = t / 60.0
        uniq_idx = np.unique(t, return_index=True)[1]
        t = t[uniq_idx]
        Qt = Qt[uniq_idx]
        Qet = Qet[uniq_idx]
        if t.size > 0 and t[0] > 0.0:
            t = np.insert(t, 0, 0.0)
            Qt = np.insert(Qt, 0, 0.0)
            Qet = np.insert(Qet, 0, 0.0)
        if Qt.size == 0 or Qet.size == 0:
            raise ValueError("simulate_dermal_absorption_agent returned empty arrays")
        return t, Qt, Qet
    except Exception as e:
        app.logger.info("Using fallback simulate_dermal_absorption_agent due to: %s", e)
        return simulate_dermal_absorption_agent_fallback(MW, logP, Pvap, Sw, Mo=Mo, tf_hours=tf_hours)


# -------------------------
# Combined plotting functions
# -------------------------
# Removed chebyshev-based plots and penetration plots per request


def combined_plot_abs_evap_all_agents(agents, Mo=1e-3):
    t_common = GLOBAL_T_COMMON  # minutes
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    cmap = plt.get_cmap('tab10')
    any_plotted = False

    for i, agent in enumerate(agents):
        try:
            name = str(agent[0])
            MW = float(agent[1])
            rho = float(agent[2]) if len(agent) > 2 else 1.0
            logP = float(agent[3]) if len(agent) > 3 else 0.0

            props = get_agent_data(name)
            Sw = float(props.get('Sw', 1.0) or 1.0)
            # Try CSV first, then PubChem, then default
            Pvap = props.get('Pvap')
            if Pvap is None:
                Pvap = 21.0
                pubchem_compound = pubchem_lookup(name)
                if pubchem_compound:
                    try:
                        vap_prop = safe_pubchem_prop(pubchem_compound, 'Vapor Pressure')
                        if vap_prop:
                            Pvap = float(vap_prop)
                    except:
                        pass
            else:
                Pvap = float(Pvap)

            tf_hours = float(GLOBAL_T_COMMON[-1]) / 60.0
            tmin, Qt, Qet = safe_simulate_dermal_absorption_agent(MW, logP, Pvap, Sw, Mo=Mo, tf_hours=tf_hours)

            tmin = np.asarray(tmin, dtype=float)
            Qt = np.asarray(Qt, dtype=float)
            Qet = np.asarray(Qet, dtype=float)
            if np.max(tmin) <= tf_hours + 1e-12:
                tmin_minutes = tmin * 60.0
            else:
                tmin_minutes = tmin

            uniq_idx = np.unique(tmin_minutes, return_index=True)[1]
            tmin_u = tmin_minutes[uniq_idx]
            Qt_u = Qt[uniq_idx]
            Qet_u = Qet[uniq_idx]

            Qt_interp = np.interp(t_common, tmin_u, Qt_u, left=0.0, right=Qt_u[-1])
            Qet_interp = np.interp(t_common, tmin_u, Qet_u, left=0.0, right=Qet_u[-1])
            Qtot_interp = Qt_interp + Qet_interp

            if Qt_interp.size > 0:
                Qt_interp[0] = 0.0
            if Qet_interp.size > 0:
                Qet_interp[0] = 0.0
            if Qtot_interp.size > 0:
                Qtot_interp[0] = 0.0

            color = cmap(i % cmap.N)

            ax.plot(t_common, Qt_interp, '-', linewidth=PLOT_LINEWIDTH, label=f"{name} Qabs (mg/cm²)", color=color)
            ax.plot(t_common, Qet_interp, '--', linewidth=PLOT_LINEWIDTH, label=f"{name} Qevap (mg/cm²)", color=color)
            ax.plot(t_common, Qtot_interp, ':', linewidth=(PLOT_LINEWIDTH - 0.4), alpha=0.8, label=f"{name} Qtotal (mg/cm²)", color=color)

            any_plotted = True

        except Exception as e:
            app.logger.exception("Failed to simulate/plot agent %r: %s", agent, e)
            continue

    if not any_plotted:
        app.logger.info("No agent data plotted for detailed dermal abs/evap — drawing fallback sample.")
        for i, agent in enumerate(agent_properties):
            name = agent[0]
            MW = float(agent[1])
            logP = float(agent[3])
            tf_hours = float(GLOBAL_T_COMMON[-1]) / 60.0
            t_hr, Qt_hr, Qet_hr = simulate_dermal_absorption_agent_fallback(MW, logP, 21.0, 1.0, Mo=Mo, tf_hours=tf_hours)
            tmin_minutes = t_hr * 60.0
            Qt = np.interp(GLOBAL_T_COMMON, tmin_minutes, Qt_hr, left=0.0, right=Qt_hr[-1])
            Qet = np.interp(GLOBAL_T_COMMON, tmin_minutes, Qet_hr, left=0.0, right=Qet_hr[-1])
            ax.plot(GLOBAL_T_COMMON, Qt, '-', linewidth=PLOT_LINEWIDTH, label=f"{name} Qabs (mg/cm²)", color=cmap(i % cmap.N))
            ax.plot(GLOBAL_T_COMMON, Qet, '--', linewidth=PLOT_LINEWIDTH, label=f"{name} Qevap (mg/cm²)", color=cmap(i % cmap.N))
        ax.set_ylim(0, 1.0)

    ax.set_xlabel("Time (minutes)", fontsize=PLOT_LABEL_FS)
    ax.set_ylabel("Mass (mg/cm²)", fontsize=PLOT_LABEL_FS)
    ax.set_title("Absorption vs Evaporation — Detailed Dermal Model (All agents)", fontsize=PLOT_TITLE_FS)
    apply_uniform_style(ax)

    handles, labels = ax.get_legend_handles_labels()
    agent_names_seen = set()
    dedup_handles = []
    dedup_labels = []

    for h, l in zip(handles, labels):
        agent_name = l.split(' Q')[0]
        if agent_name not in agent_names_seen:
            dedup_handles.append(h)
            dedup_labels.append(agent_name)
            agent_names_seen.add(agent_name)

    if len(dedup_handles) > 0:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='gray', linestyle='-'),
                        Line2D([0], [0], color='gray', linestyle='--'),
                        Line2D([0], [0], color='gray', linestyle=':')]
        custom_labels = ['Qabs', 'Qevap', 'Qtotal']

        final_handles = dedup_handles + custom_lines
        final_labels = dedup_labels + custom_labels

        ax.legend(final_handles, final_labels, fontsize=PLOT_LEGEND_FS, bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=True)
    else:
        ax.legend(fontsize=PLOT_LEGEND_FS, bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=True)

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout(rect=[0, 0, 0.78, 1.0])
    return fig_to_base64(fig)


# Replaced detailed dermal absorption plotting with original ODE-based solver


# -------------------------
# Flask route (unchanged)
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    selected_agents = []
    other_agents = ""
    applied_dose = ""
    agent_info = []
    combined_images = {'abs_vs_evap': None}

    if request.method == 'POST':
        selected_agents = request.form.getlist('agents')
        other_agents = request.form.get('other_agents', '')
        applied_dose = request.form.get('applied_dose', '0')

        try:
            dose = float(applied_dose)
            if dose <= 0:
                dose = 1e-3
                applied_dose = str(dose)
        except:
            dose = 1e-3
            applied_dose = str(dose)

        if other_agents.strip():
            others = [a.strip() for a in other_agents.split(',') if a.strip()]
            selected_agents.extend(others)

        # dedupe while preserving order
        seen = set()
        selected_agents = [x for x in selected_agents if not (x in seen or seen.add(x))]

        if not selected_agents:
            agents_to_run = agent_properties
        else:
            agents_to_run = []
            for name in selected_agents:
                matched = next((a for a in agent_properties
                                 if a[0].strip().lower() == name.strip().lower()), None)
                if matched:
                    agents_to_run.append(matched)
                else:
                    props = get_agent_data(name)
                    if props.get('MW') is not None and props.get('logP') is not None:
                        mw = props['MW']
                        rho = 1.0
                        logP = props['logP']
                        kObs = 1.0  # Default kObs for Chebyshev model
                        agents_to_run.append((name, mw, rho, logP, kObs))
                    else:
                        msg = f"Skipping {name}: insufficient properties (need MW and logP)."
                        app.logger.warning(msg)
                        agent_info.append(msg)

        if not agents_to_run:
            agents_to_run = agent_properties

        # compute Msat / status for each agent
        for ag in agents_to_run:
            name = ag[0]
            mw = float(ag[1]) if len(ag) > 1 else None
            logP = float(ag[3]) if len(ag) > 3 else None

            props = get_agent_data(name)
            Sw = props.get('Sw')
            if Sw is None:
                Sw = 1.0
            else:
                Sw = float(Sw)

            try:
                Msat, Kscw, Csat = compute_Msat_from_logP_sw(float(logP), float(Sw))
            except Exception as e:
                Msat, Kscw, Csat = (None, None, None)
                app.logger.debug("Failed to compute Msat for %s: %s", name, e)

            Mo = float(dose)
            if Msat is None:
                status = "Msat computation failed"
            else:
                if Mo > Msat:
                    status = "Above saturation (two-phase expected)"
                else:
                    status = "Below saturation (single-phase expected)"

            info_lines = [
                f"Agent: {name}",
                f"  MW = {mw:.4g}" if mw is not None else "  MW = N/A",
                f"  logP = {logP:.3g}" if logP is not None else "  logP = N/A",
                f"  Sw = {Sw:.4g}",
            ]
            if Msat is not None:
                info_lines.append(f"  Msat = {Msat:.3e} mg/cm²")
                info_lines.append(f"  Csat (sc) = {Csat:.3e} mg/cm³")
                info_lines.append(f"  Kscw = {Kscw:.3e}")
            else:
                info_lines.append("  Msat / Csat / Kscw = N/A")

            info_lines.append(f"  Applied dose (Mo) = {Mo:.3e} mg/cm²")
            info_lines.append(f"  Status: {status}")

            info_text = "\n".join(info_lines)
            agent_info.append(info_text)
            app.logger.info(info_text)

        # Dermal absorption graph only (original ODE-based model)
        try:
            # Use a 25-hour horizon as per original code
            combined_images['abs_vs_evap'] = combined_plot_dermal_absorption_original(agents_to_run, Mo=dose, tf_hours=25.0)
        except Exception as e:
            app.logger.error('Dermal absorption plotting failed: %s', e)
            agent_info.append(f'Dermal absorption plot error: {e}')

    # Build agent list using detected name column
    if df_agents is not None:
        name_col = df_agents_canonmap.get('name', df_agents.columns[0])
        agent_list = df_agents[name_col].astype(str).tolist()
    else:
        agent_list = [a[0] for a in agent_properties]

    return render_template('index.html',
                            agent_list=agent_list,
                            selected_agents=selected_agents,
                            other_agents=other_agents,
                            applied_dose=applied_dose,
                            agent_info=agent_info,
                            combined_images=combined_images)


if __name__ == "__main__":
    app.run(debug=True)
