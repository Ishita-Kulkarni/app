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

# Import comparison simulation
from comparison_model import run_comparison_simulation, run_comparison_simulation_multi

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# -------------------------
# Global plotting configuration
# -------------------------
PLOT_FIGSIZE = (10, 6)  # Uniform size with comparison plots
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
    props = {'MW': None, 'logP': None, 'Sw': None, 'CAS': None, 'SMILES': None, 'formula': None,
             'nc': None, 'nh': None, 'no': None, 'nn': None, 'nring': None}

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
                
                # Try to get atom counts directly from CSV
                nc_val = safe_get_from_row(row0, ['nc', 'nC', 'n_c', 'carbon_count'])
                nh_val = safe_get_from_row(row0, ['nh', 'nH', 'n_h', 'hydrogen_count'])
                no_val = safe_get_from_row(row0, ['no', 'nO', 'n_o', 'oxygen_count'])
                nn_val = safe_get_from_row(row0, ['nn', 'nN', 'n_n', 'nitrogen_count'])
                nring_val = safe_get_from_row(row0, ['nring', 'nRing', 'n_ring', 'ring_count'])

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
                        # CSV stores solubility in mg/L, convert to mg/cm³ (divide by 1000)
                        props['Sw'] = float(sw) / 1000.0
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
                
                # Store atom counts if found in CSV
                try:
                    if nc_val is not None and str(nc_val).strip().lower() not in ('none', 'nan', ''):
                        props['nc'] = int(float(nc_val))
                except:
                    pass
                try:
                    if nh_val is not None and str(nh_val).strip().lower() not in ('none', 'nan', ''):
                        props['nh'] = int(float(nh_val))
                except:
                    pass
                try:
                    if no_val is not None and str(no_val).strip().lower() not in ('none', 'nan', ''):
                        props['no'] = int(float(no_val))
                except:
                    pass
                try:
                    if nn_val is not None and str(nn_val).strip().lower() not in ('none', 'nan', ''):
                        props['nn'] = int(float(nn_val))
                except:
                    pass
                try:
                    if nring_val is not None and str(nring_val).strip().lower() not in ('none', 'nan', ''):
                        props['nring'] = int(float(nring_val))
                except:
                    pass
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
    
    # Parse atom counts from formula or SMILES if not directly available
    if any(props[k] is None for k in ('nc', 'nh', 'no', 'nn')):
        atom_counts = None
        if props['formula']:
            atom_counts = parse_molecular_formula(props['formula'])
        elif props['SMILES']:
            atom_counts = get_atom_counts_from_smiles(props['SMILES'])
        
        if atom_counts:
            if props['nc'] is None:
                props['nc'] = atom_counts.get('nc', 0)
            if props['nh'] is None:
                props['nh'] = atom_counts.get('nh', 0)
            if props['no'] is None:
                props['no'] = atom_counts.get('no', 0)
            if props['nn'] is None:
                props['nn'] = atom_counts.get('nn', 0)
    
    # Try to calculate nring from SMILES if not available
    if props['nring'] is None and props['SMILES']:
        try:
            props['nring'] = count_rings_from_smiles(props['SMILES'])
        except:
            props['nring'] = 0
    
    # Default atom counts to 0 if still missing
    if props['nc'] is None:
        props['nc'] = 0
    if props['nh'] is None:
        props['nh'] = 0
    if props['no'] is None:
        props['no'] = 0
    if props['nn'] is None:
        props['nn'] = 0
    if props['nring'] is None:
        props['nring'] = 0

    # WARNING: PubChem solubility values are bulk aqueous solubility, NOT SC-effective solubility!
    # The model requires SC (stratum corneum) effective solubility which is typically
    # 10-1000x lower than bulk aqueous values. For accurate results, manually provide Sw values.
    # This automated fetch is provided only as a rough fallback and may produce incorrect results.
    if props['Sw'] is None:
        c_for_sol = pubchem_lookup(agent_name)
        if c_for_sol:
            # Try to get solubility from compound record (experimental data section)
            # Note: This returns mg/L typically, we convert to mg/mL (= mg/cm³)
            try:
                # PubChem stores solubility in experimental properties, not as a simple field
                # We'll attempt basic lookup but warn user this needs manual verification
                sol_mg_per_L = getattr(c_for_sol, 'solubility', None)
                if sol_mg_per_L and isinstance(sol_mg_per_L, (int, float, str)):
                    try:
                        val = float(sol_mg_per_L)
                        # Convert mg/L to mg/cm³: divide by 1000
                        props['Sw'] = val / 1000.0
                        app.logger.warning(f"Using PubChem bulk aqueous solubility for {agent_name}: {val} mg/L = {props['Sw']} mg/cm³. This may NOT be SC-effective solubility!")
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                app.logger.debug(f"Could not fetch PubChem solubility for {agent_name}: {e}")

    return props


# -------------------------
# Helper: parse molecular formula to get atom counts
# -------------------------
def parse_molecular_formula(formula):
    """
    Parse molecular formula string to extract atom counts.
    Returns dict with keys: nc, nh, no, nn (carbon, hydrogen, oxygen, nitrogen)
    Example: "C3H8O" -> {'nc': 3, 'nh': 8, 'no': 1, 'nn': 0}
    """
    import re
    if not formula or not isinstance(formula, str):
        return {'nc': 0, 'nh': 0, 'no': 0, 'nn': 0}
    
    counts = {'nc': 0, 'nh': 0, 'no': 0, 'nn': 0}
    
    # Pattern: Element followed by optional number
    # Matches: C3, H8, O, N2, etc.
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        if element == 'C':
            counts['nc'] += count
        elif element == 'H':
            counts['nh'] += count
        elif element == 'O':
            counts['no'] += count
        elif element == 'N':
            counts['nn'] += count
    
    return counts


def get_atom_counts_from_smiles(smiles):
    """
    Estimate atom counts from SMILES string.
    This is a simple character-counting approach.
    """
    if not smiles or not isinstance(smiles, str):
        return {'nc': 0, 'nh': 0, 'no': 0, 'nn': 0}
    
    # Simple counting (not perfect but reasonable)
    nc = smiles.count('C') + smiles.count('c')  # aromatic and aliphatic
    nh = smiles.count('H')
    no = smiles.count('O') + smiles.count('o')
    nn = smiles.count('N') + smiles.count('n')
    
    return {'nc': nc, 'nh': nh, 'no': no, 'nn': nn}


def count_rings_from_smiles(smiles):
    """
    Count rings in SMILES by counting ring closure digits.
    Each unique digit represents a ring closure.
    """
    if not smiles or not isinstance(smiles, str):
        return 0
    import re
    # Find all ring digits (1-9 or %10-%99)
    ring_digits = re.findall(r'%\d\d|\d', smiles)
    # Count unique ring closures (each appears twice in SMILES)
    return len(set(ring_digits))


# -------------------------
# Helper: compute Msat and related values
# -------------------------
def compute_Msat_from_logP_sw(logP, Sw, fdep=0.1, hsc=13.4e-4):
    Kow = 10.0 ** logP
    Kscw = 0.040 * Kow**0.81 + 4.06 * Kow**0.27 + 0.359
    Csat = Kscw * Sw
    Msat = fdep * hsc * Csat
    return Msat, Kscw, Csat

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

############################################################
# Dermal absorption model (ODE-based implementation matching reference)
############################################################

def solve_dermal_absorption_original(MW, logKow, Pvap, Sw, Mo=1e-3, tf_hours=25.0, 
                                     nc=None, nh=None, no=None, nn=None, nring=None,
                                     formula=None, smiles=None):
    """
    Solve the dermal absorption model from the original script for a single agent.
    Returns t_hours, Qt (absorbed), Qet (evaporated).
    
    Parameters:
    -----------
    nc, nh, no, nn, nring: atom counts (preferred, passed directly from get_agent_data)
    formula: molecular formula string (e.g., "C4H10O2PS2" for VX) - fallback if atom counts not provided
    smiles: SMILES string as fallback for atom counts
    """
    # Use provided atom counts or calculate from formula/SMILES
    if nc is None or nh is None or no is None or nn is None:
        if formula:
            atom_counts = parse_molecular_formula(formula)
        elif smiles:
            atom_counts = get_atom_counts_from_smiles(smiles)
        else:
            raise ValueError(f"Cannot calculate atom counts: nc/nh/no/nn not provided and formula/SMILES both missing")
        
        nc = atom_counts.get('nc', 0) if nc is None else nc
        nh = atom_counts.get('nh', 0) if nh is None else nh
        no = atom_counts.get('no', 0) if no is None else no
        nn = atom_counts.get('nn', 0) if nn is None else nn
    
    # Calculate nring from SMILES if not provided
    if nring is None:
        if smiles:
            try:
                nring = count_rings_from_smiles(smiles)
            except:
                nring = 0
        else:
            nring = 0

    # System parameters (EXACTLY matching reference code)
    hsc = 13.4 * 1e-4  # Stratum corneum thickness (cm)
    h1 = hsc           # Thickness (cm)
    fdep = 0.1         # Fraction of depletion layer
    tf = float(tf_hours) * 3600.0  # Total simulation time (seconds)

    # Environmental parameters for evaporation (EXACTLY matching reference)
    u = 16.5      # Air velocity (cm/s)
    L = 13.4      # Length scale (cm)
    R = 62.37     # Gas constant (mL·Torr/K·mmol)
    T = 298.15    # Temperature (K)

    # Calculated transport properties (EXACTLY matching reference)
    Kow = 10**logKow
    Kscw = 0.040 * Kow**0.81 + 4.06 * Kow**0.27 + 0.359
    Ksc = Kscw
    Csat = Ksc * Sw

    # Permeability and diffusion (EXACTLY matching reference)
    logPscw = -2.8 + 0.66*logKow - 0.0056*MW
    Pscw = 10**logPscw
    kp = Pscw
    D1 = (Pscw * h1 / Kscw) / 3600
    Dsc = D1
    dif1 = Dsc

    # Gas phase transport for evaporation (EXACTLY matching reference)
    Vp = Pvap * 133.322  # Convert torr to Pa
    S = 16.5*nc + 1.98*nh + 5.69*nn + 5.48*no - 20.42*nring
    Dg = (10**(-3) * T**1.75 * (1/29 + 1/MW)**(1/2)) / (S**(1/3) + (20.1)**(1/3))**2
    kg = (3260/3600) * Dg**(2/3) * np.sqrt(u/L)
    K = (kg * Pvap * MW) / (R * T) * 1 / (kp * Sw)
    chi = K
    
    # Atom counts calculated: nc={nc}, nh={nh}, no={no}, nn={nn}, nring={nring}

    # System properties (EXACTLY matching reference)
    Msat = fdep * hsc * Csat
    Msurfo = Mo - Msat
    kevaprho = chi * Dsc * Csat / hsc
    
    # Log key parameters
    app.logger.info(f"logKow={logKow:.3f}, MW={MW:.2f}")
    app.logger.info(f"logPscw={logPscw:.3f}, Pscw={Pscw:.3e}")
    app.logger.info(f"Kscw={Kscw:.3f}, h1={h1:.3e}")
    app.logger.info(f"D1={D1:.3e}, Dsc={D1:.3e}")
    app.logger.info(f"Mo = {Mo:.2e} mg/cm², Msat = {Msat:.6f} mg/cm²")

    # Decide regime
    use_above = bool(Mo > Msat)

    if use_above:
        # PHASE 1 DIFFERENTIAL EQUATIONS (PERFECT SINK) - EXACTLY matching reference
        def differential_system(t, y):
            Ts2, Ts3, Ts4, Ts5, Ts6, Ts7, Ts8, Ts9, Ts10, Ts11, Qt, Qst = y
            denom = (h1 - fdep * hsc)**2
            
            dTs2_dt = (1/denom) * dif1 * (0.0 + 3699.34*Csat - 4857.68*Ts2 + 1398.55*Ts3 - 
                                          339.574*Ts4 + 155.38*Ts5 - 94.1732*Ts6 + 68.0747*Ts7 - 
                                          56.3546*Ts8 + 52.8474*Ts9 - 57.2106*Ts10 + 79.6158*Ts11)
            
            dTs3_dt = (1/denom) * dif1 * (0.0 - 216.102*Csat + 623.883*Ts2 - 666.927*Ts3 + 
                                          314.69*Ts4 - 79.6341*Ts5 + 38.4789*Ts6 - 24.8625*Ts7 + 
                                          19.3336*Ts8 - 17.4769*Ts9 + 18.5229*Ts10 - 25.5064*Ts11)
            
            dTs4_dt = (1/denom) * dif1 * (0.0 + 51.3576*Csat - 103.349*Ts2 + 214.683*Ts3 - 
                                          290.794*Ts4 + 156.338*Ts5 - 40.8813*Ts6 + 20.5419*Ts7 - 
                                          14.0087*Ts8 + 11.7504*Ts9 - 11.934*Ts10 + 16.0906*Ts11)
            
            dTs5_dt = (1/denom) * dif1 * (0.0 - 21.3451*Csat + 38.4943*Ts2 - 44.2193*Ts3 + 
                                          127.211*Ts4 - 188.669*Ts5 + 108.676*Ts6 - 29.5263*Ts7 + 
                                          15.5706*Ts8 - 11.4047*Ts9 + 10.7477*Ts10 - 13.9713*Ts11)
            
            dTs6_dt = (1/denom) * dif1 * (0.0 + 12.362*Csat - 21.2652*Ts2 + 19.4771*Ts3 - 
                                          30.3147*Ts4 + 99.0212*Ts5 - 155.568*Ts6 + 94.3278*Ts7 - 
                                          26.904*Ts8 + 15.2359*Ts9 - 12.5905*Ts10 + 15.3766*Ts11)
            
            dTs7_dt = (1/denom) * dif1 * (0.0 - 9.15831*Csat + 15.3765*Ts2 - 12.5905*Ts3 + 
                                          15.2361*Ts4 - 26.9043*Ts5 + 94.3281*Ts6 - 155.568*Ts7 + 
                                          99.0218*Ts8 - 30.3155*Ts9 + 19.4782*Ts10 - 21.267*Ts11)
            
            dTs8_dt = (1/denom) * dif1 * (0.0 + 8.43803*Csat - 13.9738*Ts2 + 10.7495*Ts3 - 
                                          11.4063*Ts4 + 15.5722*Ts5 - 29.5281*Ts6 + 108.678*Ts7 - 
                                          188.671*Ts8 + 127.214*Ts9 - 44.2228*Ts10 + 38.4997*Ts11)
            
            dTs9_dt = (1/denom) * dif1 * (0.0 - 9.80631*Csat + 16.1085*Ts2 - 11.9465*Ts3 + 
                                          11.7612*Ts4 - 14.0191*Ts5 + 20.5525*Ts6 - 40.8927*Ts7 + 
                                          156.351*Ts8 - 290.809*Ts9 + 214.702*Ts10 - 103.378*Ts11)
            
            dTs10_dt = (1/denom) * dif1 * (0.0 + 15.6376*Csat - 25.5668*Ts2 + 18.5648*Ts3 - 
                                           17.5131*Ts4 + 19.3681*Ts5 - 24.8978*Ts6 + 38.5167*Ts7 - 
                                           79.6766*Ts8 + 314.74*Ts9 - 666.991*Ts10 + 623.98*Ts11)
            
            dTs11_dt = (1/denom) * dif1 * (0.0 - 48.899*Csat + 79.7604*Ts2 - 57.3114*Ts3 + 
                                           52.935*Ts4 - 56.4392*Ts5 + 68.1623*Ts6 - 94.2691*Ts7 + 
                                           155.491*Ts8 - 339.708*Ts9 + 1398.72*Ts10 - 4857.95*Ts11)
            
            dQt_dt = -(1/(h1 - fdep*hsc)) * dif1 * (0.0 - 0.999885*Csat + 1.63008*Ts2 - 
                                                    1.16854*Ts3 + 1.07425*Ts4 - 1.1361*Ts5 + 
                                                    1.35336*Ts6 - 1.82683*Ts7 + 2.87424*Ts8 - 
                                                    5.62786*Ts9 + 16.1529*Ts10 - 123.326*Ts11)
            
            dQst_dt = -kevaprho + (1/(h1 - fdep*hsc)) * dif1 * (0.0 - 110.997*Csat + 
                                                                 123.321*Ts2 - 16.1502*Ts3 + 
                                                                 5.62579*Ts4 - 2.87258*Ts5 + 
                                                                 1.82542*Ts6 - 1.3521*Ts7 + 
                                                                 1.13491*Ts8 - 1.07303*Ts9 + 
                                                                 1.16715*Ts10 - 1.6281*Ts11)
            
            return [dTs2_dt, dTs3_dt, dTs4_dt, dTs5_dt, dTs6_dt, dTs7_dt, dTs8_dt, 
                    dTs9_dt, dTs10_dt, dTs11_dt, dQt_dt, dQst_dt]

        # PHASE 2 DIFFERENTIAL EQUATIONS (NO PERFECT SINK) - EXACTLY matching reference
        def differential_system_phase2(t, y):
            Vr2, Vr3, Vr4, Vr5, Vr6, Vr7, Vr8, Vr9, Vr10, Vr11, Q1t, Qet = y
            
            common_denom = -(110.997/hsc) - (1.0 * chi)/hsc
            flux_term = (0.0 - (123.321 * Vr2)/hsc + (16.1502 * Vr3)/hsc - 
                         (5.62579 * Vr4)/hsc + (2.87258 * Vr5)/hsc - 
                         (1.82542 * Vr6)/hsc + (1.3521 * Vr7)/hsc - 
                         (1.13491 * Vr8)/hsc + (1.07303 * Vr9)/hsc - 
                         (1.16715 * Vr10)/hsc + (1.6281 * Vr11)/hsc)
            
            dVr2_dt = ((1/hsc**2) * dif1 * (0.0 - 4857.68*Vr2 + 1398.55*Vr3 - 339.574*Vr4 + 
                                            155.38*Vr5 - 94.1732*Vr6 + 68.0747*Vr7 - 56.3546*Vr8 + 
                                            52.8474*Vr9 - 57.2106*Vr10 + 79.6158*Vr11 + 
                                            (1/common_denom) * 3699.34 * flux_term))
            
            dVr3_dt = ((1/hsc**2) * dif1 * (0.0 + 623.883*Vr2 - 666.927*Vr3 + 314.69*Vr4 - 
                                            79.6341*Vr5 + 38.4789*Vr6 - 24.8625*Vr7 + 19.3336*Vr8 - 
                                            17.4769*Vr9 + 18.5229*Vr10 - 25.5064*Vr11 - 
                                            (1/common_denom) * 216.102 * flux_term))
            
            dVr4_dt = ((1/hsc**2) * dif1 * (0.0 - 103.349*Vr2 + 214.683*Vr3 - 290.794*Vr4 + 
                                            156.338*Vr5 - 40.8813*Vr6 + 20.5419*Vr7 - 14.0087*Vr8 + 
                                            11.7504*Vr9 - 11.934*Vr10 + 16.0906*Vr11 + 
                                            (1/common_denom) * 51.3576 * flux_term))
            
            dVr5_dt = ((1/hsc**2) * dif1 * (0.0 + 38.4943*Vr2 - 44.2193*Vr3 + 127.211*Vr4 - 
                                            188.669*Vr5 + 108.676*Vr6 - 29.5263*Vr7 + 15.5706*Vr8 - 
                                            11.4047*Vr9 + 10.7477*Vr10 - 13.9713*Vr11 - 
                                            (1/common_denom) * 21.3451 * flux_term))
            
            dVr6_dt = ((1/hsc**2) * dif1 * (0.0 - 21.2652*Vr2 + 19.4771*Vr3 - 30.3147*Vr4 + 
                                            99.0212*Vr5 - 155.568*Vr6 + 94.3278*Vr7 - 26.904*Vr8 + 
                                            15.2359*Vr9 - 12.5905*Vr10 + 15.3766*Vr11 + 
                                            (1/common_denom) * 12.362 * flux_term))
            
            dVr7_dt = ((1/hsc**2) * dif1 * (0.0 + 15.3765*Vr2 - 12.5905*Vr3 + 15.2361*Vr4 - 
                                            26.9043*Vr5 + 94.3281*Vr6 - 155.568*Vr7 + 99.0218*Vr8 - 
                                            30.3155*Vr9 + 19.4782*Vr10 - 21.267*Vr11 - 
                                            (1/common_denom) * 9.15831 * flux_term))
            
            dVr8_dt = ((1/hsc**2) * dif1 * (0.0 - 13.9738*Vr2 + 10.7495*Vr3 - 11.4063*Vr4 + 
                                            15.5722*Vr5 - 29.5281*Vr6 + 108.678*Vr7 - 188.671*Vr8 + 
                                            127.214*Vr9 - 44.2228*Vr10 + 38.4997*Vr11 + 
                                            (1/common_denom) * 8.43803 * flux_term))
            
            dVr9_dt = ((1/hsc**2) * dif1 * (0.0 + 16.1085*Vr2 - 11.9465*Vr3 + 11.7612*Vr4 - 
                                            14.0191*Vr5 + 20.5525*Vr6 - 40.8927*Vr7 + 156.351*Vr8 - 
                                            290.809*Vr9 + 214.702*Vr10 - 103.378*Vr11 - 
                                            (1/common_denom) * 9.80631 * flux_term))
            
            dVr10_dt = ((1/hsc**2) * dif1 * (0.0 - 25.5668*Vr2 + 18.5648*Vr3 - 17.5131*Vr4 + 
                                             19.3681*Vr5 - 24.8978*Vr6 + 38.5167*Vr7 - 79.6766*Vr8 + 
                                             314.74*Vr9 - 666.991*Vr10 + 623.98*Vr11 + 
                                             (1/common_denom) * 15.6376 * flux_term))
            
            dVr11_dt = ((1/hsc**2) * dif1 * (0.0 + 79.7604*Vr2 - 57.3114*Vr3 + 52.935*Vr4 - 
                                             56.4392*Vr5 + 68.1623*Vr6 - 94.2691*Vr7 + 155.491*Vr8 - 
                                             339.708*Vr9 + 1398.72*Vr10 - 4857.95*Vr11 - 
                                             (1/common_denom) * 48.899 * flux_term))
            
            dQ1t_dt = (-(1/hsc) * dif1 * (0.0 + 1.63008*Vr2 - 1.16854*Vr3 + 
                                          1.07425*Vr4 - 1.1361*Vr5 + 1.35336*Vr6 - 
                                          1.82683*Vr7 + 2.87424*Vr8 - 5.62786*Vr9 + 
                                          16.1529*Vr10 - 123.326*Vr11 - 
                                          (1/common_denom) * 0.999885 * flux_term))
            
            dQet_dt = (1/hsc * dif1 * (0.0 + 123.321*Vr2 - 16.1502*Vr3 + 
                                       5.62579*Vr4 - 2.87258*Vr5 + 1.82542*Vr6 - 
                                       1.3521*Vr7 + 1.13491*Vr8 - 1.07303*Vr9 + 
                                       1.16715*Vr10 - 1.6281*Vr11 - 
                                       (1/common_denom) * 110.997 * flux_term))
            
            return [dVr2_dt, dVr3_dt, dVr4_dt, dVr5_dt, dVr6_dt, dVr7_dt, dVr8_dt, 
                    dVr9_dt, dVr10_dt, dVr11_dt, dQ1t_dt, dQet_dt]

        # RUN ABOVE SATURATION SIMULATION
        def qst_negative(t, y):
            return y[11]
        qst_negative.terminal = True
        qst_negative.direction = -1

        y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Msurfo]
        t_span = (0, tf)
        t_eval = np.linspace(0, tf, 10000)

        sol = solve_ivp(differential_system, t_span, y0, t_eval=t_eval, 
                        events=qst_negative, method='RK45', rtol=1e-8, atol=1e-10)

        t = sol.t
        Ts2, Ts3, Ts4, Ts5, Ts6, Ts7, Ts8, Ts9, Ts10, Ts11, Qt, Qst = sol.y
        Qet = kevaprho * t

        # Check for phase transition
        if sol.t_events[0].size > 0:
            ttrans = sol.t_events[0][0]
            
            # Setup for Phase 2
            ROOT = np.array([1e-20, 0.0130467, 0.0674683, 0.160295, 0.283302, 
                             0.425563, 0.574437, 0.716698, 0.839705, 0.932532, 
                             0.986953, 1.0])
            
            N2 = 12
            Tstrans = np.zeros(N2)
            Tstrans[0] = Csat
            Tstrans[1:11] = [np.interp(ttrans, t, Ts2), np.interp(ttrans, t, Ts3), 
                             np.interp(ttrans, t, Ts4), np.interp(ttrans, t, Ts5),
                             np.interp(ttrans, t, Ts6), np.interp(ttrans, t, Ts7),
                             np.interp(ttrans, t, Ts8), np.interp(ttrans, t, Ts9),
                             np.interp(ttrans, t, Ts10), np.interp(ttrans, t, Ts11)]
            Tstrans[11] = 0
            
            # Create interpolation
            zTop = ROOT * fdep * hsc
            zSkin = ROOT * (1 - fdep) * hsc + fdep * hsc
            zAll = np.concatenate([zTop[1:-1], zSkin[1:-1]])
            vTop = np.full(len(zTop[1:-1]), Csat)
            vSkin = Tstrans[2:N2-1]
            vAll = np.concatenate([vTop, vSkin])
            
            if len(zAll) != len(vAll):
                n_internal = len(ROOT) - 2
                zTop_internal = zTop[1:-1]
                zSkin_internal = zSkin[1:-1]
                vTop = np.full(len(zTop_internal), Csat)
                available_temps = len(Tstrans) - 2
                if available_temps >= len(zSkin_internal):
                    vSkin = Tstrans[1:1+len(zSkin_internal)]
                else:
                    vSkin = np.zeros(len(zSkin_internal))
                    vSkin[:available_temps] = Tstrans[1:1+available_temps]
                zAll = np.concatenate([zTop_internal, zSkin_internal])
                vAll = np.concatenate([vTop, vSkin])
            
            unique_indices = np.unique(zAll, return_index=True)[1]
            zAll_unique = zAll[unique_indices]
            vAll_unique = vAll[unique_indices]
            interpCombined = interp1d(zAll_unique, vAll_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Phase 2 initial conditions
            Vr_init = np.zeros(10)
            ROOT_positions = ROOT[1:-1]
            for i in range(10):
                pos = ROOT_positions[i] * hsc
                Vr_init[i] = interpCombined(pos)
            
            qtinitp2 = np.interp(ttrans, t, Qt)
            qevapinitp2 = kevaprho * ttrans
            y0_phase2 = np.concatenate([Vr_init, [qtinitp2, qevapinitp2]])
            
            # Solve Phase 2
            phase2_duration = tf - ttrans
            if phase2_duration >= 1.0:
                t_span_phase2 = (0, phase2_duration)
                t_eval_phase2 = np.linspace(0, phase2_duration, 5000)
                sol2 = solve_ivp(differential_system_phase2, t_span_phase2, y0_phase2, 
                                 t_eval=t_eval_phase2, method='RK45', rtol=1e-8, atol=1e-10)
                
                if sol2.success and sol2.y.size > 0:
                    t2 = sol2.t
                    Vr2, Vr3, Vr4, Vr5, Vr6, Vr7, Vr8, Vr9, Vr10, Vr11, Q1t_phase2, Qet_phase2 = sol2.y
                    
                    # Combine results
                    t_combined = np.concatenate([t, t2 + ttrans])
                    Qt_combined = np.concatenate([Qt, Q1t_phase2])
                    Qet_combined = np.concatenate([Qet, Qet_phase2])
                else:
                    t_combined = t
                    Qt_combined = Qt
                    Qet_combined = Qet
            else:
                t_combined = t
                Qt_combined = Qt
                Qet_combined = Qet
        else:
            t_combined = t
            Qt_combined = Qt
            Qet_combined = Qet

        return t_combined / 3600.0, Qt_combined, Qet_combined

    else:
        # BELOW SATURATION MODEL (SINGLE-PHASE) - EXACTLY matching reference
        # Adjust parameters for below saturation model
        h = h1  # Use the same thickness (h1 = 0.00134 cm)
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

        # Initial conditions (exact from reference code)
        y0 = np.zeros(22)
        # Ts_2 through Ts_11 = Tso
        y0[0:10] = Tso
        # Tv_2 through Tv_11 = 0
        y0[10:20] = 0.0
        # Qt[0] = 0, Qet[0] = 0
        y0[20] = 0.0  # Qt
        y0[21] = 0.0  # Qet
        
        # Debug: Log initial conditions
        # Initial conditions calculated

        # Time span
        t_span = (0, tf)
        t_eval = np.linspace(0, tf, 1000)

        # Solve the ODE system (LSODA method as in reference)
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
        
        # Debug: Log solver status
        # Solver completed
        app.logger.info(f"Solver success: {sol.success}, message: {sol.message}")
        
        if sol.success:
            # Extract solutions
            Qt_combined = sol.y[20]  # Qt is at index 20
            Qet_combined = sol.y[21]  # Qet is at index 21
            t_combined = sol.t
            app.logger.info(f"Solution range: Qt={Qt_combined.min():.3e} to {Qt_combined.max():.3e}, Qet={Qet_combined.min():.3e} to {Qet_combined.max():.3e}")
        else:
            # Fallback: simple exponentials if solver fails
            app.logger.warning(f"ODE solver failed, using fallback exponentials")
            t_combined = t_eval
            tau_abs = max(0.05 * tf, 1.0)
            tau_evap = max(0.02 * tf, 1.0)
            Qt_combined = Mo * 0.3 * (1 - np.exp(-t_combined / tau_abs))
            Qet_combined = Mo * 0.6 * (1 - np.exp(-t_combined / tau_evap))
        
        return t_combined / 3600.0, Qt_combined, Qet_combined


def combined_plot_dermal_absorption_original(agents, Mo=1e-3, tf_hours=25.0, custom_properties=None):
    """Plot all selected agents with clean styling:
    - Each agent gets one color; styles indicate quantity: Qabs (-), Qevap (--), Qtotal (:)
    - Legend shows agent names once plus a compact style key for the three quantities
    - Right axis shows percentage of initial dose linked to the left axis scale
    - custom_properties: dict mapping agent names to property dicts {agent: {prop: value}}
    """
    if custom_properties is None:
        custom_properties = {}
    
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
            
            # Use custom properties if provided, otherwise get from data
            if name in custom_properties and 'Sw' in custom_properties[name]:
                Sw = float(custom_properties[name]['Sw']) / 1000.0  # Convert mg/L to mg/cm³
            else:
                Sw = props.get('Sw')
                if Sw is None:
                    raise ValueError(f"Solubility not found for {name}. Please provide it manually.")
                Sw = float(Sw)
            
            if name in custom_properties and 'Pvap' in custom_properties[name]:
                Pvap = float(custom_properties[name]['Pvap'])
            else:
                Pvap = props.get('Pvap')
                if Pvap is None:
                    pc = pubchem_lookup(name)
                    if pc:
                        try:
                            vap_prop = safe_pubchem_prop(pc, 'Vapor Pressure')
                            if vap_prop:
                                Pvap = float(vap_prop)
                        except Exception:
                            pass
                if Pvap is None:
                    raise ValueError(f"Vapor pressure not found for {name}. Please provide it manually.")
                else:
                    Pvap = float(Pvap)
            
            # Get formula and SMILES for atom count calculation
            formula = props.get('formula')
            smiles_str = props.get('SMILES')
            
            # Get atom counts (already parsed in get_agent_data)
            nc = props.get('nc', 0)
            nh = props.get('nh', 0)
            no = props.get('no', 0)
            nn = props.get('nn', 0)
            nring = props.get('nring', 0)

            t_hr, Qt, Qet = solve_dermal_absorption_original(
                MW, logP, Pvap, Sw, Mo=Mo, tf_hours=tf_hours, 
                nc=nc, nh=nh, no=no, nn=nn, nring=nring,
                formula=formula, smiles=smiles_str
            )
            Qtot = Qt + Qet
            
            # Log what we're about to plot
            app.logger.info(f"Plotting {name}: Qt range {Qt.min():.3e} to {Qt.max():.3e}, Qet range {Qet.min():.3e} to {Qet.max():.3e}")

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
              fontsize=PLOT_LEGEND_FS, frameon=True, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), borderaxespad=0, ncol=3)

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

    apply_uniform_style(ax)
    
    # Adjust layout to prevent legend overlap - leave space at bottom
    plt.subplots_adjust(bottom=0.25)
    
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
# Flask route (unchanged)
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    selected_agents = []
    other_agents = ""
    applied_dose = ""
    td = "0.0"
    tsim = "25.0"
    agent_info = []
    combined_images = {'abs_vs_evap': None, 'comparison_plot': None}
    missing_properties = {}  # Track agents with missing properties {agent_name: [list of missing props]}
    custom_properties = {}  # Store custom property values {agent_name: {prop: value}}

    if request.method == 'POST':
        selected_agents = request.form.getlist('agents')
        other_agents = request.form.get('other_agents', '')
        applied_dose = request.form.get('applied_dose', '0')
        td = request.form.get('td', '0.0')
        tsim = request.form.get('tsim', '25.0')
        
        # Collect custom property values from form
        for key in request.form:
            if key.startswith('prop_'):
                # Format: prop_AgentName_PropertyName
                parts = key.replace('prop_', '', 1).rsplit('_', 1)
                if len(parts) == 2:
                    agent_name, prop_name = parts
                    try:
                        if agent_name not in custom_properties:
                            custom_properties[agent_name] = {}
                        custom_properties[agent_name][prop_name] = float(request.form[key])
                    except ValueError:
                        pass

        try:
            dose = float(applied_dose)
            # Validate dose is positive
            if dose <= 0:
                raise ValueError("Applied dose must be greater than 0")
        except ValueError as ve:
            agent_info.append(f"Error: {str(ve)}")
            dose = None
        except:
            agent_info.append("Error: Invalid applied dose value. Please enter a valid number.")
            dose = None
        
        if dose is None:
            # Build agent list for dropdown
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
                                td=td,
                                tsim=tsim,
                                agent_info=agent_info,
                                combined_images=combined_images,
                                missing_properties={})

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
                    
                    # Check for custom properties first
                    mw = None
                    logP = None
                    if name in custom_properties:
                        mw = custom_properties[name].get('MW')
                        logP = custom_properties[name].get('logP')
                    
                    # Fall back to fetched properties
                    if mw is None:
                        mw = props.get('MW')
                    if logP is None:
                        logP = props.get('logP')
                    
                    if mw is not None and logP is not None:
                        # Simplified agent tuple: (name, MW, placeholder, logP, placeholder)
                        # Middle values maintained for backward compatibility with existing code
                        agents_to_run.append((name, mw, 1.0, logP, 1.0))
                    else:
                        # Track which properties are missing
                        if name not in missing_properties:
                            missing_properties[name] = []
                        if mw is None:
                            missing_properties[name].append('MW')
                        if logP is None:
                            missing_properties[name].append('logP')

        if not agents_to_run:
            agents_to_run = agent_properties

        # First pass: check for missing properties (Sw and Pvap)
        for ag in agents_to_run:
            name = ag[0]
            props = get_agent_data(name)
            
            # Check for missing Sw
            Sw = props.get('Sw')
            if name in custom_properties and 'Sw' in custom_properties[name]:
                Sw = float(custom_properties[name]['Sw']) / 1000.0  # Convert mg/L to mg/cm³
            if Sw is None:
                if name not in missing_properties:
                    missing_properties[name] = []
                if 'Sw' not in missing_properties[name]:
                    missing_properties[name].append('Sw')
            
            # Check for missing Pvap
            Pvap = props.get('Pvap')
            if name in custom_properties and 'Pvap' in custom_properties[name]:
                Pvap = custom_properties[name]['Pvap']
            if Pvap is None:
                # Try PubChem one more time
                pc = pubchem_lookup(name)
                if pc:
                    try:
                        vap_prop = safe_pubchem_prop(pc, 'Vapor Pressure')
                        if vap_prop:
                            Pvap = float(vap_prop)
                    except Exception:
                        pass
                
                if Pvap is None:
                    if name not in missing_properties:
                        missing_properties[name] = []
                    if 'Pvap' not in missing_properties[name]:
                        missing_properties[name].append('Pvap')
        
        # If we have missing property values, return early with prompt
        if missing_properties:
            # Build agent list for dropdown
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
                                td=td,
                                tsim=tsim,
                                agent_info=agent_info,
                                combined_images=combined_images,
                                missing_properties=missing_properties)
        
        # Second pass: compute Msat / status and detailed parameters for each agent
        for ag in agents_to_run:
            name = ag[0]
            mw = float(ag[1]) if len(ag) > 1 else None
            logP = float(ag[3]) if len(ag) > 3 else None

            props = get_agent_data(name)
            
            # Use custom properties if provided, otherwise use fetched data
            Sw = props.get('Sw')
            if name in custom_properties and 'Sw' in custom_properties[name]:
                Sw = float(custom_properties[name]['Sw']) / 1000.0  # Convert mg/L to mg/cm³
            elif Sw is not None:
                Sw = float(Sw)
            
            # Get Pvap for calculations
            Pvap = props.get('Pvap')
            if name in custom_properties and 'Pvap' in custom_properties[name]:
                Pvap = custom_properties[name]['Pvap']
            else:
                if Pvap is None:
                    pc = pubchem_lookup(name)
                    if pc:
                        try:
                            vap_prop = safe_pubchem_prop(pc, 'Vapor Pressure')
                            if vap_prop:
                                Pvap = float(vap_prop)
                        except Exception:
                            pass
            if Pvap is not None:
                Pvap = float(Pvap)

            try:
                Msat, Kscw, Csat = compute_Msat_from_logP_sw(float(logP), float(Sw))
                
                # Calculate additional parameters matching the debug output
                hsc = 13.4 * 1e-4
                h1 = hsc
                fdep = 0.1
                Kow = 10**logP
                logPscw = -2.8 + 0.66*logP - 0.0056*mw
                Pscw = 10**logPscw
                D1 = (Pscw * h1 / Kscw) / 3600
                Dsc = D1
                
                # K calculation diagnostic (matching terminal output)
                kg_calc = None
                K_calc = None
                if Pvap is not None and Sw is not None:
                    # Get atom counts for gas diffusivity
                    formula = props.get('formula')
                    smiles_str = props.get('SMILES')
                    nc = props.get('nc', max(1, int(mw / 14)))
                    nh = props.get('nh', max(1, int(mw / 7)))
                    no = props.get('no', 1)
                    nn = props.get('nn', 0)
                    nring = props.get('nring', 0)
                    
                    # Gas phase transport
                    R = 62.37
                    T = 298.15
                    u = 16.5
                    L = 13.4
                    Vp = Pvap * 133.322  # torr to Pa
                    S = 16.5*nc + 1.98*nh + 5.69*nn + 5.48*no - 20.42*nring
                    Dg = (10**(-3) * T**1.75 * (1/29 + 1/mw)**(1/2)) / (S**(1/3) + (20.1)**(1/3))**2
                    kg_calc = (3260/3600) * Dg**(2/3) * np.sqrt(u/L)
                    kp = Pscw
                    K_calc = (kg_calc * Pvap * mw) / (R * T) * 1 / (kp * Sw) if (kp * Sw) > 0 else None
                
            except Exception as e:
                Msat, Kscw, Csat = (None, None, None)
                logPscw, Pscw, D1, Dsc, h1 = (None, None, None, None, None)
                kg_calc, K_calc = (None, None)
                app.logger.debug("Failed to compute parameters for %s: %s", name, e)

            Mo = float(dose)
            if Msat is None:
                status = "Msat computation failed"
            else:
                if Mo > Msat:
                    status = "Above saturation (two-phase expected)"
                else:
                    status = "Below saturation (single-phase expected)"

            # Display only requested parameters
            app.logger.info(f"\n{'='*60}")
            app.logger.info(f"Agent: {name}")
            app.logger.info(f"{'='*60}")
            if logP is not None and mw is not None:
                app.logger.info(f"logKow={logP:.3f}, MW={mw:.2f}")
            if logPscw is not None and Pscw is not None:
                app.logger.info(f"logPscw={logPscw:.3f}, Pscw={Pscw:.3e}")
            if Kscw is not None and h1 is not None:
                app.logger.info(f"Kscw={Kscw:.3f}, h1={h1:.3e}")
            if D1 is not None and Dsc is not None:
                app.logger.info(f"D1={D1:.3e}, Dsc={Dsc:.3e}")
            app.logger.info(f"Mo = {Mo:.2e} mg/cm², Msat = {Msat:.6f} mg/cm²" if Msat is not None else f"Mo = {Mo:.2e} mg/cm², Msat = N/A")
            
            info_lines = [
                f"Agent: {name}",
                f"MW: {mw:.4g} g/mol" if mw is not None else "MW: N/A",
                f"logP: {logP:.3g}" if logP is not None else "logP: N/A",
                f"Sw: {Sw*1000:.4g} mg/L" if Sw is not None else "Sw: N/A",  # Convert mg/cm³ back to mg/L for display
            ]
            if Pvap is not None:
                info_lines.append(f"Pvap: {Pvap:.4g} torr")
            else:
                info_lines.append("Pvap: N/A")
            if Msat is not None:
                info_lines.append(f"Msat: {Msat:.3e} mg/cm²")
                info_lines.append(f"Csat (sc): {Csat:.3e} mg/cm³")
            else:
                info_lines.append("Msat: N/A")
                info_lines.append("Csat (sc): N/A")
            info_lines.append(f"Applied dose (Mo): {Mo:.3e} mg/cm²")
            info_lines.append(f"Status: {status}")
            
            # Add K calculation diagnostic
            if kg_calc is not None and K_calc is not None:
                info_lines.append("")  # Blank line for separation
                info_lines.append("K Calculation Diagnostic:")
                info_lines.append(f"  kg: {kg_calc:.6f} cm/s")
                info_lines.append(f"  kp : {Pscw:.6f}" if Pscw is not None else "  kp: N/A")
                info_lines.append(f"  K : {K_calc:.6f}")
                

            info_text = "\n".join(info_lines)
            agent_info.append(info_text)

        # Dermal absorption graph only (original ODE-based model) - UNCHANGED
        try:
            # Use a 25-hour horizon as per original code
            combined_images['abs_vs_evap'] = combined_plot_dermal_absorption_original(
                agents_to_run, Mo=dose, tf_hours=25.0, custom_properties=custom_properties
            )
        except Exception as e:
            app.logger.error('Dermal absorption plotting failed: %s', e)
            agent_info.append(f'Dermal absorption plot error: {e}')

        # NEW: Generate comparison plot (with/without decontamination)
        try:
            td_hours = float(td)
            tsim_hours = float(tsim)
            
            # Process ALL selected agents for comparison (not just first)
            if agents_to_run:
                comparison_result = run_comparison_simulation_multi(
                    agents_to_run=agents_to_run,
                    dose=dose,
                    sim_hours=tsim_hours,
                    td_hours=td_hours,
                    custom_properties=custom_properties
                )
                
                if comparison_result and comparison_result.get('comparison_plot'):
                    combined_images['comparison_plot'] = comparison_result['comparison_plot']
                    app.logger.info(f"✓ Comparison plot added to combined_images (size: {len(comparison_result['comparison_plot'])} bytes)")
                else:
                    app.logger.error(f"✗ Comparison plot is None or empty! Status: {comparison_result.get('status', 'unknown')}")
                
                app.logger.info(f"Comparison plot for {len(agents_to_run)} agent(s), Mo={dose}, td={td_hours}h, tsim={tsim_hours}h")
        except Exception as e:
            app.logger.error('Comparison plot failed: %s', e)
            agent_info.append(f'Comparison plot error: {e}')

    # Build agent list using detected name column
    if df_agents is not None:
        name_col = df_agents_canonmap.get('name', df_agents.columns[0])
        agent_list = df_agents[name_col].astype(str).tolist()
    else:
        agent_list = [a[0] for a in agent_properties]

    # Debug: Log what's in combined_images before rendering
    app.logger.info(f"Rendering template with combined_images keys: {list(combined_images.keys())}")
    app.logger.info(f"  abs_vs_evap: {'Present' if combined_images.get('abs_vs_evap') else 'None'}")
    app.logger.info(f"  comparison_plot: {'Present (' + str(len(combined_images.get('comparison_plot', ''))) + ' bytes)' if combined_images.get('comparison_plot') else 'None'}")

    return render_template('index.html',
                            agent_list=agent_list,
                            selected_agents=selected_agents,
                            other_agents=other_agents,
                            applied_dose=applied_dose,
                            td=td,
                            tsim=tsim,
                            agent_info=agent_info,
                            combined_images=combined_images,
                            missing_properties={})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
