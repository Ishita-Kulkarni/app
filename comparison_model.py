"""
Decontamination simulation - WITH RSDL
This uses the EXACT model from the comprehensive dermal absorption code.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64


def calculate_k_from_structure(chemical_name, MW, formula=None, SMILES=None, C_RSDL_M=0.01):
    """
    Simplified k calculation for common CWA agents.
    Returns k in 1/s for use in dermal model.
    
    Based on literature RSDL reactivity data.
    """
    print(f"DEBUG calculate_k_from_structure: chemical_name='{chemical_name}', formula='{formula}'")
    
    # Known agent-specific k2 rates (M^-1 min^-1) from literature/estimates
    known_k2 = {
        # G-series nerve agents (very fast with RSDL)
        'sarin': 180.0,
        'gb': 180.0,
        'soman': 150.0,
        'gd': 150.0,
        'tabun': 90.0,
        'ga': 90.0,
        # V-series nerve agents (fast but variable)
        'vx': 50.0,
        'vr': 45.0,
        'vm': 40.0,
        # Sulfur mustards (MUCH SLOWER with RSDL)
        'sulfur mustard': 0.15,
        'hd': 0.15,
        'mustard': 0.15,
        'bis(2-chloroethyl)sulfide': 0.1,  # Actual sulfur mustard structure
        'sesquimustard': 0.1,
        'q': 0.1,
    }
    
    # Check if chemical name matches known agent
    name_lower = chemical_name.lower()
    k2 = None
    for agent, rate in known_k2.items():
        if agent in name_lower:
            k2 = rate
            break
    
    # Estimate based on formula if not found
    if k2 is None and formula:
        print(f"  DEBUG: Estimating k2 from formula: {formula}")
        # Organophosphorus compounds (contains P)
        if 'P' in formula:
            if 'F' in formula:
                k2 = 150.0  # G-series like (very reactive)
            elif 'Cl' in formula:
                k2 = 100.0
            elif 'CN' in formula or ('C' in formula and 'N' in formula):
                k2 = 80.0
            elif 'S' in formula:
                k2 = 30.0
            else:
                k2 = 10.0
            print(f"  DEBUG: OP compound detected, k2 = {k2}")
        # Halogenated compounds (contains Cl/Br/F but NOT P)
        # Note: Check this BEFORE sulfur mustards to avoid misclassification
        elif any(x in formula for x in ['Cl', 'Br', 'F']):
            k2 = 0.01  # Low reactivity for general halogenated compounds
            print(f"  DEBUG: Halogenated compound detected, k2 = {k2}")
        # Sulfur compounds (contains S but not halogenated OP)
        elif 'S' in formula:
            k2 = 0.001  # Very low reactivity
            print(f"  DEBUG: Sulfur compound detected, k2 = {k2}")
        else:
            k2 = 0.001  # Very low reactivity
            print(f"  DEBUG: Generic compound, k2 = {k2}")
    
    # Final fallback
    if k2 is None:
        k2 = 0.001
    
    # Convert k2 (M^-1 min^-1) to k (s^-1)
    k_per_min = k2 * C_RSDL_M
    k = k_per_min / 60.0
    
    return k


def run_comparison_simulation(MW, logKow, Pvap, Sw, Mo, chemical_name, sim_hours=25.0, td_hours=0.0, SMILES=None, formula=None):
    """
    Run dermal absorption simulation WITH decontamination using comprehensive model.
    This is the EXACT model from your comprehensive code, adapted for web app use.
    
    Parameters:
    -----------
    MW : float
        Molecular weight (g/mol)
    logKow : float
        Log octanol-water partition coefficient  
    Pvap : float
        Vapor pressure (torr)
    Sw : float
        Water solubility (mg/cm³)
    Mo : float
        Initial dose (mg/cm²)
    chemical_name : str
        Name of the chemical
    sim_hours : float
        Total simulation time (hours)
    td_hours : float
        Decontamination activation time (hours)
    SMILES : str, optional
        SMILES string for structure-based k calculation
    formula : str, optional
        Chemical formula for k calculation
    
    Returns:
    --------
    dict with 'comparison_plot' (base64 PNG) and 'status'
    """
    
    try:
        # Convert times
        tf = sim_hours * 3600.0  # seconds
        td = td_hours * 3600.0   # seconds
        
        # Fixed system parameters (EXACT from comprehensive model)
        hsc = 13.4 * 1e-4  # cm (stratum corneum thickness)
        h1 = hsc
        fdep = 0.1
        u = 16.5
        L = 13.4  # cm - air boundary layer for gas phase (kg calculation)
        R = 62.37
        T = 298.15
        
        # Decontamination parameters - calculate chemical-specific k
        C_RSDL_M = 0.01  # Default RSDL concentration (Molarity)
        
        # Try to calculate chemical-specific k
        try:
            k = calculate_k_from_structure(
                chemical_name=chemical_name,
                MW=MW,
                formula=formula,
                SMILES=SMILES,
                C_RSDL_M=C_RSDL_M
            )
            print(f"✓ Calculated chemical-specific k = {k:.3e} s⁻¹")
        except Exception as e:
            print(f"⚠ Could not calculate chemical-specific k: {e}")
            k = 0.008 / 3600.0  # Fallback default
            print(f"⚠ Using default k = {k:.3e} s⁻¹")
        
        # Derived properties (EXACT formulas from comprehensive model)
        Kow = 10.0**logKow
        Kscw = 0.040 * Kow**0.81 + 4.06 * Kow**0.27 + 0.359
        Ksc = Kscw
        Csat = Ksc * Sw
        
        # Permeability/diffusivity (EXACT from comprehensive model)
        logPscw = -2.8 + 0.66*logKow - 0.0056*MW
        Pscw = 10.0**logPscw
        kp = Pscw
        D1 = (Pscw * h1 / Kscw) / 3600.0
        Dsc = D1
        
        # Estimate atom counts for evaporation calculation
        nc = max(1, int(MW / 14))
        nh = max(1, int(MW / 7))
        no = 1
        nn = 0
        nring = 0
        
        # Gas phase transport for evaporation (EXACT from comprehensive model)
        Vp = Pvap * 133.322  # torr -> Pa
        S = 16.5*nc + 1.98*nh + 5.69*nn + 5.48*no - 20.42*nring
        Dg = (10**(-3) * T**1.75 * (1/29 + 1/MW)**0.5) / ((S**(1/3) + 20.1**(1/3))**2)
        kg = (3260/3600) * Dg**(2/3) * np.sqrt(u/L)
        
        # Evaporation coefficient K (EXACT from comprehensive model)
        K = (kg * Pvap * MW) / (R * T) * 1 / (kp * Sw)
        print(f"DEBUG K calculation:")
        print(f"  kg={kg:.6e}, Pvap={Pvap:.6e}, MW={MW:.6e}")
        print(f"  R={R:.6e}, T={T:.6e}, kp={kp:.6e}, Sw={Sw:.6e}")
        print(f"  K = {K:.6f}")
        
        # Note: chi = K + k*hsc²*Ksc/Dsc (hsc is stratum corneum thickness)
        # ks_eff = k * hsc * Ksc (computed inline in RHS functions)
        
        # Saturation check
        Msat = fdep * hsc * Csat
        Msurfo = Mo - Msat
        use_above_saturation = Mo > Msat
        
        # Helper functions (EXACT from comprehensive model)
        def heaviside_on(t, td_sec):
            return 1.0 if (t >= td_sec) else 0.0
        
        def k_eff_of(t, k_val, td_sec):
            return k_val * heaviside_on(t, td_sec)
        
        def ks_eff_of(t, k_val, L_sc_val, Ksc_val, td_sec):
            return (k_val * L_sc_val * Ksc_val) * heaviside_on(t, td_sec)
        
        # =============================================================
        # ABOVE SATURATION: Phase 1 -> Phase 2 (EXACT from comprehensive model)
        # =============================================================
        if use_above_saturation:
            
            def phase1_rhs(t, y):
                Ts2, Ts3, Ts4, Ts5, Ts6, Ts7, Ts8, Ts9, Ts10, Ts11, Qt, Qst, Qevap, Qreact = y
                
                # Activation (use hsc for surface reaction as in comprehensive model)
                on = 1.0 if (t >= td) else 0.0
                k_eff = k * on
                ks_eff = (k * hsc * Ksc) * on  # ks = k * hsc * Ksc
                chi = K + (hsc / Dsc) * ks_eff  # chi = K + k*hsc²*Ksc/Dsc
                
                # Phase 1: Constant Csat boundary
                kevaprho = (Dsc * Csat / hsc) * chi  # Total flux = (Dsc*Csat/hsc) * chi
                evap_rate = K * Dsc * Csat / hsc  # Evaporation component
                kevap = evap_rate
                
                denom = (h1 - fdep * hsc)**2
                
                # Diffusion equations (EXACT coefficients from comprehensive model)
                dTs2_dt = (1/denom) * (Dsc * (3699.34*Csat - 4857.68*Ts2 + 1398.55*Ts3 -
                           339.574*Ts4 + 155.38*Ts5 - 94.1732*Ts6 + 68.0747*Ts7 -
                           56.3546*Ts8 + 52.8474*Ts9 - 57.2106*Ts10 + 79.6158*Ts11)) - k_eff*Ts2

                dTs3_dt = (1/denom) * (Dsc * (-216.102*Csat + 623.883*Ts2 - 666.927*Ts3 +
                           314.69*Ts4 - 79.6341*Ts5 + 38.4789*Ts6 - 24.8625*Ts7 +
                           19.3336*Ts8 - 17.4769*Ts9 + 18.5229*Ts10 - 25.5064*Ts11)) - k_eff*Ts3

                dTs4_dt = (1/denom) * (Dsc * (51.3576*Csat - 103.349*Ts2 + 214.683*Ts3 -
                           290.794*Ts4 + 156.338*Ts5 - 40.8813*Ts6 + 20.5419*Ts7 -
                           14.0087*Ts8 + 11.7504*Ts9 - 11.934*Ts10 + 16.0906*Ts11)) - k_eff*Ts4

                dTs5_dt = (1/denom) * (Dsc * (-21.3451*Csat + 38.4943*Ts2 - 44.2193*Ts3 +
                           127.211*Ts4 - 188.669*Ts5 + 108.676*Ts6 - 29.5263*Ts7 +
                           15.5706*Ts8 - 11.4047*Ts9 + 10.7477*Ts10 - 13.9713*Ts11)) - k_eff*Ts5

                dTs6_dt = (1/denom) * (Dsc * (12.362*Csat - 21.2652*Ts2 + 19.4771*Ts3 -
                           30.3147*Ts4 + 99.0212*Ts5 - 155.568*Ts6 + 94.3278*Ts7 -
                           26.904*Ts8 + 15.2359*Ts9 - 12.5905*Ts10 + 15.3766*Ts11)) - k_eff*Ts6

                dTs7_dt = (1/denom) * (Dsc * (-9.15831*Csat + 15.3765*Ts2 - 12.5905*Ts3 +
                           15.2361*Ts4 - 26.9043*Ts5 + 94.3281*Ts6 - 155.568*Ts7 +
                           99.0218*Ts8 - 30.3155*Ts9 + 19.4782*Ts10 - 21.267*Ts11)) - k_eff*Ts7

                dTs8_dt = (1/denom) * (Dsc * (8.43803*Csat - 13.9738*Ts2 + 10.7495*Ts3 -
                           11.4063*Ts4 + 15.5722*Ts5 - 29.5281*Ts6 + 108.678*Ts7 -
                           188.671*Ts8 + 127.214*Ts9 - 44.2228*Ts10 + 38.4997*Ts11)) - k_eff*Ts8

                dTs9_dt = (1/denom) * (Dsc * (-9.80631*Csat + 16.1085*Ts2 - 11.9465*Ts3 +
                           11.7612*Ts4 - 14.0191*Ts5 + 20.5525*Ts6 - 40.8927*Ts7 +
                           156.351*Ts8 - 290.809*Ts9 + 214.702*Ts10 - 103.378*Ts11)) - k_eff*Ts9

                dTs10_dt = (1/denom) * (Dsc * (15.6376*Csat - 25.5668*Ts2 + 18.5648*Ts3 -
                            17.5131*Ts4 + 19.3681*Ts5 - 24.8978*Ts6 + 38.5167*Ts7 -
                            79.6766*Ts8 + 314.74*Ts9 - 666.991*Ts10 + 623.98*Ts11)) - k_eff*Ts10

                dTs11_dt = (1/denom) * (Dsc * (-48.899*Csat + 79.7604*Ts2 - 57.3114*Ts3 +
                            52.935*Ts4 - 56.4392*Ts5 + 68.1623*Ts6 - 94.2691*Ts7 +
                            155.491*Ts8 - 339.708*Ts9 + 1398.72*Ts10 - 4857.95*Ts11)) - k_eff*Ts11
                
                dQt_dt = -(1/(h1 - fdep*hsc)) * Dsc * (-0.999885*Csat + 1.63008*Ts2 - 1.16854*Ts3 +
                          1.07425*Ts4 - 1.1361*Ts5 + 1.35336*Ts6 - 1.82683*Ts7 +
                          2.87424*Ts8 - 5.62786*Ts9 + 16.1529*Ts10 - 123.326*Ts11)
                
                dQevap_dt = kevap
                
                dQst_dt = -kevaprho + (1/(h1 - fdep*hsc)) * Dsc * (-110.997*Csat + 123.321*Ts2 -
                          16.1502*Ts3 + 5.62579*Ts4 - 2.87258*Ts5 + 1.82542*Ts6 -
                          1.3521*Ts7 + 1.13491*Ts8 - 1.07303*Ts9 + 1.16715*Ts10 - 1.6281*Ts11)
                
                # Bulk reaction using Gauss-Legendre weights
                W_interior = np.array([0.0333356721543441, 0.0747256745752903, 0.109543181257991,
                                      0.134633359654998, 0.147762112357376, 0.147762112357376,
                                      0.134633359654998, 0.109543181257991, 0.0747256745752903,
                                      0.0333356721543441])
                
                Ts_vec = np.array([Ts2, Ts3, Ts4, Ts5, Ts6, Ts7, Ts8, Ts9, Ts10, Ts11], dtype=float)
                Ts_vec_clipped = np.clip(Ts_vec, 0.0, None)
                int_Ts_vec = np.dot(W_interior, Ts_vec_clipped)
                dQreact_dt = k_eff * int_Ts_vec * (h1 - fdep*hsc)
                
                return [dTs2_dt, dTs3_dt, dTs4_dt, dTs5_dt, dTs6_dt, dTs7_dt, dTs8_dt, dTs9_dt,
                       dTs10_dt, dTs11_dt, dQt_dt, dQst_dt, dQevap_dt, dQreact_dt]
            
            # Initial conditions
            y0_p1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Msurfo, 0, 0]
            t_span = (0, tf)
            t_eval = np.linspace(0, tf, 5000)
            
            def event_Qst_zero(t, y):
                return y[11]
            event_Qst_zero.terminal = True
            event_Qst_zero.direction = -1
            
            sol1 = solve_ivp(phase1_rhs, t_span, y0_p1, t_eval=t_eval, events=event_Qst_zero,
                           method="RK45", rtol=1e-9, atol=1e-11)
            
            t1 = sol1.t
            Ts2, Ts3, Ts4, Ts5, Ts6, Ts7, Ts8, Ts9, Ts10, Ts11, Qt1, Qst1, Qevap1, Qreact1 = sol1.y
            
            # Check for Phase 2 transition
            if sol1.t_events[0].size > 0:
                ttrans = sol1.t_events[0][0]
                
                # Build Phase 2 initial profile (EXACT from comprehensive model)
                ROOT = np.array([1e-20, 0.0130467, 0.0674683, 0.160295, 0.283302,
                                0.425563, 0.574437, 0.716698, 0.839705, 0.932532,
                                0.986953, 1.0])
                
                N2 = 12
                Tstrans = np.zeros(N2)
                Tstrans[0] = Csat
                Tstrans[1:11] = [np.interp(ttrans, t1, Ts2), np.interp(ttrans, t1, Ts3),
                                np.interp(ttrans, t1, Ts4), np.interp(ttrans, t1, Ts5),
                                np.interp(ttrans, t1, Ts6), np.interp(ttrans, t1, Ts7),
                                np.interp(ttrans, t1, Ts8), np.interp(ttrans, t1, Ts9),
                                np.interp(ttrans, t1, Ts10), np.interp(ttrans, t1, Ts11)]
                Tstrans[11] = 0
                
                zTop = ROOT * fdep * hsc
                zSkin = ROOT * (1 - fdep) * hsc + fdep * hsc
                zAll = np.concatenate([zTop[1:-1], zSkin[1:-1]])
                vTop = np.full(len(zTop[1:-1]), Csat)
                vSkin = Tstrans[1:11]
                vAll = np.concatenate([vTop, vSkin])
                
                unique_indices = np.unique(zAll, return_index=True)[1]
                zAll_u = zAll[unique_indices]
                vAll_u = vAll[unique_indices]
                interpCombined = interp1d(zAll_u, vAll_u, kind='linear', bounds_error=False, fill_value='extrapolate')
                
                Vr_init = np.zeros(10)
                ROOT_positions = ROOT[1:-1]
                for i in range(10):
                    Vr_init[i] = interpCombined(ROOT_positions[i] * hsc)
                
                Qt_init_p2 = float(np.interp(ttrans, t1, Qt1))
                Qevap_init_p2 = float(np.interp(ttrans, t1, Qevap1))
                Qreact_init_p2 = float(np.interp(ttrans, t1, Qreact1))
                
                # Phase 2 RHS (EXACT from comprehensive model)
                def phase2_rhs(t2, y2):
                    Vr2, Vr3, Vr4, Vr5, Vr6, Vr7, Vr8, Vr9, Vr10, Vr11, Qt, Qevap, Qreact = y2
                    
                    # Activation & chi (matching comprehensive model formulation)
                    keff = k_eff_of(t2, k, td)
                    on_val = 1.0 if (t2 >= td) else 0.0
                    ks_eff = (k * hsc * Ksc) * on_val  # ks = k * hsc * Ksc
                    chi = K + (hsc / Dsc) * ks_eff  # chi = K + k*hsc²*Ksc/Dsc
                    
                    common_denom = -(110.997/hsc) - (1.0 * chi)/hsc
                    flux_term = (0.0 - (123.321 * Vr2)/hsc + (16.1502 * Vr3)/hsc -
                                (5.62579 * Vr4)/hsc + (2.87258 * Vr5)/hsc - (1.82542 * Vr6)/hsc +
                                (1.3521 * Vr7)/hsc - (1.13491 * Vr8)/hsc + (1.07303 * Vr9)/hsc -
                                (1.16715 * Vr10)/hsc + (1.6281 * Vr11)/hsc)
                    
                    dVr2_dt = ((1/hsc**2) * Dsc * (-4857.68*Vr2 + 1398.55*Vr3 - 339.574*Vr4 +
                                155.38*Vr5 - 94.1732*Vr6 + 68.0747*Vr7 - 56.3546*Vr8 +
                                52.8474*Vr9 - 57.2106*Vr10 + 79.6158*Vr11 +
                                (1/common_denom) * 3699.34 * flux_term)) - keff*Vr2

                    dVr3_dt = ((1/hsc**2) * Dsc * (623.883*Vr2 - 666.927*Vr3 + 314.69*Vr4 -
                                79.6341*Vr5 + 38.4789*Vr6 - 24.8625*Vr7 + 19.3336*Vr8 -
                                17.4769*Vr9 + 18.5229*Vr10 - 25.5064*Vr11 -
                                (1/common_denom) * 216.102 * flux_term)) - keff*Vr3

                    dVr4_dt = ((1/hsc**2) * Dsc * (-103.349*Vr2 + 214.683*Vr3 - 290.794*Vr4 +
                                156.338*Vr5 - 40.8813*Vr6 + 20.5419*Vr7 - 14.0087*Vr8 +
                                11.7504*Vr9 - 11.934*Vr10 + 16.0906*Vr11 +
                                (1/common_denom) * 51.3576 * flux_term)) - keff*Vr4

                    dVr5_dt = ((1/hsc**2) * Dsc * (38.4943*Vr2 - 44.2193*Vr3 + 127.211*Vr4 -
                                188.669*Vr5 + 108.676*Vr6 - 29.5263*Vr7 + 15.5706*Vr8 -
                                11.4047*Vr9 + 10.7477*Vr10 - 13.9713*Vr11 -
                                (1/common_denom) * 21.3451 * flux_term)) - keff*Vr5

                    dVr6_dt = ((1/hsc**2) * Dsc * (-21.2652*Vr2 + 19.4771*Vr3 - 30.3147*Vr4 +
                                99.0212*Vr5 - 155.568*Vr6 + 94.3278*Vr7 - 26.904*Vr8 +
                                15.2359*Vr9 - 12.5905*Vr10 + 15.3766*Vr11 +
                                (1/common_denom) * 12.362 * flux_term)) - keff*Vr6

                    dVr7_dt = ((1/hsc**2) * Dsc * (15.3765*Vr2 - 12.5905*Vr3 + 15.2361*Vr4 -
                                26.9043*Vr5 + 94.3281*Vr6 - 155.568*Vr7 + 99.0218*Vr8 -
                                30.3155*Vr9 + 19.4782*Vr10 - 21.267*Vr11 -
                                (1/common_denom) * 9.15831 * flux_term)) - keff*Vr7

                    dVr8_dt = ((1/hsc**2) * Dsc * (-13.9738*Vr2 + 10.7495*Vr3 - 11.4063*Vr4 +
                                15.5722*Vr5 - 29.5281*Vr6 + 108.678*Vr7 - 188.671*Vr8 +
                                127.214*Vr9 - 44.2228*Vr10 + 38.4997*Vr11 +
                                (1/common_denom) * 8.43803 * flux_term)) - keff*Vr8

                    dVr9_dt = ((1/hsc**2) * Dsc * (16.1085*Vr2 - 11.9465*Vr3 + 11.7612*Vr4 -
                                14.0191*Vr5 + 20.5525*Vr6 - 40.8927*Vr7 + 156.351*Vr8 -
                                290.809*Vr9 + 214.702*Vr10 - 103.378*Vr11 -
                                (1/common_denom) * 9.80631 * flux_term)) - keff*Vr9

                    dVr10_dt = ((1/hsc**2) * Dsc * (-25.5668*Vr2 + 18.5648*Vr3 - 17.5131*Vr4 +
                                19.3681*Vr5 - 24.8978*Vr6 + 38.5167*Vr7 - 79.6766*Vr8 +
                                314.74*Vr9 - 666.991*Vr10 + 623.98*Vr11 +
                                (1/common_denom) * 15.6376 * flux_term)) - keff*Vr10

                    dVr11_dt = ((1/hsc**2) * Dsc * (79.7604*Vr2 - 57.3114*Vr3 + 52.935*Vr4 -
                                56.4392*Vr5 + 68.1623*Vr6 - 94.2691*Vr7 + 155.491*Vr8 -
                                339.708*Vr9 + 1398.72*Vr10 - 4857.95*Vr11 -
                                (1/common_denom) * 48.899 * flux_term)) - keff*Vr11

                    dQt_dt = (-(1/hsc) * Dsc * (1.63008*Vr2 - 1.16854*Vr3 + 1.07425*Vr4 - 1.1361*Vr5 +
                              1.35336*Vr6 - 1.82683*Vr7 + 2.87424*Vr8 - 5.62786*Vr9 + 16.1529*Vr10 -
                              123.326*Vr11 - (1/common_denom) * 0.999885 * flux_term))
                    
                    dQevap_dt = ((1/hsc) * Dsc * (123.321*Vr2 - 16.1502*Vr3 + 5.62579*Vr4 - 2.87258*Vr5 +
                                 1.82542*Vr6 - 1.3521*Vr7 + 1.13491*Vr8 - 1.07303*Vr9 + 1.16715*Vr10 -
                                 1.6281*Vr11 - (1/common_denom) * 110.997 * flux_term))
                    
                    Vr_vec = np.array([Vr2, Vr3, Vr4, Vr5, Vr6, Vr7, Vr8, Vr9, Vr10, Vr11], dtype=float)
                    W_interior = np.array([0.0333356721543441, 0.0747256745752903, 0.109543181257991,
                                          0.134633359654998, 0.147762112357376, 0.147762112357376,
                                          0.134633359654998, 0.109543181257991, 0.0747256745752903,
                                          0.0333356721543441])
                    Vr_vec_clipped = np.clip(Vr_vec, 0.0, None)
                    int_Vr_vec = np.dot(W_interior, Vr_vec_clipped)
                    dQreact_dt = keff * int_Vr_vec * hsc
                    
                    return [dVr2_dt, dVr3_dt, dVr4_dt, dVr5_dt, dVr6_dt, dVr7_dt, dVr8_dt, dVr9_dt,
                           dVr10_dt, dVr11_dt, dQt_dt, dQevap_dt, dQreact_dt]
                
                phase2_duration = tf - ttrans
                if phase2_duration > 0:
                    y0_p2 = np.concatenate([Vr_init, [Qt_init_p2, Qevap_init_p2, Qreact_init_p2]])
                    t_span2 = (0, phase2_duration)
                    t_eval2 = np.linspace(0, phase2_duration, 5000)
                    sol2 = solve_ivp(phase2_rhs, t_span2, y0_p2, t_eval=t_eval2,
                                   method="RK45", rtol=1e-9, atol=1e-11)
                    
                    if sol2.success and sol2.y.size > 0:
                        t2 = sol2.t
                        Qt_p2, Qevap_p2, Qreact_p2 = sol2.y[10], sol2.y[11], sol2.y[12]
                        
                        t_combined = np.concatenate([t1, t2 + ttrans])
                        Qt_combined = np.concatenate([Qt1, Qt_p2])
                        Qst_combined = np.concatenate([Qst1, Qevap_p2])  # Phase2 uses Qevap for total
                        Qreact_combined = np.concatenate([Qreact1, Qreact_p2])
                    else:
                        t_combined = t1
                        Qt_combined = Qt1
                        Qst_combined = Qst1
                        Qreact_combined = Qreact1
                else:
                    t_combined = t1
                    Qt_combined = Qt1
                    Qst_combined = Qst1
                    Qreact_combined = Qreact1
            else:
                t_combined = t1
                Qt_combined = Qt1
                Qst_combined = Qst1
                Qreact_combined = Qreact1
        
        # =============================================================
        # BELOW SATURATION (Single-Phase) - EXACT from comprehensive model
        # =============================================================
        else:
            h = h1
            Tso = Mo / (fdep * h)
            
            def below_rhs(t, y):
                """
                y = [Ts(0..9), Tv(0..9), Qt, Qevap, Qreact]
                """
                Ts = y[0:10]
                Tv = y[10:20]
                
                # Activation & chi (matching comprehensive model formulation)
                keff = k_eff_of(t, k, td)
                on_val = 1.0 if (t >= td) else 0.0
                ks_eff = (k * hsc * Ksc) * on_val  # ks = k * hsc * Ksc
                chi = K + (hsc / Dsc) * ks_eff  # chi = K + k*hsc²*Ksc/Dsc
                
                denom1 = ((111.0 * D1) / (fdep * h) +
                         (110.997 * D1) / (h - 1.0 * fdep * h) +
                         (0.998546 * D1) / (fdep**2 * h**2 * (-(110.997/(fdep * h)) - (1.0 * chi)/h)))
                
                chi_term_denom = -(110.997/(fdep * h)) - (1.0 * chi)/h
                
                complex_flux = (0.0 - (1.63008 * D1 * Ts[0])/(fdep * h) -
                               (123.307 * D1 * Ts[0])/(fdep**2 * h**2 * chi_term_denom) +
                               (1.16854 * D1 * Ts[1])/(fdep * h) +
                               (16.1483 * D1 * Ts[1])/(fdep**2 * h**2 * chi_term_denom) -
                               (1.07425 * D1 * Ts[2])/(fdep * h) -
                               (5.62515 * D1 * Ts[2])/(fdep**2 * h**2 * chi_term_denom) +
                               (1.1361 * D1 * Ts[3])/(fdep * h) +
                               (2.87225 * D1 * Ts[3])/(fdep**2 * h**2 * chi_term_denom) -
                               (1.35336 * D1 * Ts[4])/(fdep * h) -
                               (1.82521 * D1 * Ts[4])/(fdep**2 * h**2 * chi_term_denom) +
                               (1.82683 * D1 * Ts[5])/(fdep * h) +
                               (1.35194 * D1 * Ts[5])/(fdep**2 * h**2 * chi_term_denom) -
                               (2.87424 * D1 * Ts[6])/(fdep * h) -
                               (1.13478 * D1 * Ts[6])/(fdep**2 * h**2 * chi_term_denom) +
                               (5.62786 * D1 * Ts[7])/(fdep * h) +
                               (1.0729 * D1 * Ts[7])/(fdep**2 * h**2 * chi_term_denom) -
                               (16.1529 * D1 * Ts[8])/(fdep * h) -
                               (1.16702 * D1 * Ts[8])/(fdep**2 * h**2 * chi_term_denom) +
                               (123.326 * D1 * Ts[9])/(fdep * h) +
                               (1.62791 * D1 * Ts[9])/(fdep**2 * h**2 * chi_term_denom) +
                               (123.321 * D1 * Tv[0])/(h - 1.0 * fdep * h) -
                               (16.1502 * D1 * Tv[1])/(h - 1.0 * fdep * h) +
                               (5.62579 * D1 * Tv[2])/(h - 1.0 * fdep * h) -
                               (2.87258 * D1 * Tv[3])/(h - 1.0 * fdep * h) +
                               (1.82542 * D1 * Tv[4])/(h - 1.0 * fdep * h) -
                               (1.3521 * D1 * Tv[5])/(h - 1.0 * fdep * h) +
                               (1.13491 * D1 * Tv[6])/(h - 1.0 * fdep * h) -
                               (1.07303 * D1 * Tv[7])/(h - 1.0 * fdep * h) +
                               (1.16715 * D1 * Tv[8])/(h - 1.0 * fdep * h) -
                               (1.6281 * D1 * Tv[9])/(h - 1.0 * fdep * h))
                
                surface_term = (-123.321 * Ts[0]/(fdep * h) +
                               16.1502 * Ts[1]/(fdep * h) -
                               5.62579 * Ts[2]/(fdep * h) +
                               2.87258 * Ts[3]/(fdep * h) -
                               1.82542 * Ts[4]/(fdep * h) +
                               1.3521 * Ts[5]/(fdep * h) -
                               1.13491 * Ts[6]/(fdep * h) +
                               1.07303 * Ts[7]/(fdep * h) -
                               1.16715 * Ts[8]/(fdep * h) +
                               1.6281 * Ts[9]/(fdep * h) -
                               (0.99866 * complex_flux)/(fdep * h * denom1))
                
                dydt = np.zeros(23)
                
                # ----- Ts (top layer) -----
                dydt[0] = (1/(fdep**2 * h**2) * D1 *
                          (-4857.68 * Ts[0] + 1398.55 * Ts[1] - 339.574 * Ts[2] + 155.38 * Ts[3] -
                           94.1732 * Ts[4] + 68.0747 * Ts[5] - 56.3546 * Ts[6] + 52.8474 * Ts[7] -
                           57.2106 * Ts[8] + 79.6158 * Ts[9] -
                           (1/denom1) * 48.8097 * complex_flux +
                           (1/chi_term_denom) * 3699.34 * surface_term)) - keff*Ts[0]
                
                dydt[1] = (1/(fdep**2 * h**2) * D1 *
                          (623.883 * Ts[0] - 666.927 * Ts[1] + 314.69 * Ts[2] - 79.6341 * Ts[3] +
                           38.4789 * Ts[4] - 24.8625 * Ts[5] + 19.3336 * Ts[6] - 17.4769 * Ts[7] +
                           18.5229 * Ts[8] - 25.5064 * Ts[9] +
                           (15.6003 * complex_flux)/denom1 -
                           (216.102/chi_term_denom) * surface_term)) - keff*Ts[1]
                
                dydt[2] = (1/(fdep**2 * h**2) * D1 *
                          (-103.349 * Ts[0] + 214.683 * Ts[1] - 290.794 * Ts[2] + 156.338 * Ts[3] -
                           40.8813 * Ts[4] + 20.5419 * Ts[5] - 14.0087 * Ts[6] + 11.7504 * Ts[7] -
                           11.934 * Ts[8] + 16.0906 * Ts[9] -
                           (9.79523 * complex_flux)/denom1 +
                           (51.3576/chi_term_denom) * surface_term)) - keff*Ts[2]
                
                dydt[3] = (1/(fdep**2 * h**2) * D1 *
                          (38.4943 * Ts[0] - 44.2193 * Ts[1] + 127.211 * Ts[2] - 188.669 * Ts[3] +
                           108.676 * Ts[4] - 29.5263 * Ts[5] + 15.5706 * Ts[6] - 11.4047 * Ts[7] +
                           10.7477 * Ts[8] - 13.9713 * Ts[9] +
                           (8.43649 * complex_flux)/denom1 -
                           (21.3451/chi_term_denom) * surface_term)) - keff*Ts[3]
                
                dydt[4] = (1/(fdep**2 * h**2) * D1 *
                          (-21.2652 * Ts[0] + 19.4771 * Ts[1] - 30.3147 * Ts[2] + 99.0212 * Ts[3] -
                           155.568 * Ts[4] + 94.3278 * Ts[5] - 26.904 * Ts[6] + 15.2359 * Ts[7] -
                           12.5905 * Ts[8] + 15.3766 * Ts[9] -
                           (9.15837 * complex_flux)/denom1 +
                           (12.362/chi_term_denom) * surface_term)) - keff*Ts[4]
                
                dydt[5] = (1/(fdep**2 * h**2) * D1 *
                          (15.3765 * Ts[0] - 12.5905 * Ts[1] + 15.2361 * Ts[2] - 26.9043 * Ts[3] +
                           94.3281 * Ts[4] - 155.568 * Ts[5] + 99.0218 * Ts[6] - 30.3155 * Ts[7] +
                           19.4782 * Ts[8] - 21.267 * Ts[9] +
                           (12.363 * complex_flux)/denom1 -
                           (9.15831/chi_term_denom) * surface_term)) - keff*Ts[5]
                
                dydt[6] = (1/(fdep**2 * h**2) * D1 *
                          (-13.9738 * Ts[0] + 10.7495 * Ts[1] - 11.4063 * Ts[2] + 15.5722 * Ts[3] -
                           29.5281 * Ts[4] + 108.678 * Ts[5] - 188.671 * Ts[6] + 127.214 * Ts[7] -
                           44.2228 * Ts[8] + 38.4997 * Ts[9] -
                           (21.3485 * complex_flux)/denom1 +
                           (8.43803/chi_term_denom) * surface_term)) - keff*Ts[6]
                
                dydt[7] = (1/(fdep**2 * h**2) * D1 *
                          (16.1085 * Ts[0] - 11.9465 * Ts[1] + 11.7612 * Ts[2] - 14.0191 * Ts[3] +
                           20.5525 * Ts[4] - 40.8927 * Ts[5] + 156.351 * Ts[6] - 290.809 * Ts[7] +
                           214.702 * Ts[8] - 103.378 * Ts[9] +
                           (51.3756 * complex_flux)/denom1 -
                           (9.80631/chi_term_denom) * surface_term)) - keff*Ts[7]
                
                dydt[8] = (1/(fdep**2 * h**2) * D1 *
                          (-25.5668 * Ts[0] + 18.5648 * Ts[1] - 17.5131 * Ts[2] + 19.3681 * Ts[3] -
                           24.8978 * Ts[4] + 38.5167 * Ts[5] - 79.6766 * Ts[6] + 314.74 * Ts[7] -
                           666.991 * Ts[8] + 623.98 * Ts[9] -
                           (216.163 * complex_flux)/denom1 +
                           (15.6376/chi_term_denom) * surface_term)) - keff*Ts[8]
                
                dydt[9] = (1/(fdep**2 * h**2) * D1 *
                          (79.7604 * Ts[0] - 57.3114 * Ts[1] + 52.935 * Ts[2] - 56.4392 * Ts[3] +
                           68.1623 * Ts[4] - 94.2691 * Ts[5] + 155.491 * Ts[6] - 339.708 * Ts[7] +
                           1398.72 * Ts[8] - 4857.95 * Ts[9] +
                           (3699.51 * complex_flux)/denom1 -
                           (48.899/chi_term_denom) * surface_term)) - keff*Ts[9]
                
                # ----- Tv (viable epidermis layer) -----
                dydt[10] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 - 4857.68 * Tv[0] + 1398.55 * Tv[1] - 339.574 * Tv[2] + 155.38 * Tv[3] -
                            94.1732 * Tv[4] + 68.0747 * Tv[5] - 56.3546 * Tv[6] + 52.8474 * Tv[7] -
                            57.2106 * Tv[8] + 79.6158 * Tv[9] +
                            (3699.34 * complex_flux)/denom1)) - keff*Tv[0]
                
                dydt[11] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 + 623.883 * Tv[0] - 666.927 * Tv[1] + 314.69 * Tv[2] - 79.6341 * Tv[3] +
                            38.4789 * Tv[4] - 24.8625 * Tv[5] + 19.3336 * Tv[6] - 17.4769 * Tv[7] +
                            18.5229 * Tv[8] - 25.5064 * Tv[9] -
                            (216.102 * complex_flux)/denom1)) - keff*Tv[1]
                
                dydt[12] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 - 103.349 * Tv[0] + 214.683 * Tv[1] - 290.794 * Tv[2] + 156.338 * Tv[3] -
                            40.8813 * Tv[4] + 20.5419 * Tv[5] - 14.0087 * Tv[6] + 11.7504 * Tv[7] -
                            11.934 * Tv[8] + 16.0906 * Tv[9] +
                            (51.3576 * complex_flux)/denom1)) - keff*Tv[2]
                
                dydt[13] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 + 38.4943 * Tv[0] - 44.2193 * Tv[1] + 127.211 * Tv[2] - 188.669 * Tv[3] +
                            108.676 * Tv[4] - 29.5263 * Tv[5] + 15.5706 * Tv[6] - 11.4047 * Tv[7] +
                            10.7477 * Tv[8] - 13.9713 * Tv[9] -
                            (21.3451 * complex_flux)/denom1)) - keff*Tv[3]
                
                dydt[14] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 - 21.2652 * Tv[0] + 19.4771 * Tv[1] - 30.3147 * Tv[2] + 99.0212 * Tv[3] -
                            155.568 * Tv[4] + 94.3278 * Tv[5] - 26.904 * Tv[6] + 15.2359 * Tv[7] -
                            12.5905 * Tv[8] + 15.3766 * Tv[9] +
                            (12.362 * complex_flux)/denom1)) - keff*Tv[4]
                
                dydt[15] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 + 15.3765 * Tv[0] - 12.5905 * Tv[1] + 15.2361 * Tv[2] - 26.9043 * Tv[3] +
                            94.3281 * Tv[4] - 155.568 * Tv[5] + 99.0218 * Tv[6] - 30.3155 * Tv[7] +
                            19.4782 * Tv[8] - 21.267 * Tv[9] -
                            (9.15831 * complex_flux)/denom1)) - keff*Tv[5]
                
                dydt[16] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 - 13.9738 * Tv[0] + 10.7495 * Tv[1] - 11.4063 * Tv[2] + 15.5722 * Tv[3] -
                            29.5281 * Tv[4] + 108.678 * Tv[5] - 188.671 * Tv[6] + 127.214 * Tv[7] -
                            44.2228 * Tv[8] + 38.4997 * Tv[9] +
                            (8.43803 * complex_flux)/denom1)) - keff*Tv[6]
                
                dydt[17] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 + 16.1085 * Tv[0] - 11.9465 * Tv[1] + 11.7612 * Tv[2] - 14.0191 * Tv[3] +
                            20.5525 * Tv[4] - 40.8927 * Tv[5] + 156.351 * Tv[6] - 290.809 * Tv[7] +
                            214.702 * Tv[8] - 103.378 * Tv[9] -
                            (9.80631 * complex_flux)/denom1)) - keff*Tv[7]
                
                dydt[18] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 - 25.5668 * Tv[0] + 18.5648 * Tv[1] - 17.5131 * Tv[2] + 19.3681 * Tv[3] -
                            24.8978 * Tv[4] + 38.5167 * Tv[5] - 79.6766 * Tv[6] + 314.74 * Tv[7] -
                            666.991 * Tv[8] + 623.98 * Tv[9] +
                            (15.6376 * complex_flux)/denom1)) - keff*Tv[8]
                
                dydt[19] = (1/(h - fdep * h)**2 * D1 *
                           (0.0 + 79.7604 * Tv[0] - 57.3114 * Tv[1] + 52.935 * Tv[2] - 56.4392 * Tv[3] +
                            68.1623 * Tv[4] - 94.2691 * Tv[5] + 155.491 * Tv[6] - 339.708 * Tv[7] +
                            1398.72 * Tv[8] - 4857.95 * Tv[9] -
                            (48.899 * complex_flux)/denom1)) - keff*Tv[9]
                
                # Interface cumulative terms
                dydt[20] = (-(1/(h - fdep * h)) * D1 *
                           (0.0 + 1.63008 * Tv[0] - 1.16854 * Tv[1] + 1.07425 * Tv[2] - 1.1361 * Tv[3] +
                            1.35336 * Tv[4] - 1.82683 * Tv[5] + 2.87424 * Tv[6] - 5.62786 * Tv[7] +
                            16.1529 * Tv[8] - 123.326 * Tv[9] -
                            (0.999885 * complex_flux)/denom1))
                
                dydt[21] = ((1/(fdep * h)) * D1 *
                           (123.321 * Ts[0] - 16.1502 * Ts[1] + 5.62579 * Ts[2] - 2.87258 * Ts[3] +
                            1.82542 * Ts[4] - 1.3521 * Ts[5] + 1.13491 * Ts[6] - 1.07303 * Ts[7] +
                            1.16715 * Ts[8] - 1.6281 * Ts[9] +
                            (0.99866 * complex_flux)/denom1 -
                            (110.997/chi_term_denom) * surface_term))
                
                # Bulk reaction integral
                W_interior = np.array([0.0333356721543441, 0.0747256745752903, 0.109543181257991,
                                      0.134633359654998, 0.147762112357376, 0.147762112357376,
                                      0.134633359654998, 0.109543181257991, 0.0747256745752903,
                                      0.0333356721543441])
                
                Ts_clipped = np.clip(Ts, 0.0, None)
                Tv_clipped = np.clip(Tv, 0.0, None)
                
                int_Ts = np.dot(W_interior, Ts_clipped)
                int_Tv = np.dot(W_interior, Tv_clipped)
                
                dydt[22] = keff * ((int_Ts * (fdep*h) + int_Tv * (h - fdep*h)))
                
                return dydt
            
            # Initial conditions for below saturation
            y0_b = np.zeros(23)
            y0_b[0:10] = Tso    # Ts IC
            y0_b[10:20] = 0.0   # Tv IC
            # Qt, Qevap, Qreact already 0
            
            t_span_b = (0, tf)
            t_eval_b = np.linspace(0, tf, 2000)
            
            solb = solve_ivp(below_rhs, t_span_b, y0_b, t_eval=t_eval_b,
                           method="BDF", rtol=1e-9, atol=1e-11)
            
            if not solb.success:
                raise RuntimeError(f"Below-saturation integration failed: {solb.message}")
            
            t_combined = solb.t
            Qt_combined = solb.y[20]      # Qabs
            Qst_combined = solb.y[21]     # Total top surface loss (includes chi)
            Qreact_combined = solb.y[22]  # Qreact
        
        # =============================================================
        # POST-PROCESSING (EXACT from comprehensive model)
        # =============================================================
        Qabs = Qt_combined
        Qlosstop = Qst_combined  # Total top surface loss
        Qreact = Qreact_combined
        Qremain = Mo - (Qabs + Qlosstop + Qreact)
        Qremain = np.maximum(Qremain, 0.0)
        
        # Decompose Qlosstop using chi ratio (evaporation vs surface reaction)
        # CRITICAL: Original code uses hsc in decomposition, NOT the L used in ODEs
        # This creates the observed 75/25 evap/react split
        Qlosstop_evap = np.zeros_like(Qlosstop)
        Qlosstop_react = np.zeros_like(Qlosstop)
        
        # DIAGNOSTIC: Print decomposition calculation details at final time
        print(f"\n{'='*60}")
        print(f"DECOMPOSITION DIAGNOSTIC (at t={t_combined[-1]/3600:.2f} h)")
        print(f"{'='*60}")
        print(f"k = {k:.6e} s⁻¹")
        print(f"L (air boundary layer, for kg only) = {L:.6e} cm")
        print(f"hsc (stratum corneum thickness, used in chi) = {hsc:.6e} cm")
        print(f"Ksc = {Ksc:.6f}")
        print(f"Dsc = {Dsc:.6e} cm²/s")
        print(f"K (evap coeff) = {K:.6f}")
        print(f"td = {td:.2f} s ({td/3600:.2f} h)")
        
        for i, t in enumerate(t_combined):
            on_val = 1.0 if (t >= td) else 0.0
            ks_eff_t = (k * hsc * Ksc) * on_val  # ks = k * hsc * Ksc (matches notebook)
            K_term = K  # Evaporation is always active (matches Jupyter notebook)
            ks_term = (hsc / Dsc) * ks_eff_t  # chi term calculation (matches notebook)
            chi_t = K_term + ks_term
            
            if chi_t > 0:
                evap_frac = K_term / chi_t
                react_frac = ks_term / chi_t
                Qlosstop_evap[i] = evap_frac * Qlosstop[i]
                Qlosstop_react[i] = react_frac * Qlosstop[i]
            else:
                # This should never happen since K_term is always > 0
                Qlosstop_evap[i] = Qlosstop[i]  # All evaporation if no reaction
                Qlosstop_react[i] = 0.0
                
            # Print details at final time
            if i == len(t_combined) - 1:
                print(f"\nAt final time (t={t/3600:.2f} h):")
                print(f"  on_val = {on_val}")
                print(f"  ks_eff_t = k * hsc * Ksc * on_val = {ks_eff_t:.6e}")
                print(f"  K_term = {K_term:.6f}")
                print(f"  ks_term = (hsc/Dsc) * ks_eff_t = {ks_term:.6f}")
                print(f"  chi_t = K_term + ks_term = {chi_t:.6f}")
                print(f"  evap_frac = K_term / chi_t = {evap_frac:.6f}")
                print(f"  react_frac = ks_term / chi_t = {react_frac:.6f}")
                print(f"  Qlosstop[final] = {Qlosstop[i]:.6e} mg/cm²")
                print(f"  Qlosstop_evap[final] = {Qlosstop_evap[i]:.6e} mg/cm²")
                print(f"  Qlosstop_react[final] = {Qlosstop_react[i]:.6e} mg/cm²")
        print(f"{'='*60}\n")
        
        t_hours = t_combined / 3600.0
        
        # Create plot (matching comprehensive model style)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_hours, Qabs, label="Qabs (absorbed into body)", linewidth=2)
        ax.plot(t_hours, Qlosstop_evap, label="Qlosstop_evap (evaporation)", linestyle='--', linewidth=2)
        ax.plot(t_hours, Qlosstop_react, label="Qlosstop_react (surface rxn)", linestyle='--', linewidth=2)
        ax.plot(t_hours, Qreact, label="Qreact (bulk reaction)", linewidth=2)
        ax.plot(t_hours, Qremain, label="Qremain (on surface)", linewidth=2)
        
        # Initial dose line
        ax.axhline(Mo, color='black', linestyle=':', linewidth=1.5, label='Initial Dose (Mo)')
        
        # RSDL activation line
        if td_hours > 0:
            ax.axvline(td_hours, color='red', linestyle='--', linewidth=2,
                      label=f'RSDL activation (td={td_hours:.1f}h)', alpha=0.7)
        
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Mass (mg/cm²)", fontsize=12)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        # Right axis as % of Mo
        to_pct = lambda y: 100.0 * y / Mo
        from_pct = lambda p: (p / 100.0) * Mo
        ax2 = ax.secondary_yaxis('right', functions=(to_pct, from_pct))
        ax2.set_ylabel("Percent of initial dose (% of Mo)", fontsize=12)
        ax2.set_ylim(to_pct(ax.get_ylim()[0]), to_pct(ax.get_ylim()[1]))
        ax2.set_yticks([0, 25, 50, 75, 100])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=True)
        plt.subplots_adjust(bottom=0.25)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Calculate final values and statistics
        Qabs_final = Qabs[-1]
        Qlosstop_final = Qlosstop[-1]
        Qlosstop_evap_final = Qlosstop_evap[-1]
        Qlosstop_react_final = Qlosstop_react[-1]
        Qreact_final = Qreact[-1]
        Qremain_final = Qremain[-1]
        
        mass_balance_total = Qabs_final + Qlosstop_final + Qreact_final + Qremain_final
        evap_frac = Qlosstop_evap_final / Qlosstop_final if Qlosstop_final > 0 else 0
        react_frac = Qlosstop_react_final / Qlosstop_final if Qlosstop_final > 0 else 0
        top_loss_frac = Qlosstop_final / (Qlosstop_final + Qreact_final) if (Qlosstop_final + Qreact_final) > 0 else 0
        bulk_loss_frac = Qreact_final / (Qlosstop_final + Qreact_final) if (Qlosstop_final + Qreact_final) > 0 else 0
        
        print(f"✓ Comparison plot generated successfully for {chemical_name}")
        print(f"  Image size: {len(img_base64)} bytes (base64)")
        print(f"  Final values: Qabs={Qabs_final:.3e}, Qremain={Qremain_final:.3e}")
        
        return {
            'comparison_plot': img_base64,
            'status': 'success',
            # Also return raw data for multi-agent plotting
            'data': {
                't_hr': t_hours,
                'Qabs': Qabs,
                'Qlosstop_evap': Qlosstop_evap,
                'Qlosstop_react': Qlosstop_react,
                'Qreact': Qreact,
                'Qremain': Qremain,
                'Mo': Mo
            },
            # Summary statistics
            'summary': {
                'Qabs_final': Qabs_final,
                'Qlosstop_final': Qlosstop_final,
                'Qlosstop_evap_final': Qlosstop_evap_final,
                'Qlosstop_react_final': Qlosstop_react_final,
                'Qreact_final': Qreact_final,
                'Qremain_final': Qremain_final,
                'mass_balance_total': mass_balance_total,
                'evap_frac': evap_frac,
                'react_frac': react_frac,
                'top_loss_frac': top_loss_frac,
                'bulk_loss_frac': bulk_loss_frac,
                'Mo': Mo,
                'sim_hours': sim_hours,
                'td_hours': td_hours,
                'chemical_name': chemical_name
            }
        }
        
    except Exception as e:
        import traceback
        print(f"✗ Error in comparison simulation for {chemical_name}: {e}")
        print(traceback.format_exc())
        return {
            'comparison_plot': None,
            'status': f'error: {e}'
        }


def run_comparison_simulation_multi(agents_to_run, dose, sim_hours, td_hours, custom_properties=None):
    """
    Run dermal absorption simulation WITH decontamination for MULTIPLE agents.
    Each agent gets its own simulation, then all results are overlaid on one plot.
    
    Parameters:
    -----------
    agents_to_run : list of tuples
        List of (name, MW, _, logKow, ...) tuples from agent_properties
    dose : float
        Initial dose (mg/cm²)
    sim_hours : float
        Total simulation time (hours)
    td_hours : float
        Decontamination activation time (hours)
    custom_properties : dict
        Optional custom property overrides {agent_name: {prop: value}}
        
    Returns:
    --------
    dict with 'comparison_plot' (base64 PNG) and 'status'
    """
    from app import get_agent_data, pubchem_lookup, safe_pubchem_prop
    
    if custom_properties is None:
        custom_properties = {}
    
    try:
        # Create combined figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        cmap = plt.get_cmap('tab10')
        
        successful_agents = 0
        all_agent_results = []
        
        # Run simulation for each agent
        for i, agent in enumerate(agents_to_run):
            try:
                name = str(agent[0])
                MW = float(agent[1])
                logKow = float(agent[3])
                
                # Get properties
                props = get_agent_data(name)
                
                # Use custom properties if provided
                if name in custom_properties and 'Sw' in custom_properties[name]:
                    Sw = float(custom_properties[name]['Sw']) / 1000.0
                else:
                    Sw = props.get('Sw')
                    if Sw is None:
                        print(f"Warning: Sw not found for {name}, skipping")
                        continue
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
                        print(f"Warning: Pvap not found for {name}, skipping")
                        continue
                    else:
                        Pvap = float(Pvap)
                
                print(f"\n{'='*60}")
                print(f"Multi-agent comparison - Agent {i+1}/{len(agents_to_run)}: {name}")
                print(f"MW={MW}, logKow={logKow}, Pvap={Pvap}, Sw={Sw}")
                
                # Get SMILES and formula for k calculation
                SMILES = props.get('SMILES', None)
                formula = props.get('formula', None)
                
                # Run single-agent simulation (which now returns data too)
                result = run_comparison_simulation(
                    MW=MW, logKow=logKow, Pvap=Pvap, Sw=Sw, Mo=dose,
                    chemical_name=name, sim_hours=sim_hours, td_hours=td_hours,
                    SMILES=SMILES, formula=formula
                )
                
                if result is None or result.get('status') != 'success' or 'data' not in result:
                    print(f"✗ Failed to get data for {name}")
                    continue
                
                # Extract time series from result
                data = result['data']
                t_hr = data['t_hr']
                Qabs = data['Qabs']
                Qlosstop_evap = data['Qlosstop_evap']
                Qlosstop_react = data['Qlosstop_react']
                Qreact = data['Qreact']
                Qremain = data['Qremain']
                
                all_agent_results.append((name, data))
                
                # Plot with unique color per agent
                color = cmap(i % cmap.N)
                
                # Use distinct line styles for different quantities
                ax.plot(t_hr, Qabs, color=color, linestyle='-', linewidth=2.5, 
                       label=f'{name} — Absorbed' if len(agents_to_run) > 1 else 'Absorbed')
                ax.plot(t_hr, Qlosstop_evap, color=color, linestyle='--', linewidth=2.0, 
                       label=f'{name} — Evaporation' if len(agents_to_run) > 1 else 'Evaporation')
                ax.plot(t_hr, Qlosstop_react, color=color, linestyle='-.', linewidth=2.0, 
                       label=f'{name} — Surface Rxn' if len(agents_to_run) > 1 else 'Surface Rxn')
                ax.plot(t_hr, Qreact, color=color, linestyle=':', linewidth=2.0, 
                       label=f'{name} — Bulk Rxn' if len(agents_to_run) > 1 else 'Bulk Rxn')
                ax.plot(t_hr, Qremain, color=color, linestyle='-', linewidth=1.5, alpha=0.5,
                       label=f'{name} — Remaining' if len(agents_to_run) > 1 else 'Remaining')
                
                successful_agents += 1
                print(f"✓ Successfully plotted {name}")
                
            except Exception as e:
                print(f"✗ Error processing {agent[0]}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if successful_agents == 0:
            print("✗ No agents were successfully plotted!")
            plt.close(fig)
            return {
                'comparison_plot': None,
                'status': 'error: No agents could be plotted'
            }
        
        # Add RSDL activation line
        if td_hours > 0:
            ax.axvline(x=td_hours, color='red', linestyle='--', linewidth=2.0, alpha=0.7, 
                      label=f'RSDL activated (t={td_hours:.1f}h)', zorder=10)
        
        # Add initial dose reference line
        ax.axhline(y=dose, color='black', linestyle=':', linewidth=1.5, alpha=0.5, 
                  label=f'Initial dose (Mo={dose:.2e})', zorder=5)
        
        # Format plot
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Mass (mg/cm²)", fontsize=12)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        # Add title
        if len(agents_to_run) == 1:
            title = f"Dermal Absorption with Decontamination: {agents_to_run[0][0]}"
        else:
            title = f"Multi-Agent Comparison ({successful_agents} agents)"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # Right axis as % of Mo
        to_pct = lambda y: 100.0 * y / dose
        from_pct = lambda p: (p / 100.0) * dose
        ax2 = ax.secondary_yaxis('right', functions=(to_pct, from_pct))
        ax2.set_ylabel("Percent of initial dose (% of Mo)", fontsize=12)
        ax2.set_ylim(to_pct(ax.get_ylim()[0]), to_pct(ax.get_ylim()[1]))
        ax2.set_yticks([0, 25, 50, 75, 100])
        
        # Legend - place at bottom with better formatting
        if len(agents_to_run) == 1:
            # For single agent, simple legend outside plot area
            ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=10, 
                     frameon=True, shadow=True, borderpad=1)
        else:
            # For multiple agents, compact legend at bottom
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=9, 
                     frameon=True, ncol=min(3, successful_agents))
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        if len(agents_to_run) > 1:
            plt.subplots_adjust(bottom=0.25)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        print(f"\n✓ Multi-agent comparison plot generated for {successful_agents}/{len(agents_to_run)} agent(s)")
        print(f"  Image size: {len(img_base64)} bytes (base64)")
        
        # Collect summaries from all agents
        agent_summaries = []
        for agent_name, agent_data in all_agent_results:
            agent_summaries.append({
                'chemical_name': agent_name,
                'data': agent_data
            })
        
        return {
            'comparison_plot': img_base64,
            'status': 'success',
            'agent_summaries': agent_summaries,
            'sim_hours': sim_hours,
            'td_hours': td_hours
        }
        
    except Exception as e:
        import traceback
        print(f"✗ Error in multi-agent comparison simulation: {e}")
        traceback.print_exc()
        return {
            'comparison_plot': None,
            'status': f'error: {e}'
        }
