"""
Decontamination simulation - WITH RSDL
Uses same ODE system as original model, but adds decontamination reactions
"""
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy.integrate import solve_ivp


def run_comparison_simulation(MW, logKow, Pvap, Sw, Mo=1.0e-3, chemical_name="Chemical",
                                sim_hours=25.0, td_hours=5.0, nc=0, nh=0, no=0, nn=0, nring=0,
                                formula=None, smiles=None):
    """
    Generate plot showing WITH decontamination (RSDL)
    Uses the SAME ODE solver as original model + decontamination reactions
    """
    
    # SAME physics calculations as original model
    Kow = 10.0**logKow
    Kscw = 0.040 * Kow**0.81 + 4.06 * Kow**0.27 + 0.359
    
    logPscw = -2.8 + 0.66*logKow - 0.0056*MW
    Pscw = 10.0**logPscw
    kp = Pscw
    
    # Diffusivity
    hsc = 13.4 * 1e-4  # cm
    h1 = hsc
    fdep = 0.1
    Dsc = (Pscw * h1 / Kscw) / 3600.0
    D1 = Dsc
    
    # Saturation
    Csat = Kscw * Sw
    Msat = fdep * hsc * Csat
    
    # Evaporation coefficient
    R = 62.37
    T = 298.15
    kg = 0.000001
    
    try:
        Dg = 0.001 * T**1.75 * np.sqrt((1/MW) + (1/29)) / (101325 * (0.0001*((20.1)**(1/3) + (20.1)**(1/3)))**2)
    except:
        Dg = 0.05
    
    try:
        logHlc_calc = 0.67 * logKow - np.log10(Sw) + 2.57
        Hlc = 10**logHlc_calc
    except:
        Hlc = 1.0
    
    chi = kg * Hlc
    K_evap = (kg * Pvap * MW) / (R * T) * 1 / (kp * Sw) if (kp * Sw) > 0 else 0.1
    
    # Decontamination rate constants
    k_react_surface = 0.008  # /hour - surface reaction
    k_react_bulk = 0.003     # /hour - bulk reaction
    
    tf = sim_hours * 3600.0  # Convert to seconds
    
    # BELOW SATURATION MODEL (like original)
    h = h1
    Tso = Mo / (fdep * h)
    
    def ode_system_with_decon(t, y):
        """Same ODE as original, but tracks decontamination components"""
        Ts = y[0:10]
        Tv = y[10:20]
        Qt = y[20]        # absorbed
        Qet = y[21]       # evaporated
        Qsr = y[22]       # surface reaction (decon)
        Qbr = y[23]       # bulk reaction (decon)
        
        # Calculate current time in hours for decontamination activation
        t_hours = t / 3600.0
        
        # Same complex flux calculation as original
        denom1 = ((111.0 * D1) / (fdep * h) + (110.997 * D1) / (h - 1.0 * fdep * h) + 
                  (0.998546 * D1) / (fdep ** 2 * h ** 2 * (-(110.997 / (fdep * h)) - (1.0 * chi) / h)))
        chi_term_denom = -(110.997 / (fdep * h)) - (1.0 * chi) / h
        complex_flux = (0.0 - (1.63008 * D1 * Ts[0]) / (fdep * h) - (123.307 * D1 * Ts[0]) / (fdep ** 2 * h ** 2 * chi_term_denom) + 
                       (1.16854 * D1 * Ts[1]) / (fdep * h) + (16.1483 * D1 * Ts[1]) / (fdep ** 2 * h ** 2 * chi_term_denom) - 
                       (1.07425 * D1 * Ts[2]) / (fdep * h) - (5.62515 * D1 * Ts[2]) / (fdep ** 2 * h ** 2 * chi_term_denom) + 
                       (1.1361 * D1 * Ts[3]) / (fdep * h) + (2.87225 * D1 * Ts[3]) / (fdep ** 2 * h ** 2 * chi_term_denom) - 
                       (1.35336 * D1 * Ts[4]) / (fdep * h) - (1.82521 * D1 * Ts[4]) / (fdep ** 2 * h ** 2 * chi_term_denom) + 
                       (1.82683 * D1 * Ts[5]) / (fdep * h) + (1.35194 * D1 * Ts[5]) / (fdep ** 2 * h ** 2 * chi_term_denom) - 
                       (2.87424 * D1 * Ts[6]) / (fdep * h) - (1.13478 * D1 * Ts[6]) / (fdep ** 2 * h ** 2 * chi_term_denom) + 
                       (5.62786 * D1 * Ts[7]) / (fdep * h) + (1.0729 * D1 * Ts[7]) / (fdep ** 2 * h ** 2 * chi_term_denom) - 
                       (16.1529 * D1 * Ts[8]) / (fdep * h) - (1.16702 * D1 * Ts[8]) / (fdep ** 2 * h ** 2 * chi_term_denom) + 
                       (123.326 * D1 * Ts[9]) / (fdep * h) + (1.62791 * D1 * Ts[9]) / (fdep ** 2 * h ** 2 * chi_term_denom) + 
                       (123.321 * D1 * Tv[0]) / (h - 1.0 * fdep * h) - (16.1502 * D1 * Tv[1]) / (h - 1.0 * fdep * h) + 
                       (5.62579 * D1 * Tv[2]) / (h - 1.0 * fdep * h) - (2.87258 * D1 * Tv[3]) / (h - 1.0 * fdep * h) + 
                       (1.82542 * D1 * Tv[4]) / (h - 1.0 * fdep * h) - (1.3521 * D1 * Tv[5]) / (h - 1.0 * fdep * h) + 
                       (1.13491 * D1 * Tv[6]) / (h - 1.0 * fdep * h) - (1.07303 * D1 * Tv[7]) / (h - 1.0 * fdep * h) + 
                       (1.16715 * D1 * Tv[8]) / (h - 1.0 * fdep * h) - (1.6281 * D1 * Tv[9]) / (h - 1.0 * fdep * h))
        
        surface_term = (-123.321 * Ts[0] / (fdep * h) + 16.1502 * Ts[1] / (fdep * h) - 
                       5.62579 * Ts[2] / (fdep * h) + 2.87258 * Ts[3] / (fdep * h) - 
                       1.82542 * Ts[4] / (fdep * h) + 1.3521 * Ts[5] / (fdep * h) - 
                       1.13491 * Ts[6] / (fdep * h) + 1.07303 * Ts[7] / (fdep * h) - 
                       1.16715 * Ts[8] / (fdep * h) + 1.6281 * Ts[9] / (fdep * h) - 
                       (0.99866 * complex_flux) / (fdep * h * denom1))
        
        dydt = np.zeros(24)  # Extended to include Qsr and Qbr
        
        # All Ts derivatives (same as original)
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
        
        # All Tv derivatives (same as original)
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
        
        # Qt (absorption) - same as original
        dydt[20] = (-(1 / (h - fdep * h)) * D1 * (0.0 + 1.63008 * Tv[0] - 1.16854 * Tv[1] + 1.07425 * Tv[2] - 1.1361 * Tv[3] + 1.35336 * Tv[4] - 1.82683 * Tv[5] + 2.87424 * Tv[6] - 5.62786 * Tv[7] + 16.1529 * Tv[8] - 123.326 * Tv[9] - (0.999885 * complex_flux) / denom1))
        
        # Qet (evaporation) - same as original
        dydt[21] = ((1 / (fdep * h)) * D1 * (123.321 * Ts[0] - 16.1502 * Ts[1] + 5.62579 * Ts[2] - 2.87258 * Ts[3] + 1.82542 * Ts[4] - 1.3521 * Ts[5] + 1.13491 * Ts[6] - 1.07303 * Ts[7] + 1.16715 * Ts[8] - 1.6281 * Ts[9] + (0.99866 * complex_flux) / denom1 - (110.997 / chi_term_denom) * surface_term))
        
        # NEW: Decontamination reactions (activated after td_hours)
        if t_hours >= td_hours:
            # Calculate mass remaining on surface
            Qremain = Mo - (Qt + Qet + Qsr + Qbr)
            Qremain = max(0, Qremain)
            
            # Surface reaction (per hour)
            dydt[22] = k_react_surface * Qremain / 3600.0  # Convert to per second
            
            # Bulk reaction (per hour)
            dydt[23] = k_react_bulk * Qremain / 3600.0  # Convert to per second
        else:
            dydt[22] = 0.0
            dydt[23] = 0.0
        
        return dydt
    
    # Initial conditions
    y0 = np.zeros(24)
    y0[0:10] = Tso  # Ts
    y0[10:20] = 0.0  # Tv
    y0[20] = 0.0  # Qt (absorbed)
    y0[21] = 0.0  # Qet (evaporated)
    y0[22] = 0.0  # Qsr (surface reaction)
    y0[23] = 0.0  # Qbr (bulk reaction)
    
    # Solve ODE
    t_span = (0, tf)
    t_eval = np.linspace(0, tf, 1000)
    
    sol = solve_ivp(ode_system_with_decon, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        return {
            'comparison_plot': None,
            'status': 'failed',
            'error': sol.message
        }
    
    # Extract solutions
    t_hours = sol.t / 3600.0
    Qabs = sol.y[20]
    Qlosstop_evap = sol.y[21]
    Qlosstop_react = sol.y[22]
    Qreact = sol.y[23]
    Qremain = Mo - (Qabs + Qlosstop_evap + Qlosstop_react + Qreact)
    Qremain = np.maximum(0, Qremain)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(t_hours, Qabs, label="Qabs (absorbed into body)", linewidth=2, color='tab:blue')
    ax.plot(t_hours, Qlosstop_evap, label="Qlosstop_evap (evaporation)", linestyle='--', linewidth=2, color='tab:orange')
    ax.plot(t_hours, Qlosstop_react, label="Qlosstop_react (surface rxn)", linestyle='--', linewidth=2, color='tab:green')
    ax.plot(t_hours, Qreact, label="Qreact (bulk reaction)", linewidth=2, color='tab:red')
    ax.plot(t_hours, Qremain, label="Qremain (on surface)", linewidth=2, color='tab:purple')
    ax.axhline(Mo, color='black', linestyle=':', linewidth=1.5, label='Initial Dose (Mo)')
    
    if td_hours > 0:
        ax.axvline(td_hours, color='red', linestyle='--', linewidth=2,
                   label=f'RSDL activation (td={td_hours:.1f}h)', alpha=0.7)
    
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Mass (mg/cm²)", fontsize=12)
    ax.set_xlim(0, sim_hours)
    ax.set_ylim(bottom=0, top=Mo*1.15)
    ax.grid(True, alpha=0.3)
    
    regime = "Above Saturation (Two-Phase)" if Mo > Msat else "Below Saturation (Single-Phase)"
    ax.set_title(f"Dermal Absorption: {chemical_name}\n{regime}", fontsize=12)
    
    # Right axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Percent of initial dose (% of Mo)", fontsize=12)
    ax2.set_ylim(0, 115)
    ax2.set_yticks([0, 25, 50, 75, 100])
    
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=9)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return {
        'comparison_plot': plot_b64,
        'status': 'success'
    }
