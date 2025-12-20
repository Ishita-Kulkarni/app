#!/usr/bin/env python3
"""
Diagnostic script to compare K calculations between original and comparison_model.py
"""

import numpy as np

# Chemical properties for 1,3-Bis(2-chloroethylthio)-n-propane
MW = 233.2
logKow = 3.48
Pvap = 0.00101  # torr
Sw_mgL = 0.03  # mg/L
Sw = Sw_mgL / 1000.0  # mg/cm³

# Estimate atom counts
nc = max(1, int(MW / 14))
nh = max(1, int(MW / 7))
no = 1
nn = 0
nring = 0

# Fixed parameters
u = 16.5
R = 62.37
T = 298.15

# Calculate logPscw and kp
logPscw = -2.8 + 0.66*logKow - 0.0056*MW
Pscw = 10.0**logPscw
kp = Pscw

print("="*60)
print("K CALCULATION COMPARISON")
print("="*60)
print(f"Chemical: 1,3-Bis(2-chloroethylthio)-n-propane")
print(f"MW = {MW} g/mol")
print(f"logKow = {logKow}")
print(f"Pvap = {Pvap} torr")
print(f"Sw = {Sw:.6f} mg/cm³")
print(f"kp = {kp:.6e}")
print()

# --- ORIGINAL CODE METHOD ---
print("ORIGINAL CODE (using L=13.4 cm):")
print("-"*60)
L_orig = 13.4  # cm
Vp_orig = Pvap * 133.322  # torr -> Pa
S_orig = 16.5*nc + 1.98*nh + 5.69*nn + 5.48*no - 20.42*nring
Dg_orig = (10**(-3) * T**1.75 * (1/29 + 1/MW)**0.5) / ((S_orig**(1/3) + 20.1**(1/3))**2)
kg_orig = (3260/3600) * Dg_orig**(2/3) * np.sqrt(u/L_orig)
K_orig = (kg_orig * Pvap * MW) / (R * T) * 1 / (kp * Sw)

print(f"L = {L_orig} cm")
print(f"Vp = {Vp_orig:.6f} Pa")
print(f"S = {S_orig:.6f}")
print(f"Dg = {Dg_orig:.6e} cm²/s")
print(f"kg = {kg_orig:.6f} cm/s")
print(f"K = {K_orig:.6f}")
print()

# --- COMPARISON_MODEL.PY METHOD ---
print("COMPARISON_MODEL.PY (using L_air=13.4 cm):")
print("-"*60)
L_air = 13.4  # cm - air boundary layer
Vp_new = Pvap * 133.322  # torr -> Pa
S_new = 16.5*nc + 1.98*nh + 5.69*nn + 5.48*no - 20.42*nring
Dg_new = (10**(-3) * T**1.75 * (1/29 + 1/MW)**0.5) / ((S_new**(1/3) + 20.1**(1/3))**2)
kg_new = (3260/3600) * Dg_new**(2/3) * np.sqrt(u/L_air)
K_new = (kg_new * Pvap * MW) / (R * T) * 1 / (kp * Sw)

print(f"L_air = {L_air} cm")
print(f"Vp = {Vp_new:.6f} Pa")
print(f"S = {S_new:.6f}")
print(f"Dg = {Dg_new:.6e} cm²/s")
print(f"kg = {kg_new:.6f} cm/s")
print(f"K = {K_new:.6f}")
print()

# --- COMPARISON ---
print("="*60)
print("COMPARISON:")
print("="*60)
print(f"K_original = {K_orig:.6f}")
print(f"K_new      = {K_new:.6f}")
print(f"Difference = {K_new - K_orig:.6f}")
print(f"Ratio      = {K_new / K_orig:.6f}")
print()

# What K do we need for 75/25 split?
k = 1.667e-06  # s⁻¹
L_sc = 1.34e-03  # cm
Kscw = 0.040 * (10**logKow)**0.81 + 4.06 * (10**logKow)**0.27 + 0.359
Ksc = Kscw
Dsc = (Pscw * L_sc / Kscw) / 3600.0

ks_term = 1.994  # from your output

K_needed_for_75_25 = ks_term * 3.0  # ratio of 75/25 = 3.0

print("FOR 75/25 SPLIT:")
print("-"*60)
print(f"ks_term = {ks_term:.6f}")
print(f"K needed for 75/25 = {K_needed_for_75_25:.6f}")
print(f"Current K_new = {K_new:.6f}")
print(f"Shortfall = {K_needed_for_75_25 - K_new:.6f}")
print()

# The issue must be in the original code's different calculation
# Let's check if the original uses a different formula
print("HYPOTHESIS: Original code uses different K formula or parameters")
print("-"*60)
print("Checking if original might have used Vp instead of Pvap in K formula...")
K_with_Vp_error = (kg_orig * Vp_orig * MW) / (R * T) * 1 / (kp * Sw)
print(f"K (if using Vp instead of Pvap) = {K_with_Vp_error:.6f}")
print(f"This would be {K_with_Vp_error/K_orig:.1f}× larger")
print()

print("="*60)
