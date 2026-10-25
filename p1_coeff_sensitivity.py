# -*- coding: utf-8 -*-
"""
P1 supplementary ablation: coefficient-sensitivity of the g=1->g=2 upgrade
marginal reward (Root-cause-I structural-misalignment check).

Does NOT modify uav_semantic_multi_seed_ablation.py. Pure analytic script.

Scope (matches the manuscript's Root-cause-I derivation):
  The diagnosed misalignment is the *utility-vs-bit-cost* trade-off under the
  ORIGINAL (uncorrected) reward (9) WITHOUT the recall-aware term (correction A),
  i.e.  R = sum_i s_i*U(g_i,SNR) - lambda1*bits - lambda2*latency - lambda_energy*energy.
  We compute the per-region per-step NET marginal reward of upgrading ONE
  high-priority region from g=1 to g=2 at (s_i, SNR) = (0.875, 10 dB):

      net  =  DeltaU  -  lambda1*DeltaBits  -  lambda2*DeltaLatency
              -  lambda_energy*DeltaCommEnergy

  - DeltaU depends on base(g)  (via U(g,SNR)=base(g)*sigmoid((SNR-5)/4)).
  - DeltaBits = 16 (8->24) is INDEPENDENT of every semantic coefficient.
  - DeltaLatency, DeltaCommEnergy depend only on bits & bandwidth, also
    independent of base(g)/base_r(g)  -> a fixed offset across all perturbations.

  Therefore perturbing the hand-set coefficients can only rescale DeltaU, never
  the bit/latency/energy penalties. The misalignment is structural iff the
  uncorrected net stays negative across reasonable perturbations.

Reference operating point (paper Root-cause-I): s_i=0.875, SNR=10 dB.
lambda1 in {0.015 (uncorrected), 0.008 (corrected)}.
lambda2=0.020, lambda_energy=0.008, lambda_R=0.40 (correction A, reported separately).
Representative bandwidth avail_bw = mean of U(0.2,1.0) = 0.6 (only affects the
fixed latency/energy offset, not the perturbation comparison).
"""

import numpy as np

# ---- nominal coefficients (from authoritative script) ----
BASE_G_NOM = {1: 0.40, 2: 0.70, 3: 0.95}   # base(g) for utility
BASE_R_NOM = {2: 0.70, 3: 0.92}             # base_r(g) for P_recall
LAMBDA1_UNCORR = 0.015
LAMBDA1_CORR = 0.008
LAMBDA2 = 0.020
LAMBDA_ENERGY = 0.008
LAMBDA_R = 0.40

# ---- operating point ----
S_I = 0.875
SNR = 10.0
DELTA_BITS = 24 - 8          # g=1->g=2
BW_REP = 0.6                 # representative avail_bw

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def utility(base_g, g, snr):
    return base_g[g] * sigmoid((snr - 5.0) / 4.0)

def d_util(base_g, snr, s_i):
    """Marginal utility of upgrading high-prio region g=1->g=2, s_i-weighted."""
    return s_i * (utility(base_g, 2, snr) - utility(base_g, 1, snr))

def d_latency_penalty():
    # latency = total_bits/(bw*80); marginal for +16 bits = 16/(bw*80)
    return LAMBDA2 * (DELTA_BITS / (BW_REP * 80.0))

def d_energy_penalty():
    # comm_energy = total_bits*0.02; marginal = 16*0.02 (flight unchanged)
    return LAMBDA_ENERGY * (DELTA_BITS * 0.02)

def d_recall_bonus(base_r):
    # P_recall(2,10)=base_r(2)*clip(10/20)=base_r(2)*0.5 ; P_recall(1,*)=0
    return LAMBDA_R * (base_r[2] * np.clip(SNR / 20.0, 0, 1))

# fixed offsets (independent of semantic coefficients)
LAT_PEN = d_latency_penalty()
ENR_PEN = d_energy_penalty()
print(f"Representative fixed offsets: dLatency penalty = {LAT_PEN:.5f}, "
      f"dCommEnergy penalty = {ENR_PEN:.5f}, total = {LAT_PEN+ENR_PEN:.5f}")
print(f"sigmoid((10-5)/4) = {sigmoid((SNR-5.0)/4.0):.5f}\n")

def net_marginal(base_g, snr, s_i, lambda1):
    du = d_util(base_g, snr, s_i)
    bit_pen = lambda1 * DELTA_BITS
    return du - bit_pen - LAT_PEN - ENR_PEN, du, bit_pen

# ---- perturbation sets ----
# (label, base_g dict scaled/perturbed, base_r dict)
def scale_base(sf):
    return {g: BASE_G_NOM[g] * sf for g in BASE_G_NOM}

perturbations = [
    ("base(g) x 0.80 (ratios kept)", scale_base(0.80), BASE_R_NOM),
    ("base(g) x 1.00 (nominal)",     scale_base(1.00), BASE_R_NOM),
    ("base(g) x 1.20 (ratios kept)", scale_base(1.20), BASE_R_NOM),
    ("base_r +0.10 (g2:0.70->0.80, g3:0.92->1.02)", BASE_G_NOM,
     {2: BASE_R_NOM[2] + 0.10, 3: BASE_R_NOM[3] + 0.10}),
    ("base_r -0.10 (g2:0.70->0.60, g3:0.92->0.82)", BASE_G_NOM,
     {2: BASE_R_NOM[2] - 0.10, 3: BASE_R_NOM[3] - 0.10}),
]

print("=" * 96)
print("NET per-region per-step marginal reward of g=1 -> g=2 upgrade "
      "(s_i=0.875, SNR=10 dB)")
print("=" * 96)
print(f"{'perturbation':52s} | {'dU':>6s} | {'uncorr net (l1=.015)':>20s} | "
      f"{'corr net (l1=.008)':>18s} | {'recall bonus':>12s}")
print("-" * 96)

rows = []
for label, bg, br in perturbations:
    net_u, du, bp_u = net_marginal(bg, SNR, S_I, LAMBDA1_UNCORR)
    net_c, _ , bp_c = net_marginal(bg, SNR, S_I, LAMBDA1_CORR)
    bonus = d_recall_bonus(br)
    rows.append((label, du, net_u, net_c, bonus))
    print(f"{label:52s} | {du:6.4f} | {net_u:20.4f} | {net_c:18.4f} | {bonus:12.4f}")

print("-" * 96)
print("Sign check (uncorrected must be <=0, corrected must be >=0):")
for label, du, net_u, net_c, bonus in rows:
    ok = (net_u <= 0) and (net_c >= 0)
    print(f"  {label:52s} uncorr={net_u:+.4f} {'OK(neg)' if net_u<=0 else 'POS!'} | "
          f"corr={net_c:+.4f} {'OK(pos)' if net_c>=0 else 'NEG!'} | {'' if ok else '*** FAIL ***'}")

# Also report the full *corrected* reward net (incl. recall term, correction A)
# to show the implemented corrected reward is strongly positive and robust.
# full = net_I_corrected + recall_bonus  (correction A adds a strictly positive term)
print("\nFull corrected-reward net (net_I_corr + lambda_R*dRecall bonus):")
for label, bg, br in [(p[0], p[1], p[2]) for p in perturbations]:
    net_c, _, _ = net_marginal(bg, SNR, S_I, LAMBDA1_CORR)
    bonus = d_recall_bonus(br)
    print(f"  {label:52s} net_full = {net_c + bonus:+.4f}  (net_I_corr {net_c:+.4f} + bonus {bonus:+.4f})")
