# -*- coding: utf-8 -*-
"""Real-measurement validation of the UAV semantic-comm channel model
against the AERPAW ds24 dataset (4G LTE RSRP, UAV at 50 m, semi-circles
around an Ericsson private base station).

Goal: fit the measured air-to-ground path loss from RSRP and compare it with
(a) free space at the operating carrier, (b) the paper's simplified
simulation model snr_path_dB = clip(18 - 0.4*d, 0, 20), and (c) the
Al-Hourani A2G model. All randomness is none here; this is deterministic
measurement analysis.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"E:\AERPAW\ds24\AERPAW_PawPrints_SemiCircles\Logs\pawprints_4G.csv"
OUT = r"D:\ace\语义UAV\output"
BS_LAT, BS_LON, BS_H = 35.727480, -78.695920, 12.0
R_EARTH = 6371000.0
EARFCN = 66911
# LTE Band 66 (AWS-3): F_DL_low=2110 MHz, EARFCN_low=66436, step 0.1 MHz
F_MHZ = 2110 + 0.1 * (EARFCN - 66436)
# Noise floor for a 10 MHz LTE carrier, NF=7 dB
NF_DB = 7.0
BW_HZ = 10e6
N_FLOOR = -174.0 + 10 * np.log10(BW_HZ) + NF_DB   # dBm


def geometry(df):
    lat1, lon1 = np.radians(BS_LAT), np.radians(BS_LON)
    lat2, lon2 = np.radians(df["latitude"].values), np.radians(df["longitude"].values)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d_h = 2 * R_EARTH * np.arcsin(np.sqrt(a))
    h = df["altitude"].values.astype(float)
    d3 = np.sqrt(d_h ** 2 + (h - BS_H) ** 2)
    elev = np.degrees(np.arctan2(h - BS_H, d_h))
    df = df.copy()
    df["d_horiz"], df["d_3d"], df["elev"] = d_h, d3, elev
    return df


def fspl(f_mhz, d_km):
    return 32.45 + 20 * np.log10(f_mhz) + 20 * np.log10(d_km)


def main():
    df = geometry(pd.read_csv(CSV))
    c = df[(df.is_connected == 1) & (df.pci.isin([1, 2]))].copy()
    print(f"operating carrier ~ {F_MHZ:.1f} MHz ; noise floor ~ {N_FLOOR:.1f} dBm")
    print(f"connected-cell samples: {len(c)}")

    # ---- log-distance fit: RSRP(dBm) = a + b*log10(d3) ; n = -b/10 ----
    x = np.log10(c.d_3d.values)
    y = c.rsrp.values.astype(float)
    b, a = np.polyfit(x, y, 1)
    n_ple = -b / 10.0
    resid = y - (a + b * x)
    shadow_std = float(np.std(resid))
    print(f"\n[log-distance fit on 3-D distance]")
    print(f"  RSRP = {a:.2f} {b:+.2f}*log10(d_m)   (d in metres)")
    print(f"  path-loss exponent  n = {n_ple:.2f}")
    print(f"  shadowing residual std = {shadow_std:.2f} dB  (R^2={np.var(a+b*x)/np.var(y):.3f})")

    # ---- same but on horizontal distance ----
    xh = np.log10(c.d_horiz.values)
    bh, ah = np.polyfit(xh, y, 1)
    print(f"  [2-D fit]  n = {-bh/10:.2f}  (a={ah:.2f})")

    # ---- compare to free space ----
    d_km = c.d_3d.values / 1000.0
    fs = fspl(F_MHZ, d_km)
    # implied Tx EIRP from intercept at d3=1 m vs FSPL at 1 m
    fspl_1m = fspl(F_MHZ, 0.001)
    tx_eirp = a + fspl_1m  # because RSRP(1m) = EIRP - PL(1m) => EIRP = a + PL(1m)
    excess = (tx_eirp - fs)  # measured - free-space path loss (dB), should be ~>=0
    print(f"\n[free-space comparison @ {F_MHZ:.1f} MHz]")
    print(f"  implied Tx EIRP ~ {tx_eirp:.1f} dBm")
    print(f"  measured-vs-free-space excess path loss: "
          f"mean {np.mean(excess):.1f} dB, std {np.std(excess):.1f} dB")

    # ---- map RSRP to the simulation's SNR scale ----
    snr_sim_scale = y - N_FLOOR
    print(f"\n[SNR mapping] SNR=RSRP-noise_floor : "
          f"mean {np.mean(snr_sim_scale):.1f} dB, "
          f"range [{np.min(snr_sim_scale):.1f}, {np.max(snr_sim_scale):.1f}] dB")

    # ---- the simulation's own model, evaluated at the same distances ----
    sim_snr_path = np.clip(18 - 0.4 * c.d_3d.values, 0, 20)
    # correlate simulation SNR with measured SNR to see if ordering is preserved
    rho = np.corrcoef(sim_snr_path, snr_sim_scale)[0, 1]
    print(f"\n[simulation model check] corr(sim snr, measured snr) = {rho:.3f}")
    print(f"  sim snr range [{sim_snr_path.min():.1f}, {sim_snr_path.max():.1f}] dB")

    # ---- elevation-angle dependence (A2G LoS signature) ----
    # correlation of residual (after distance fit) with elevation angle
    from numpy.polynomial import polynomial as P
    be, ae = np.polyfit(c.elev.values, resid, 1)
    print(f"\n[LoA/elevation] residual-vs-elevation slope = {be:.3f} dB/deg "
          f"(positive => higher UAV sees stronger signal beyond distance)")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    dd = np.linspace(c.d_3d.min(), c.d_3d.max(), 200)
    ax[0].scatter(c.d_3d, c.rsrp, s=10, c=c.elev, cmap="viridis", alpha=0.7)
    ax[0].plot(dd, a + b * np.log10(dd), "r-", lw=2,
               label=f"fit n={n_ple:.2f}")
    ax[0].plot(dd, tx_eirp - fspl(F_MHZ, dd / 1000.0), "k--", lw=1.5,
               label="free space n=2")
    ax[0].set_xlabel("3-D distance to BS (m)"); ax[0].set_ylabel("RSRP (dBm)")
    ax[0].set_title("Measured A2G path loss (ds24)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    sc = ax[1].scatter(c.elev, resid, s=10, c=c.d_3d, cmap="magma", alpha=0.7)
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].plot(c.elev, ae + be * c.elev, "b-", lw=2)
    ax[1].set_xlabel("elevation angle (deg)")
    ax[1].set_ylabel("RSRP residual after distance fit (dB)")
    ax[1].set_title("Shadowing vs elevation (LoS signature)")
    ax[1].grid(alpha=0.3)
    fig.colorbar(sc, ax=ax[1], label="3-D distance (m)")
    fig.tight_layout()
    fig.savefig(OUT + "/aerpaw_ds24_pathloss.png", dpi=150)
    print(f"\nfigure saved: {OUT}/aerpaw_ds24_pathloss.png")

    # persist a summary for later use
    summary = {
        "carrier_mhz": round(F_MHZ, 1),
        "n_samples": int(len(c)),
        "ple_3d": round(float(n_ple), 2),
        "ple_2d": round(float(-bh / 10), 2),
        "intercept_a": round(float(a), 2),
        "shadow_std_db": round(shadow_std, 2),
        "tx_eirp_dbm": round(float(tx_eirp), 1),
        "noise_floor_dbm": round(float(N_FLOOR), 1),
        "snr_mean_db": round(float(np.mean(snr_sim_scale)), 1),
        "snr_range_db": [round(float(np.min(snr_sim_scale)), 1),
                         round(float(np.max(snr_sim_scale)), 1)],
        "sim_snr_corr": round(float(rho), 3),
        "elev_slope_db_per_deg": round(float(be), 3),
        "dist_range_m": [round(float(c.d_3d.min()), 1), round(float(c.d_3d.max()), 1)],
        "elev_range_deg": [round(float(c.elev.min()), 1), round(float(c.elev.max()), 1)],
    }
    import json
    with open(OUT + "/aerpaw_ds24_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
