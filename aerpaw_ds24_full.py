# -*- coding: utf-8 -*-
"""Refined AERPAW ds24 path-loss validation.

Adds over aerpaw_validate.py:
  - flight-regime filter (steady cruise) vs takeoff/landing
  - LoS (high elevation) vs sub-LoS split -> two-regime path-loss exponents
  - shadowing / fast-fading decomposition with lag-1 autocorrelation
  - Al-Hourani LoS-probability curve from measured elevation
"""
import json
import numpy as np
import pandas as pd

CSV = r"E:\AERPAW\ds24\AERPAW_PawPrints_SemiCircles\Logs\pawprints_4G.csv"
OUT = r"D:\ace\语义UAV\output"
BS_LAT, BS_LON, BS_H = 35.727480, -78.695920, 12.0
R_EARTH = 6371000.0
EARFCN = 66911
F_MHZ = 2110 + 0.1 * (EARFCN - 66436)
N_FLOOR = -174.0 + 10 * np.log10(10e6) + 7.0  # 10 MHz BW, NF 7 dB


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


def fit_ple(sub, label):
    x = np.log10(sub.d_3d.values)
    y = sub.rsrp.values.astype(float)
    b, a = np.polyfit(x, y, 1)
    n = -b / 10.0
    resid = y - (a + b * x)
    print(f"  [{label}] N={len(sub):4d}  n={n:.2f}  a={a:6.2f}  "
          f"shadow_std={np.std(resid):.2f} dB  "
          f"d3D=[{sub.d_3d.min():.0f},{sub.d_3d.max():.0f}]m  "
          f"elev=[{sub.elev.min():.0f},{sub.elev.max():.0f}]deg")
    return {"n": round(float(n), 2), "a": round(float(a), 2),
            "shadow_std_db": round(float(np.std(resid)), 2),
            "N": int(len(sub))}


def alhourani_plos(elev_deg, a_par=9.61, b_par=0.16):
    # Wafaa 2014 (Al-Hourani) LoS probability, theta in degrees
    th = np.radians(elev_deg)
    return 1.0 / (1.0 + a_par * np.exp(-b_par * (np.degrees(th) - a_par)))


def main():
    df = geometry(pd.read_csv(CSV))
    c = df[(df.is_connected == 1) & (df.pci.isin([1, 2]))].copy()
    print(f"carrier {F_MHZ:.1f} MHz ; connected samples {len(c)}")

    print("\n=== path-loss exponent by regime ===")
    r_all = fit_ple(c, "all altitudes")
    flight = c[c.altitude >= 25.0]
    r_flight = fit_ple(flight, "cruise (alt>=25m)")
    takeoff = c[c.altitude < 25.0]
    r_take = fit_ple(takeoff, "takeoff/land (alt<25m)")

    # LoS (high elevation) vs low elevation split, cruise only
    los = flight[flight.elev >= 30.0]
    nlos = flight[(flight.elev >= 0) & (flight.elev < 30.0)]
    r_los = fit_ple(los, "cruise LoS (elev>=30)")
    r_nlos = fit_ple(nlos, "cruise sub-LoS (0<=elev<30)")

    # Al-Hourani LoS prob distribution over measured elevations
    plos = alhourani_plos(c.elev.values)
    print(f"\n=== Al-Hourani LoS probability (a=9.61,b=0.16) ===")
    print(f"  mean P_los={np.mean(plos):.3f}  "
          f"at elev 0deg: {alhourani_plos(np.array([0]))[0]:.3f}, "
          f"30deg: {alhourani_plos(np.array([30]))[0]:.3f}, "
          f"51deg: {alhourani_plos(np.array([51]))[0]:.3f}")

    # ---- shadowing / fast-fading decomposition on cruise ----
    # residual after log-distance fit, ordered by time
    x = np.log10(flight.d_3d.values)
    y = flight.rsrp.values.astype(float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    # slow component = moving average (window 5 ~ 25 m at 5 m/s / 1 Hz), fast = remainder
    w = 5
    slow = pd.Series(resid).rolling(w, center=True, min_periods=1).mean().values
    fast = resid - slow
    print(f"\n=== shadowing / fading decomposition (cruise, w={w}) ===")
    print(f"  total resid std = {np.std(resid):.2f} dB")
    print(f"  slow (shadowing) std = {np.std(slow):.2f} dB")
    print(f"  fast (small-scale) std = {np.std(fast):.2f} dB")
    lag1_slow = np.corrcoef(slow[:-1], slow[1:])[0, 1]
    lag1_fast = np.corrcoef(fast[:-1], fast[1:])[0, 1]
    print(f"  lag-1 autocorr: slow={lag1_slow:.3f}  fast={lag1_fast:.3f}")

    # ---- SNR scale ----
    snr = y - N_FLOOR
    print(f"\n=== SNR mapping (cruise) ===")
    print(f"  SNR mean={np.mean(snr):.1f} dB  std={np.std(snr):.1f}  "
          f"range=[{np.min(snr):.1f},{np.max(snr):.1f}]")

    summary = {
        "carrier_mhz": round(F_MHZ, 1),
        "n_connected": int(len(c)),
        "ple": {"all": r_all, "cruise": r_flight, "takeoff": r_take,
                "cruise_los": r_los, "cruise_nlos": r_nlos},
        "alhourani_mean_plos": round(float(np.mean(plos)), 3),
        "shadow": {"total_std_db": round(float(np.std(resid)), 2),
                   "slow_std_db": round(float(np.std(slow)), 2),
                   "fast_std_db": round(float(np.std(fast)), 2),
                   "lag1_slow": round(float(lag1_slow), 3),
                   "lag1_fast": round(float(lag1_fast), 3)},
        "snr_cruise": {"mean_db": round(float(np.mean(snr)), 1),
                       "std_db": round(float(np.std(snr)), 1),
                       "range_db": [round(float(np.min(snr)), 1),
                                    round(float(np.max(snr)), 1)]},
    }
    with open(OUT + "/aerpaw_ds24_full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
