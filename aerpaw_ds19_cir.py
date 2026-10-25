# -*- coding: utf-8 -*-
"""Complex-baseband CSI validation from AERPAW ds19 channel-sounder I/Q.

For each SigMF capture we recover the channel impulse response (CIR) by
matched filtering the received I/Q against the transmitted Zadoff-Chu pilot
zc[n] = exp(-j*pi*u*n^2/L), u=200, L=401 (verified: sharp peak, PAR~100).
From the CIR we measure per-capture:
  - dominant-tap channel gain |H|  (complex)
  - RMS delay spread from the power-delay profile
  - Rician K-factor (specular / diffuse power)
Aggregating over the flight we measure:
  - A2G path-loss exponent at 3.564 GHz (slope of 10log10|H|^2 vs log10 d3D)
  - fading-amplitude distribution (Rayleigh vs Rician)
  - delay-spread statistics
  - spatial autocorrelation of the channel gain
All gain statistics are calibration-normalised (we report exponents and
distributions, not absolute dBm), so Tx/Rx gain offsets cancel.
"""
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\AERPAW\ds19"
OUT = r"D:\ace\语义UAV\output"
L, U = 401, 200
ZC = np.exp(-1j * np.pi * U * np.arange(L) ** 2 / L)
R_EARTH = 6371000.0
BS_LAT, BS_LON, BS_H = 35.72747884, -78.69591754, 12.0
N_MAX = 500  # cap captures processed for runtime


def read_meta(meta_path):
    with open(meta_path) as f:
        m = json.load(f)
    caps = m.get("captures", [])
    if not caps:
        return None
    c = caps[0]
    tx, rx = c.get("core:tx_location", {}), c.get("core:rx_location", {})
    return {
        "meta": meta_path,
        "data": meta_path.replace(".sigmf-meta", ".sigmf-data"),
        "t": c.get("core:timestamp"),
        "dist": c.get("core:dist"),
        "stage": c.get("core:flight_stage"),
        "tx": tx, "rx": rx,
        "freq": caps[0].get("core:frequency", m["captures"][0].get("core:frequency")),
    }


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def cir_from_capture(data_path):
    raw = np.fromfile(data_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    if iq.size < L * 2:
        return None
    xc = np.correlate(iq, ZC, mode="valid")
    p = np.abs(xc)
    par = p.max() / (np.mean(p) + 1e-12)
    if par < 8:  # no clean pilot alignment -> skip weak/noisy capture
        return None
    k = int(np.argmax(p))
    W = 24
    seg = xc[max(0, k - W):k + W + 1]
    pdp = np.abs(seg) ** 2
    pdp = pdp / (pdp.sum() + 1e-12)
    taps = np.arange(len(seg)) - (k - max(0, k - W))
    mean_delay = np.sum(pdp * taps)
    rms_ds = np.sqrt(np.sum(pdp * (taps - mean_delay) ** 2))
    # K-factor: strongest tap vs rest (proxy within the CIR window)
    h = xc[k]
    # Rician K from CIR: ratio of coherent (peak) to diffuse power in window
    peak_pow = pdp.max()
    kfactor = peak_pow / (1 - peak_pow + 1e-12)
    return {"gain": h, "absgain": abs(h), "rms_delay": rms_ds,
            "kfac": kfactor, "par": par}


def main():
    metas = glob.glob(os.path.join(ROOT, "*", "*.sigmf-meta"))
    print(f"total sigmf-meta files: {len(metas)}")
    recs = []
    for mp in metas:
        r = read_meta(mp)
        if r is None:
            continue
        if r["stage"] != "Flight":
            continue
        if r["dist"] is None or r["rx"].get("latitude") is None:
            continue
        recs.append(r)
    df = pd.DataFrame(recs)
    # geometry
    d_h = haversine(BS_LAT, BS_LON, df["rx"].apply(lambda x: x["latitude"]).values,
                    df["rx"].apply(lambda x: x["longitude"]).values)
    h_uav = df["rx"].apply(lambda x: x["altitude"]).values.astype(float)
    df["d3d"] = np.sqrt(d_h ** 2 + (h_uav - BS_H) ** 2)
    df["elev"] = np.degrees(np.arctan2(h_uav - BS_H, d_h))
    df = df.sort_values("t").reset_index(drop=True)
    print(f"Flight captures: {len(df)}  dist=[{df.dist.min():.0f},{df.dist.max():.0f}]m  "
          f"alt=[{h_uav.min():.0f},{h_uav.max():.0f}]m")

    # sample evenly across distance for a representative subset
    if len(df) > N_MAX:
        idx = np.linspace(0, len(df) - 1, N_MAX).astype(int)
        df = df.iloc[idx].reset_index(drop=True)

    rows = []
    for i, r in df.iterrows():
        c = cir_from_capture(r["data"])
        if c is None:
            continue
        rows.append({"t": r["t"], "d3d": r["d3d"], "elev": r["elev"],
                     "dist": r["dist"], "absgain": c["absgain"],
                     "rms_delay": c["rms_delay"], "kfac": c["kfac"],
                     "par": c["par"], "re": c["gain"].real, "im": c["gain"].imag})
    cdf = pd.DataFrame(rows)
    print(f"processed CIRs: {len(cdf)}")
    cdf.to_csv(OUT + "/aerpaw_ds19_cir.csv", index=False)

    # ---- path-loss exponent (slope, calibration-independent) ----
    g_db = 20 * np.log10(cdf.absgain.values + 1e-12)
    x = np.log10(cdf.d3d.values)
    slope, intercept = np.polyfit(x, g_db, 1)
    ple = -slope / 20.0
    resid = g_db - (intercept + slope * x)
    print(f"\n[A2G path loss @ 3.564 GHz]  n = {ple:.2f}  "
          f"(slope {slope:.2f} dB/decade)  resid_std={np.std(resid):.2f} dB  "
          f"R^2={np.var(intercept+slope*x)/np.var(g_db):.3f}")

    # ---- fading amplitude distribution (Rayleigh vs Rician) ----
    amp = cdf.absgain.values
    amp_n = amp / np.sqrt(np.mean(amp ** 2))
    # Rayleigh would have mean(amp_n^2)=1 by construction; check K via E[|H|^4]/E[|H|^2]^2
    m2 = np.mean(amp ** 2)
    m4 = np.mean(amp ** 4)
    aic = m4 / (m2 ** 2)   # AIC: 2 for Rayleigh, larger for Rician
    print(f"\n[fading] amplitude AIC = E[|H|^4]/E[|H|^2]^2 = {aic:.2f} "
          f"(2.0=Rayleigh, >2=specular/Rician)")
    print(f"  median CIR K-factor (peak/window) = {np.median(cdf.kfac):.2f}")
    print(f"  mean |H|^2 = {m2:.4f}  (arb. units)")

    # ---- RMS delay spread ----
    # samples at 56 MS/s -> ns
    rms_ns = cdf.rms_delay.values / 56e6 * 1e9
    print(f"\n[delay spread] RMS: mean={np.mean(rms_ns):.1f} ns, "
          f"median={np.median(rms_ns):.1f} ns, "
          f"max={np.max(rms_ns):.1f} ns  (window +-24 taps ~ {48/56e6*1e9:.0f} ns)")

    # ---- spatial autocorrelation of gain vs distance spacing ----
    cdf2 = cdf.sort_values("t")
    g = np.log10(cdf2.absgain.values + 1e-12)
    dd = cdf2.d3d.diff().values
    lag1 = np.corrcoef(g[:-1], g[1:])[0, 1]
    print(f"\n[channel-gain autocorrelation] lag-1 (consecutive captures) = {lag1:.3f}  "
          f"mean spacing={np.nanmean(dd):.1f} m")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(cdf.d3d, g_db, s=8, alpha=0.5)
    xs = np.linspace(cdf.d3d.min(), cdf.d3d.max(), 100)
    ax[0].plot(xs, intercept + slope * np.log10(xs), "r-", lw=2, label=f"n={ple:.2f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("3-D distance (m)")
    ax[0].set_ylabel("20log10|H| (dB, rel.)")
    ax[0].set_title("A2G path loss @3.564 GHz"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].hist(amp_n ** 2, bins=60, density=True, alpha=0.7)
    rr = np.linspace(0, amp_n.max() ** 2, 200)
    ax[1].plot(rr, np.exp(-rr), "k-", lw=2, label="exp (Rayleigh |H|^2)")
    ax[1].set_xlabel("normalised |H|^2"); ax[1].set_ylabel("pdf")
    ax[1].set_title(f"fading dist. AIC={aic:.2f}"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].hist(rms_ns, bins=40, density=True, alpha=0.7)
    ax[2].set_xlabel("RMS delay spread (ns)"); ax[2].set_ylabel("pdf")
    ax[2].set_title(f"delay spread (med {np.median(rms_ns):.0f} ns)")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + "/aerpaw_ds19_channel.png", dpi=150)

    summary = {
        "n_cir": int(len(cdf)),
        "carrier_ghz": 3.564,
        "dist_range_m": [round(float(cdf.d3d.min()), 0), round(float(cdf.d3d.max()), 0)],
        "ple": round(float(ple), 2),
        "ple_resid_std_db": round(float(np.std(resid)), 2),
        "ple_r2": round(float(np.var(intercept + slope * x) / np.var(g_db)), 3),
        "fading_aic": round(float(aic), 2),
        "median_kfac": round(float(np.median(cdf.kfac)), 2),
        "rms_delay_ns": {"mean": round(float(np.mean(rms_ns)), 1),
                         "median": round(float(np.median(rms_ns)), 1),
                         "max": round(float(np.max(rms_ns)), 1)},
        "gain_autocorr_lag1": round(float(lag1), 3),
    }
    with open(OUT + "/aerpaw_ds19_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nfigure: {OUT}/aerpaw_ds19_channel.png")


if __name__ == "__main__":
    main()
