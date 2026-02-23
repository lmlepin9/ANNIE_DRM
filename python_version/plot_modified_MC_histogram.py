#!/usr/bin/env python3
import argparse
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from charge_to_E import *  # your existing function


def smear_event(E, C, rng):
    """Event-by-event fractional smearing: sigma_E = C * E."""
    E = np.asarray(E, float)
    C = float(C)
    if C <= 0:
        return E.copy()
    sigma = C * E
    return E + rng.normal(0.0, sigma, size=E.size)


def smear_histogram_constant(bin_centers, mc_counts, C, nsig=6):
    """
    Histogram smearing by discrete convolution with Gaussian kernel of fixed sigma = C.

    Assumptions:
      - uniform bin spacing
      - bin_centers correspond to mc_counts bins

    Returns:
      smeared_counts (same shape as mc_counts)
    """
    bin_centers = np.asarray(bin_centers, float)
    mc_counts   = np.asarray(mc_counts, float)
    C = float(C)

    if C <= 0:
        return mc_counts.copy()

    # uniform spacing check
    dx = np.diff(bin_centers)
    if not np.allclose(dx, dx[0], rtol=1e-6, atol=1e-12):
        raise ValueError("smear_histogram_constant assumes uniform bin spacing.")
    binw = dx[0]
    n = bin_centers.size
    smeared = np.zeros_like(mc_counts)

    # truncate kernel at ±nsig sigma expressed in bins
    half_width_bins = int(np.ceil((nsig * C) / binw))
    print(half_width_bins)

    for i, Ei in enumerate(bin_centers):
        j0 = max(0, i - half_width_bins)
        j1 = min(n - 1, i + half_width_bins)

        js = np.arange(j0, j1 + 1)
        Ej = bin_centers[js]
        w = np.exp(-0.5 * ((Ej - Ei) / C) ** 2)
        wsum = w.sum()
        if wsum > 0:
            w /= wsum

        smeared[js] += mc_counts[i] * w

    return smeared


def apply_scale(E, A):
    """Scale shift: E' = E*(1 + A)."""
    E = np.asarray(E, float)
    return E * (1.0 + float(A))


def calculate_hist_chi2_reduced(data_counts, mc_counts):
    data_counts = np.asarray(data_counts,float)
    mc_counts = np.asarray(mc_counts,float)
    non_zero_mc_bins = mc_counts>0
    data_counts = data_counts[non_zero_mc_bins]
    mc_counts = mc_counts[non_zero_mc_bins]
    n_dof = len(data_counts)
    temp_chi2 = np.sum(np.square(data_counts - mc_counts)/mc_counts)
    return temp_chi2/n_dof


def main():
    ap = argparse.ArgumentParser(
        description="Plot Data vs Nominal MC vs Modified MC (scale A + smear C) and report chi2 in legend."
    )
    ap.add_argument("--data-file", default="../samples/through_going_muon_data.root")
    ap.add_argument("--data-tree", default="MuonData")
    ap.add_argument("--mc-file", default="../samples/thrugoing_muons_MC_tilted.root")
    ap.add_argument("--mc-tree", default="Muons")
    ap.add_argument("--branch", default="cluster_pe")

    ap.add_argument("--A", type=float, default=0.0, help="Scale shift parameter (E -> E*(1+A))")
    ap.add_argument("--C", type=float, default=0.0, help="Fractional smearing (sigma_E = C*E)")

    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--emin", type=float, default=100)
    ap.add_argument("--emax", type=float, default=500)

    ap.add_argument("--smear-mode", choices=["event", "hist"], default="hist",
                    help="How to apply smearing: per-event Gaussian or histogram convolution.")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed used if smear-mode=event")

    ap.add_argument("--out", default="../figures/data_mc_modified_mc.png")
    ap.add_argument("--title", default="Data vs MC (nominal and modified)")
    args = ap.parse_args()

    # --- Load ---
    t_data = uproot.open(args.data_file)[args.data_tree]
    t_mc   = uproot.open(args.mc_file)[args.mc_tree]

    pe_data = t_data[args.branch].array(library="np")
    pe_mc   = t_mc[args.branch].array(library="np")

    # --- Convert PE -> Energy ---
    E_data = reco_energy_muons(pe_data)
    E_mc   = reco_energy_muons(pe_mc)

    # --- Binning ---
    if args.emin is None:
        args.emin = float(min(np.min(E_data), np.min(E_mc)))
    if args.emax is None:
        args.emax = float(max(np.max(E_data), np.max(E_mc)))

    edges = np.linspace(args.emin, args.emax, args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    data_counts, _ = np.histogram(E_data, bins=edges)
    mc_counts_nom, _ = np.histogram(E_mc, bins=edges)

    # Optional: normalize MC to data entries (common for shape comparisons)
    n_data = data_counts.sum()
    n_mc_nom = mc_counts_nom.sum()
    scale_to_data = (n_data / n_mc_nom) if n_mc_nom > 0 else 1.0
    mc_counts_nom_scaled = mc_counts_nom * scale_to_data

    # --- Modified MC: apply A then C ---
    E_mod = apply_scale(E_mc, args.A)

    if args.smear_mode == "event":
        rng = np.random.default_rng(args.seed)
        E_mod = smear_event(E_mod, args.C, rng)
        mc_counts_mod, _ = np.histogram(E_mod, bins=edges)
        mc_counts_mod = mc_counts_mod * scale_to_data
    else:
        # Histogram route: scale approximately by remapping centers (interp density), then convolve
        # Step 1: build counts histogram after scaling at event-level (exact for scaling, still fast)
        # (This avoids having to implement exact edge-remap here.)
        mc_counts_scaled_evt, _ = np.histogram(E_mod, bins=edges)
        mc_counts_scaled_evt = mc_counts_scaled_evt * scale_to_data

        # Step 2: apply histogram smearing convolution
        mc_counts_mod = smear_histogram_constant(centers, mc_counts_scaled_evt, args.C)

    # --- Chi2 ---
    chi2_nom = calculate_hist_chi2_reduced(data_counts, mc_counts_nom_scaled)
    chi2_mod = calculate_hist_chi2_reduced(data_counts, mc_counts_mod)

    # --- Plot ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(dpi=200)

    # Data with Poisson errors
    yerr = np.sqrt(data_counts)
    plt.errorbar(centers, data_counts, yerr=yerr, fmt="o", ms=4, capsize=2, label="Data", color="black")

    # Nominal MC
    plt.hist(edges[:-1], bins=edges, weights=mc_counts_nom_scaled,
             histtype="step",
             label=f"MC nominal  (χ²/ndof={chi2_nom:.1f})")

    # Modified MC
    plt.hist(edges[:-1], bins=edges, weights=mc_counts_mod,
             histtype="step", ls="--",
             label=f"MC modified A={args.A:+.3f}, C={args.C:.3f}  (χ²/ndof={chi2_mod:.1f})")

    plt.xlabel("Energy [MeV]")
    plt.ylabel("Entries / bin")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=6
    )
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
