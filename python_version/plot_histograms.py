import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def calculate_hist_chi2_reduced(data_counts, mc_counts):
    non_zero_mc_bins = mc_counts>0
    data_counts = data_counts[non_zero_mc_bins]
    mc_counts = mc_counts[non_zero_mc_bins]
    n_dof = len(data_counts)
    temp_chi2 = np.sum(np.square(data_counts - mc_counts)/mc_counts)
    return temp_chi2/n_dof


def plot_data_mc_with_ratio(
    data,
    mc,
    output_path,
    bins=20,
    range=None,
    density=False,
    label_data="Data",
    label_mc="MC-world (scaled)",
    ratio_ylim=(0.0, 2.0),
    show_poisson_errors=True,
    figsize=(7, 6),
    dpi=150,
):
    """
    Plot Data vs MC histograms (MC scaled to match Data entries) + ratio panel (Data/MC),
    and save the figure to disk.

    Parameters
    ----------
    data, mc : array-like
        1D arrays of values.
    output_path : str
        Path where the figure will be saved (e.g. "figures/data_mc_ratio.png").
    """

    data = np.asarray(data).ravel()
    mc   = np.asarray(mc).ravel()

    # Decide plotting range if not provided
    if range is None:
        combined = np.concatenate([
            data[np.isfinite(data)],
            mc[np.isfinite(mc)]
        ])
        if combined.size == 0:
            raise ValueError("Both data and mc are empty (or all non-finite).")
        lo, hi = np.min(combined), np.max(combined)
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        range = (lo, hi)

    # Histogram counts
    data_counts, edges = np.histogram(data[np.isfinite(data)], bins=bins, range=range)
    mc_counts, _       = np.histogram(mc[np.isfinite(mc)],   bins=edges)

    # Scale MC to Data entries
    n_data = data_counts.sum()
    n_mc   = mc_counts.sum()
    scale  = (n_data / n_mc) if n_mc > 0 else 0.0
    mc_scaled = mc_counts * scale

    # Display normalization (optional)
    if density:
        data_display = data_counts / n_data if n_data > 0 else data_counts.astype(float)
        mc_display   = mc_scaled / n_data if n_data > 0 else mc_scaled.astype(float)
        y_label = "arbitrary units (density)"
    else:
        data_display = data_counts.astype(float)
        mc_display   = mc_scaled.astype(float)
        y_label = "arbitrary units"

    centers = 0.5 * (edges[:-1] + edges[1:])

    # Figure + axes
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax  = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    # --- Top panel ---
    ax.step(edges[:-1], mc_display, where="post", label=label_mc,color='red')
    ax.errorbar(
        centers,
        data_display,
        yerr=(np.sqrt(data_counts) / n_data if (density and show_poisson_errors and n_data > 0)
              else (np.sqrt(data_counts) if show_poisson_errors else None)),
        fmt="o",
        ms=4,
        capsize=2,
        label=label_data,
        color='black'
    )

# Collect existing legend entries (MC + Data)
    handles, labels = ax.get_legend_handles_labels()

    # Add a pure-text entry at the end (no marker/line)
    text_handle = Line2D([], [], color='none', label=fr"$\chi^2 / \mathrm{{ndf}} = {calculate_hist_chi2_reduced(data_counts,mc_display):.2f}$")
    handles.append(text_handle)
    labels.append(text_handle.get_label())

    ax.set_ylabel(y_label)

    ax.legend(handles,labels)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    # --- Ratio panel ---
    ratio = np.full_like(centers, np.nan, dtype=float)
    mask = mc_scaled > 0
    ratio[mask] = data_counts[mask] / mc_scaled[mask]

    if show_poisson_errors:
        ratio_err = np.full_like(centers, np.nan, dtype=float)
        ratio_err[mask] = np.sqrt(data_counts[mask]) / mc_scaled[mask]
        rax.errorbar(centers, ratio, yerr=ratio_err, fmt="o", ms=4, capsize=2,color='black')
    else:
        rax.plot(centers, ratio, "o", ms=4)

    rax.axhline(1.0, linestyle="--", linewidth=1)
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel("Data/MC")
    rax.set_xlabel("cluster charge [p.e]")
    rax.grid(True, alpha=0.3)

    # Save and close
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Chi2: {calculate_hist_chi2_reduced(data_counts,mc_display)}")

    return {
        "edges": edges,
        "data_counts": data_counts,
        "mc_counts": mc_counts,
        "mc_scaled": mc_scaled,
        "scale_factor": scale,
        "ratio": ratio,
    }




def plot_histograms(tag="nominal"):

    # Input ROOT files and TTree names
    file1 = "../samples/through_going_muon_data.root"
    tree1_name = "MuonData"

    file2 = f"../samples/thrugoing_muons_MC_{tag}.root"
    tree2_name = "Muons"

    file3 = "../samples/michel_electron_data.root"
    tree3_name = "MichelData"

    file4 = f"../samples/michel_electrons_MC_{tag}.root"
    tree4_name = "Michels"

    # Open the ROOT files
    tree1 = uproot.open(file1)[tree1_name]
    tree2 = uproot.open(file2)[tree2_name]
    tree3 = uproot.open(file3)[tree3_name]
    tree4 = uproot.open(file4)[tree4_name]

    # Read the "cluster_pe" branch as numpy arrays
    cluster_pe_mu_data = tree1["cluster_pe"].array()
    cluster_pe_mu_mc   = tree2["cluster_pe"].array()
    cluster_pe_me_data = tree3["cluster_pe"].array()
    cluster_pe_me_mc   = tree4["cluster_pe"].array()

    output_directory = "../figures/"



    plot_data_mc_with_ratio(cluster_pe_mu_data,cluster_pe_mu_mc, output_directory + "throughgoing_muons_mc_data_comparison_{tag}.png",range=(1000,6000))
    plot_data_mc_with_ratio(cluster_pe_me_data,cluster_pe_me_mc, output_directory + "michel_electrons_mc_data_comparison_{tag}.png",range=(0,650))


if __name__ == "main":
    