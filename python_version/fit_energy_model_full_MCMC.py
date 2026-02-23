import uproot
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

# Reco Energy
from charge_to_E import *

# Stuff for pyMC
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Op
import arviz as az

# To make histograms
from plot_histograms import plot_data_mc_with_ratio

# Needed by smear_histogram
from scipy.stats import norm

# ---- Force float64 everywhere to avoid Op dtype mismatches ----
pytensor.config.floatX = "float64"


def get_clusters_pe(sample_path, tree_name):
    this_tree = uproot.open(sample_path)[tree_name]
    cluster_pe = this_tree["cluster_pe"].array().to_numpy()
    return cluster_pe


def make_data_hist(data, range, Nbins=20):
    data_counts, _ = np.histogram(data, bins=Nbins)
    return _, data_counts


def smear_histogram(bin_centers, mc_counts, C):
    mc_counts = np.asarray(mc_counts, dtype=np.float64)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    C = np.float64(C)

    smeared = np.zeros_like(mc_counts, dtype=np.float64)

    for i, E in enumerate(bin_centers):
        sigma = C * E
        if sigma <= 0:
            smeared[i] += mc_counts[i]
            continue
        weights = norm.pdf(bin_centers, loc=E, scale=sigma)
        wsum = np.sum(weights)
        if wsum > 0:
            weights /= wsum
            smeared += mc_counts[i] * weights
        else:
            smeared[i] += mc_counts[i]

    # Preserve total counts
    ssum = np.sum(smeared)
    msum = np.sum(mc_counts)
    if ssum > 0 and msum > 0:
        smeared *= (msum / ssum)

    return smeared

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


def save_posterior_plot(
    trace,
    parameter_name,
    outdir="../figures",
    bins=60,
    hdi_prob=0.68,
    category="default",
    report_outdir="../outputs",
    cov_params=None,
):
    """
    Plot 1D posterior for `parameter_name` and save to:
      {outdir}/posterior_{category}_{parametername}.png

    Features:
      - Marks the HDI limits (default 68%) with vertical dashed lines
      - Draws the posterior MEDIAN as a solid line and shows it in the legend
      - Writes a text report containing:
          median, +/- uncertainties (from HDI bounds), and covariance matrix
        to:
          {report_outdir}/posterior_report_{category}.txt
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(report_outdir, exist_ok=True)

    # --- Flatten samples across chains/draws ---
    vals = np.asarray(trace.posterior[parameter_name].values).reshape(-1).astype(np.float64)

    # --- Summary stats ---
    med = float(np.median(vals))
    hdi = az.hdi(vals, hdi_prob=hdi_prob)
    lo, hi = float(hdi[0]), float(hdi[1])

    # Asymmetric uncertainties around the median using HDI bounds
    err_minus = med - lo
    err_plus  = hi - med

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=bins, density=True, histtype="step")

    ax.axvline(lo, linestyle="--", color="black", label=f"{int(hdi_prob*100)}% HDI")
    ax.axvline(hi, linestyle="--", color="black")

    # Median line + legend entry
    ax.axvline(med, linestyle="-", color="black", alpha=0.7,
               label=f"median = {med:.6g}")

    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Posterior density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(outdir, f"posterior_{category}_{parameter_name}.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(
        f"Saved posterior plot to: {outpath}  "
        f"(median={med:.6g}, -{err_minus:.6g}/+{err_plus:.6g}, "
        f"HDI {hdi_prob:.2f}: {lo:.6g}–{hi:.6g})"
    )

    # --- Covariance matrix report ---
    if cov_params is None:
        cov_params = list(trace.posterior.data_vars.keys())
    else:
        cov_params = list(cov_params)

    # Build sample matrix [n_params, n_samples]
    samples = []
    for p in cov_params:
        v = np.asarray(trace.posterior[p].values).reshape(-1).astype(np.float64)
        samples.append(v)
    samples = np.vstack(samples)

    cov = np.cov(samples)  # shape (P,P)

    report_path = os.path.join(report_outdir, f"posterior_report_{category}.txt")
    with open(report_path, "w") as f:
        f.write(f"Posterior report (category = {category})\n")
        f.write("=" * 60 + "\n\n")

        # Parameter-wise median and uncertainties from HDI bounds
        f.write(f"Parameter summaries (median -Δ/+Δ from {int(hdi_prob*100)}% HDI):\n")
        for p, v in zip(cov_params, samples):
            med_p = float(np.median(v))
            hdi_p = az.hdi(v, hdi_prob=hdi_prob)
            lo_p, hi_p = float(hdi_p[0]), float(hdi_p[1])
            em = med_p - lo_p
            ep = hi_p - med_p
            f.write(f"  {p:>12s} : {med_p:.8g}  -{em:.8g}  +{ep:.8g}\n")

        f.write("\nCovariance matrix (order: " + ", ".join(cov_params) + ")\n")
        f.write(np.array2string(cov, precision=6, suppress_small=False) + "\n")

    print(f"Wrote posterior report to: {report_path}")



class Chi2Op(Op):
    # Always accept full parameter set (A, B, C) as float64 scalars
    itypes = [pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dscalar]

    def __init__(self, input_config):
        self.samples_name = []
        self.samples_data_qe_vals = []
        self.samples_mc_qe_vals = []
        self.samples_range = []
        self.samples_data_bins = []

        with open(input_config, "r") as f:
            config = yaml.safe_load(f)
            print(f"Reading data for {config['MC_config']} MC samples")

            for sample in config["calib_samples"]:
                temp_mc_qe = get_clusters_pe(sample["mc_sample"], sample["mc_tree"])
                temp_data_qe = get_clusters_pe(sample["data_sample"], sample["data_tree"])

                if sample["sample_name"] == "throughgoing_muons":
                    temp_mc_E = reco_energy_muons(temp_mc_qe)
                    temp_data_E = reco_energy_muons(temp_data_qe)
                else:
                    temp_mc_E = reco_energy_michels(temp_mc_qe)
                    temp_data_E = reco_energy_michels(temp_data_qe)

                self.samples_name.append(sample["sample_name"])
                self.samples_range.append((sample["E_low"], sample["E_up"]))

                self.samples_data_qe_vals.append(np.asarray(temp_data_E, dtype=np.float64))
                self.samples_mc_qe_vals.append(np.asarray(temp_mc_E, dtype=np.float64))


                data_counts, _ = np.histogram(temp_data_E,bins=20,range=((sample["E_low"], sample["E_up"])))

                self.samples_data_bins.append(np.asarray(data_counts, dtype=np.float64))

                # Sanity check plot (untransformed)
                output_figure_path = (
                    f"../figures/data_mc_comparison_{config['MC_config']}_{sample['sample_name']}.png"
                )
                plot_data_mc_with_ratio(
                    temp_data_E,
                    temp_mc_E,
                    output_figure_path,
                    bins=20,
                    range=(sample["E_low"], sample["E_up"]),
                )

        print(f"Calib samples found: {self.samples_name}")

    def PoissonChi2(self, data_bins, mc_bins, mu_floor=1e-12):
        n  = np.asarray(data_bins, dtype=np.float64)
        mu = np.asarray(mc_bins,   dtype=np.float64)

        # If model predicts ~0 where data has counts, that's impossible → inf
        if np.any((mu <= mu_floor) & (n > 0)):
            return np.inf

        # Safe floor for numerical stability (only affects bins where n==0 or mu is tiny)
        mu_safe = np.clip(mu, mu_floor, None)

        # Use analytic limit for n==0: contribution = mu
        out = np.empty_like(mu_safe)
        mask0 = (n == 0)
        out[mask0] = mu_safe[mask0]

        # Regular bins (n>0)
        mask = ~mask0
        out[mask] = mu_safe[mask] - n[mask] + n[mask] * (np.log(n[mask]) - np.log(mu_safe[mask]))

        return 2.0 * np.sum(out)


    def perform(self, node, inputs, outputs):
        A_in, B_in, C_in = inputs

        # Force float64 scalars (handles PyMC float32 inputs safely)
        A = np.float64(A_in)
        B = np.float64(B_in)
        C = np.float64(C_in)

        chi2 = 0.0
        n_samples = len(self.samples_name)

        for i in range(n_samples):
            E = self.samples_mc_qe_vals[i]  # already float64

            # Energy transform: E' = E*(1+A) + B*E^2  (equiv to E*(1+A+B*E))
            E_prime = E * (1.0 + A) + B * (E ** 2)

            # Histogram MC after transform
            mc_counts, edges = np.histogram(E_prime,bins=20, range=self.samples_range[i])
            mc_counts = np.asarray(mc_counts, dtype=np.float64)

            # Smear histogram (fractional resolution: sigma = C*E)
            if C > 0:
                centers = 0.5 * (edges[:-1] + edges[1:])
                mc_counts = smear_histogram_constant(centers, mc_counts, C)

            denom = np.sum(mc_counts)
            if denom <= 0:
                chi2 = np.inf
                break

            scaling = np.sum(self.samples_data_bins[i]) / denom
            mc_counts_scaled = mc_counts * scaling

            chi2 += self.PoissonChi2(self.samples_data_bins[i], mc_counts_scaled)

        outputs[0][0] = np.array(-0.5 * chi2, dtype=np.float64)


def parameter_optimization(input_yaml, fit_params=("A", "B", "C"),this_category="default"):
    """
    fit_params can be any subset of ("A","B","C"), e.g. ("A","C") or ("B",).
    Non-fitted params are fixed to 0.0.
    """
    fit_params = tuple(fit_params)
    logp_op = Chi2Op(input_yaml)

    with pm.Model() as model:
        # Free or fixed parameters (fixed -> float64 tensor 0.0)
        if "A" in fit_params:
            A = pm.Normal("A", mu=0.0, sigma=0.2)
            #A = pm.Uniform("A",lower=-0.2,upper=0.2)
        else:
            A = pt.as_tensor_variable(np.float64(0.0))

        if "B" in fit_params:
            # Sensible default: allows up to ~10% nonlinearity at E=500 => |B| ~ 2e-4
            B = pm.Normal("B", mu=0.0, sigma=5e-5)   # tune scale
        else:
            B = pt.as_tensor_variable(np.float64(0.0))

        if "C" in fit_params:
            # Fractional resolution >= 0
            #C = pm.Uniform("C",lower=0.,upper=50)
            C = pm.HalfNormal("C", sigma=np.float64(100))
        else:
            C = pt.as_tensor_variable(np.float64(0.0))

        pm.Potential("likelihood", logp_op(A, B, C))

        trace = pm.sample(
            50000,
            tune=2000,
            step=pm.Metropolis(),
            cores=4,
            chains=4,
        )

    # Summaries/plots only for free params
    var_names = [p for p in ("A", "B", "C") if p in fit_params]
    if var_names:
        summary = az.summary(trace, var_names=var_names)
        print(summary)

        # Posterior plots
        for p in var_names:
            save_posterior_plot(trace, p, outdir="../figures",category=this_category)

        # Posterior covariance matrix (free params order = var_names)
        samples = np.vstack([trace.posterior[p].values.reshape(-1) for p in var_names])
        cov = np.cov(samples)
        print("\nPosterior covariance (order: {}):\n{}".format(var_names, cov))

    return trace


if __name__ == "__main__":
    # Examples:
    parameter_optimization("./michels_tilted_wvfmsim.yaml", fit_params=("A","C"),this_category="michels_tilted_wvfmsim")
    parameter_optimization("./thru_mu_tilted_wvfmsim.yaml", fit_params=("A","C"),this_category="muons_tilted_wvfmsim")
    parameter_optimization("./thru_mu_michels_tilted_wvfmsim.yaml", fit_params=("A", "C"),this_category="muons_michels_tilted_wvfmsim")

    # parameter_optimization("./thru_mu_michels_nominal_parametric.yaml", fit_params=("A","B","C"))
    #parameter_optimization("./michels_nominal_parametric.yaml", fit_params=("A"),this_category="michels_nominal_parametric_A")
    #parameter_optimization("./michels_tilted_parametric.yaml", fit_params=("A"),this_category="michels_tilted_parametric_A")
    #parameter_optimization("./michels_nominal_parametric.yaml", fit_params=("A","C"),this_category="michels_nominal_parametric")
    #parameter_optimization("./michels_tilted_parametric.yaml", fit_params=("A","C"),this_category="michels_tilted_parametric")
    #parameter_optimization("./thru_mu_nominal_parametric.yaml", fit_params=("A"),this_category="muons_nominal_parametric_A")
    #parameter_optimization("./thru_mu_tilted_parametric.yaml", fit_params=("A"),this_category="muons_tilted_parametric_A")
    #parameter_optimization("./thru_mu_nominal_parametric.yaml", fit_params=("A","C"),this_category="muons_nominal_parametric")
    #parameter_optimization("./thru_mu_tilted_parametric.yaml", fit_params=("A","C"),this_category="muons_tilted_parametric")
    #parameter_optimization("./thru_mu_michels_nominal_parametric.yaml", fit_params=("A",))

    #parameter_optimization("./thru_mu_tilted_parametric.yaml", fit_params=("A","C"),this_category="muons_tilted_parametric")
    #parameter_optimization("./michels_tilted_parametric.yaml", fit_params=("A","C"),this_category="michels_tilted_parametric")
    #parameter_optimization("./thru_mu_michels_tilted_parametric.yaml", fit_params=("A","C"),this_category="muons_michels_tilted_parametric")
