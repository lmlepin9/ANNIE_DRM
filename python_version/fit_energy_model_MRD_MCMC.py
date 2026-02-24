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

from common_tools import * 

# ---- Force float64 everywhere to avoid Op dtype mismatches ----
pytensor.config.floatX = "float64"


class Chi2Op(Op):
    # Always accept full parameter set (A, B, C) as float64 scalars
    itypes = [pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dscalar]

    def __init__(self, input_config):
        self.samples_name = []
        self.samples_data_E_vals = []
        self.samples_mc_E_vals = []
        self.samples_range = []
        self.samples_data_bins = []

        with open(input_config, "r") as f:
            config = yaml.safe_load(f)
            print(f"Reading data for {config['MC_config']} MC samples")

            for sample in config["calib_samples"]:
                temp_mc_Eloss = get_mrd_Eloss(sample["mc_sample"],sample["mc_tree"])
                temp_data_Eloss = get_mrd_Eloss(sample["data_sample"],sample["data_tree"])


                self.samples_name.append(sample["sample_name"])
                self.samples_range.append((sample["E_low"], sample["E_up"]))

                self.samples_data_E_vals.append(np.asarray(temp_data_Eloss, dtype=np.float64))
                self.samples_mc_E_vals.append(np.asarray(temp_mc_Eloss, dtype=np.float64))

                data_counts, _ = np.histogram(temp_data_Eloss,bins=20,range=((sample["E_low"], sample["E_up"])))
                self.samples_data_bins.append(np.asarray(data_counts, dtype=np.float64))


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
            E = self.samples_mc_E_vals[i]  # already float64

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
    parameter_optimization("../configs/thru_mu_MRD_parametric.yaml", fit_params=("A","C"),this_category="muons_MRD_parametric")
