import uproot
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

from common_tools import *



def make_MRD_histograms(data_sample,mc_sample):
    
    data_file = uproot.open(data_sample)
    mc_file = uproot.open(mc_sample)

    data_tree = data_file["Muons"]
    mc_tree = mc_file["Muons"]

    data_Eloss = data_tree["mrd_track_Eloss"].array()
    data_stop = data_tree["mrd_stop"].array()
    data_thru = data_tree["mrd_thru"].array()

    mc_Eloss = mc_tree["mrd_track_Eloss"].array()
    mc_stop = mc_tree["mrd_stop"].array()
    mc_thru = mc_tree["mrd_thru"].array() 

    print(f"Number of entries in the data sample: {len(data_Eloss)}")
    print(f"Number of entries in the MC sample: {len(mc_Eloss)}")

    data_stop_mask = (data_stop == 1)
    data_Eloss_stop = data_Eloss[data_stop_mask]
    print(f"Number of stopping muons in the data sample: {len(data_Eloss_stop)}")

    mc_stop_mask = (mc_stop == 1)
    mc_Eloss_stop = mc_Eloss[mc_stop_mask]
    print(f"Number of stopping muons in the MC sample: {len(mc_Eloss_stop)}")

    data_thru_mask = (data_thru == 1)
    data_Eloss_thru = data_Eloss[data_thru_mask]
    print(f"Number of thru muons in the data sample: {len(data_Eloss_thru)}")

    mc_thru_mask = (mc_thru == 1)
    mc_Eloss_thru = mc_Eloss[mc_thru_mask]
    print(f"Number of thru muons in the MC sample: {len(mc_Eloss_thru)}")


    plot_data_mc_with_ratio(data_Eloss_stop,mc_Eloss_stop,"../figures/MRD_stopping_muons_comparison.png",range=(300,1000),bins=20)
    plot_data_mc_with_ratio(data_Eloss_thru,mc_Eloss_thru,"../figures/MRD_thru_muons_comparison.png",range=(650,1000),bins=20)
    plot_data_mc_with_ratio(data_Eloss,mc_Eloss,"../figures/MRD_all_muons_comparison.png",range=(300,1000),bins=20)







if __name__ == "__main__":
    my_data_sample = "/exp/annie/app/users/lmoralep/clean_repo/ANNIE_DRM/samples/throughgoing_muons_data.root"
    my_mc_sample = "/exp/annie/app/users/lmoralep/clean_repo/ANNIE_DRM/samples/thrugoing_muons_MC_tilted_MRD_only.root"
    make_MRD_histograms(my_data_sample,my_mc_sample)