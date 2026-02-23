print('\nLoading packages...\n')
import os
import uproot        
import numpy as np
from tqdm import trange
import matplotlib as mpl
import matplotlib.pyplot as plt
import time 
import pandas as pd 
font = {'family' : 'serif', 'size' : 10 }
mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = 'cm' # Set the math font to Computer Modern
mpl.rcParams['legend.fontsize'] = 1


'''
We go with the following approach: 

- Read entries in the phaseIITriggerTree -> This is the true tree, but also has all the entries.
  We can check a bunch of different things here, before we dive into looking at the actual clusters.
  By using the event number, I can match the respective entries on the tank and mrd trees. 

  * MRD_yes
  * TankMRDCoinc
  * EXT (has extended window)

  if any of these fail, we skip the event.


'''

TEST_RUN = False
TAG="MC_tilted"
directory = "/pnfs/annie/persistent/analysis/v1.3.0/MC/world_tilt_shift/"
#directory = "/pnfs/annie/persistent/analysis/v1.3.0/MC/world/"


def array_to_root_branch(
    array,
    root_filename,
    tree_name,
    branch_name,
    mode="recreate",
):
    """
    Store a 1D NumPy array as a TBranch in a TTree (not RNTuple)
    using uproot >= 5.7 (explicit mktree).

    Parameters
    ----------
    array : np.ndarray
        1D NumPy array (length = number of entries)
    root_filename : str
        Name of the ROOT file
    tree_name : str
        Name of the TTree
    branch_name : str
        Name of the TBranch
    mode : str
        "recreate" (default) or "update"
    """

    array = np.asarray(array)

    if array.ndim != 1:
        raise ValueError("Only 1D arrays are supported")

    file_opener = uproot.recreate if mode == "recreate" else uproot.update

    with file_opener(root_filename) as f:

        # If the tree does not exist yet, create it explicitly
        if tree_name not in f:
            f.mktree(
                tree_name,
                {branch_name: array.dtype}
            )

        tree = f[tree_name]

        # Extend (fill) the branch
        tree.extend(
            {branch_name: array}
        )


def filter_hits_charge_and_z(charges, zcoords, lo=0.0, hi=350.0):
    """
    charges, zcoords: list of lists with identical ragged structure
    """
    if len(charges) != len(zcoords):
        raise ValueError("charges and zcoords must have same number of clusters")

    charges_out = []
    zcoords_out = []

    for cl_q, cl_z in zip(charges, zcoords):
        if len(cl_q) != len(cl_z):
            raise ValueError("Charge/Z mismatch inside a cluster")

        q_keep = []
        z_keep = []

        for q, z in zip(cl_q, cl_z):
            if lo <= q <= hi:
                q_keep.append(q)
                z_keep.append(z)

        charges_out.append(q_keep)
        zcoords_out.append(z_keep)

    return charges_out, zcoords_out



file_names = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

total_events = 0 
presel_events = 0
sel_events = 0

file_counter=0

thru_mu_cluster_pe = [] 
# MRD variables 
thru_mu_mrd_angle = []
thru_mu_mrd_tlength = []
thru_mu_mrd_eloss = []
thru_mu_mrd_side = []
thru_mu_mrd_stop = []
thru_mu_mrd_thru = [] 

start_time = time.time() 

for file_name in file_names:
    print(f"Processing file {file_name}")
    file_counter+=1
    if (TEST_RUN == True) and (file_counter > 100):
            break
    with uproot.open(directory + file_name) as file:

      
      # Truth tree
      try: 
        Trig = file["phaseIITriggerTree"]
        TEVN = Trig["eventNumber"].array()
        THTN = Trig["HasTank"].array()
        THMR = Trig["HasMRD"].array() # This is always one (?) for some reason??, I'll drop it for now...  
        TTMR = Trig["TankMRDCoinc"].array()
        TNVT = Trig["NoVeto"].array()
        THEX = Trig["Extended"].array()
        TTWR = Trig["trigword"].array()
        TBOK = Trig["beam_ok"].array()
        TNTR = Trig["numMRDTracks"].array() # my attempt to demand no MRD activity (?) 


        # MRD variables
        MRDTree = file["phaseIIMRDClusterTree"]
        MRDEVN = MRDTree['eventNumber'].array()
        MRDAngle = MRDTree['MRDTrackAngle'].array()
        MRDTrackL = MRDTree['MRDTrackLength'].array()
        MRDELoss = MRDTree['MRDEnergyLoss'].array()
        MRDSX = MRDTree['MRDTrackStartX'].array()
        MRDSY = MRDTree['MRDTrackStartY'].array()
        MRDSZ = MRDTree['MRDTrackStartZ'].array()
        MRDEX = MRDTree['MRDTrackStopX'].array()
        MRDEY = MRDTree['MRDTrackStopY'].array()
        MRDEZ = MRDTree['MRDTrackStopZ'].array()
        MRDSide = MRDTree['MRDSide'].array()
        MRDStop = MRDTree['MRDStop'].array()
        MRDTh = MRDTree['MRDThrough'].array()


        # Cluster tree
        Event = file["phaseIITankClusterTree"]
        ENUM = Event["eventNumber"].array()
        CT2 = Event["clusterTime"].array()
        CPE1 = Event["clusterPE"].array()
        CCB1 = Event["clusterChargeBalance"].array()
        CN1 = Event["clusterNumber"].array()
        CH1 = Event["clusterHits"].array()

        # These guys arrays of arrays...
        HZ1 = Event['hitZ'].array()
        HPE1 = Event['hitPE'].array()
        HT1 = Event['hitT'].array()
        HID1 = Event['hitDetID'].array()


        # Manipulate hits! 

        PRESEL_EVN = TEVN[(TTMR == 1) & (TNVT==0) & (TTWR==5) & (TBOK==1) &
                          (TNTR == 1) ]
        if(len(PRESEL_EVN)==0):continue 
        total_events += len(TEVN)
        presel_events += len(PRESEL_EVN)
    
        for ievent in PRESEL_EVN:
          this_ev_mask = np.where(ENUM==ievent)
          this_ev_mrd_mask = np.where(MRDEVN==ievent)
          this_ev_ct2 = CT2[this_ev_mask]
          this_ev_pe1 = CPE1[this_ev_mask]
          this_ev_cb1 = CCB1[this_ev_mask]
          this_ev_cn1 = CN1[this_ev_mask]
          this_ev_ch1 = CH1[this_ev_mask]
          this_ev_chz = HZ1[this_ev_mask]
          this_ev_chpe = HPE1[this_ev_mask]

          this_ev_mrd_angle = MRDAngle[this_ev_mrd_mask]
          this_ev_mrd_eloss = MRDELoss[this_ev_mrd_mask]
          this_ev_mrd_trackl = MRDTrackL[this_ev_mrd_mask]
          this_ev_mrd_side = MRDSide[this_ev_mrd_mask]
          this_ev_mrd_stop = MRDStop[this_ev_mrd_mask]
          this_ev_mrd_th = MRDTh[this_ev_mrd_mask]


          # Filter hits
          this_ev_chpe_filt, this_ev_chz_filt = filter_hits_charge_and_z(this_ev_chpe,this_ev_chz)

          # Get new cluster charges
          new_cluster_charge = np.array([np.sum(i) for i in this_ev_chpe_filt])
          this_ev_a = np.array([np.sum(np.array(i)*np.array(j)) for i,j in zip(this_ev_chz_filt,this_ev_chpe_filt)])

          # Get barycenter
          #this_ev_a =  np.sum(this_ev_chz_filt * this_ev_chpe_filt, axis=1)
          pre_cluster_mask = ((new_cluster_charge > 1000) & (new_cluster_charge < 6000) & 
                              (this_ev_ch1 > 100) & 
                              (this_ev_ct2 < 2000.) & 
                              (new_cluster_charge == np.max(new_cluster_charge)) &                  
                              (this_ev_cb1 > 0.) & (this_ev_cb1 <0.2) & (this_ev_a > 0.)) # Might need to recalculate the charge-balance?? 

 
  

          # Store the thru-mu charge 
          if len(pre_cluster_mask[pre_cluster_mask==1]) >0:
            thru_mu_cluster_pe.append(new_cluster_charge[pre_cluster_mask][0])
            thru_mu_mrd_angle.append(this_ev_mrd_angle[0][0])
            thru_mu_mrd_tlength.append(this_ev_mrd_trackl[0][0])
            thru_mu_mrd_eloss.append(this_ev_mrd_eloss[0][0])
            thru_mu_mrd_side.append(this_ev_mrd_side[0][0])
            thru_mu_mrd_stop.append(this_ev_mrd_stop[0][0])
            thru_mu_mrd_thru.append(this_ev_mrd_th[0][0])
            sel_events+=1
          else:
            continue

      except Exception as e:
        print("Something is wrong with this file...")
        print("Error occurred:", e)
      


elapsed_time = time.time() - start_time
print(f"Total processing time: {elapsed_time/60.} minutes")

print(f"Number of events {total_events}")
print(f"Number of preselected trigger {presel_events}")
print(f"Number of selected throug-going muons {sel_events}")
print(f"Fraction of thru-mus {(sel_events/total_events):.2f}")



#-------------- Make figures ----------------------------------------- 
fig = plt.figure(dpi=200)
plt.hist(thru_mu_cluster_pe,bins=40,color='red',histtype='step')
plt.xlabel("cluster charge [p.e]")
plt.ylabel("Number of entries")
plt.savefig(f"../figures/thru_mu_world_sample_{TAG}.png",bbox_inches='tight')


dirt_df = pd.DataFrame(thru_mu_cluster_pe,columns=["cluster_charge"])
dirt_df.to_csv(f"../samples/MC_thru_muon_{TAG}.csv")
array_to_root_branch(thru_mu_cluster_pe,f"../samples/thrugoing_muons_{TAG}.root","Muons","cluster_pe")
array_to_root_branch(thru_mu_mrd_angle,f"../samples/thrugoing_muons_{TAG}.root","Muons","MRDTrackAngle")
array_to_root_branch(thru_mu_mrd_tlength,f"../samples/thrugoing_muons_{TAG}.root","Muons","MRDTrackLength")
array_to_root_branch(thru_mu_mrd_eloss,f"../samples/thrugoing_muons_{TAG}.root","Muons","MRDTrackELoss")
array_to_root_branch(thru_mu_mrd_thru,f"../samples/thrugoing_muons_{TAG}.root","Muons","MRDThru")
array_to_root_branch(thru_mu_mrd_stop,f"../samples/thrugoing_muons_{TAG}.root","Muons","MRDStop")
array_to_root_branch(thru_mu_mrd_side,f"../samples/thrugoing_muons_{TAG}.root","Muons","MRDSide")
