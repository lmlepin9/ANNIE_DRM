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

TEST_RUN = True
TAG="Data"
directory = "/pnfs/annie/persistent/processed/BeamClusterTrees/"


import numpy as np
import uproot


def arrays_to_root_tree(
    arrays,
    root_filename,
    tree_name,
    branch_names,
    mode="recreate",
    allow_overwrite_tree=False,
):
    """
    Store multiple 1D NumPy arrays as branches in a single TTree (not RNTuple)
    using uproot >= 5.7 (explicit mktree).

    Parameters
    ----------
    arrays : Sequence[np.ndarray]
        List/tuple of 1D NumPy arrays. All arrays must have the same length
        (number of entries).
    root_filename : str
        Name of the ROOT file.
    tree_name : str
        Name of the TTree.
    branch_names : Sequence[str]
        List/tuple of branch names, same length as `arrays`.
    mode : str
        "recreate" (default) or "update".
        - "recreate": overwrite file
        - "update": open existing file and update/create tree
    allow_overwrite_tree : bool
        If True and mode="update", will delete and recreate `tree_name` if it
        exists but is incompatible (missing branches / dtype mismatches).
        If False, will raise on incompatibilities.

    Notes
    -----
    - This writes fixed-length scalar branches (one value per entry).
    - For jagged arrays (variable-length per entry), you'd want awkward arrays
      and a different branch type.
    """

    if len(arrays) != len(branch_names):
        raise ValueError(
            f"`arrays` and `branch_names` must have the same length "
            f"({len(arrays)} vs {len(branch_names)})."
        )

    # Convert & validate
    arrays_np = []
    n_entries = None
    for i, a in enumerate(arrays):
        a = np.asarray(a)
        if a.ndim != 1:
            raise ValueError(
                f"Only 1D arrays are supported. "
                f"arrays[{i}] has ndim={a.ndim}."
            )
        if n_entries is None:
            n_entries = a.shape[0]
        elif a.shape[0] != n_entries:
            raise ValueError(
                f"All arrays must have the same length. "
                f"arrays[0] has {n_entries}, arrays[{i}] has {a.shape[0]}."
            )
        arrays_np.append(a)

    # Tree schema (branch: dtype)
    schema = {bn: arr.dtype for bn, arr in zip(branch_names, arrays_np)}

    file_opener = uproot.recreate if mode == "recreate" else uproot.update

    with file_opener(root_filename) as f:

        if tree_name not in f:
            # Create new tree with all branches
            f.mktree(tree_name, schema)
            tree = f[tree_name]
            tree.extend({bn: arr for bn, arr in zip(branch_names, arrays_np)})
            return

        # Tree exists
        tree = f[tree_name]

        # Check compatibility
        existing = tree.keys()  # branch names in file (strings)
        missing = [bn for bn in branch_names if bn not in existing]

        # dtype check: uproot returns a NumPy dtype for scalar branches
        dtype_mismatch = []
        for bn, arr in zip(branch_names, arrays_np):
            if bn in existing:
                try:
                    file_dtype = tree[bn].dtype
                    if np.dtype(file_dtype) != np.dtype(arr.dtype):
                        dtype_mismatch.append((bn, np.dtype(file_dtype), np.dtype(arr.dtype)))
                except Exception:
                    # If dtype can't be determined cleanly, treat as mismatch
                    dtype_mismatch.append((bn, None, np.dtype(arr.dtype)))

        if missing or dtype_mismatch:
            if mode != "update":
                raise ValueError(
                    f"Tree '{tree_name}' exists but is incompatible. "
                    f"Missing={missing}, dtype_mismatch={dtype_mismatch}."
                )

            if not allow_overwrite_tree:
                msg = [f"Tree '{tree_name}' exists but is incompatible:"]
                if missing:
                    msg.append(f"  - Missing branches: {missing}")
                if dtype_mismatch:
                    msg.append(
                        "  - Dtype mismatches: "
                        + ", ".join([f"{bn} ({fd} -> {ad})" for bn, fd, ad in dtype_mismatch])
                    )
                msg.append(
                    "Set allow_overwrite_tree=True to delete+recreate the tree, "
                    "or ensure the existing tree schema matches."
                )
                raise ValueError("\n".join(msg))

            # Delete and recreate tree with desired schema
            del f[tree_name]
            f.mktree(tree_name, schema)
            tree = f[tree_name]

        # Fill/extend
        tree.extend({bn: arr for bn, arr in zip(branch_names, arrays_np)})


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

# Tank variable
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

    if(".root" not in file_name):
        continue 

    print(f"Processing file {file_name}")
    file_counter+=1
    if (TEST_RUN == True) and (file_counter > 100):
            break
    with uproot.open(directory + file_name) as file:

        try: 

            Trig = file["Event"]
            TEVN = Trig["eventNumber"].array()
            THTN = Trig["HasTank"].array()
            THMR = Trig["HasMRD"].array() # This is always one (?) for some reason??, I'll drop it for now...  
            TTMR = Trig["TankMRDCoinc"].array()
            TNVT = Trig["NoVeto"].array()
            THEX = Trig["Extended"].array()
            TTWR = Trig["trigword"].array()
            TBOK = Trig["beam_ok"].array()

            # Cluster tree
            CT2 = Trig["clusterTime"].array()
            CPE1 = Trig["clusterPE"].array()
            CCB1 = Trig["clusterChargeBalance"].array()
            CH1 = Trig["clusterHits"].array()

            # These guys arrays of arrays...
            HZ1 = Trig['hitZ'].array()
            HPE1 = Trig['hitPE'].array()
            HT1 = Trig['hitT'].array()
            HID1 = Trig['hitDetID'].array()

            # MRD variables
            MRDAngle = Trig['MRDTrackAngle'].array()
            MRDTrackL = Trig['MRDTrackLength'].array()
            MRDELoss = Trig['MRDEnergyLoss'].array()
            MRDSX = Trig['MRDTrackStartX'].array()
            MRDSY = Trig['MRDTrackStartY'].array()
            MRDSZ = Trig['MRDTrackStartZ'].array()
            MRDEX = Trig['MRDTrackStopX'].array()
            MRDEY = Trig['MRDTrackStopY'].array()
            MRDEZ = Trig['MRDTrackStopZ'].array()
            MRDSide = Trig['MRDSide'].array()
            MRDStop = Trig['MRDStop'].array()
            MRDTh = Trig['MRDThrough'].array()

            # Use these with data???
            CLHITZ = Trig['Cluster_HitZ'].array()
            CLHITPE = Trig['Cluster_HitPE'].array()
            # Create array of number of tracks in MRD
            TNTR = np.array([len(x) for x in MRDAngle])


            PRESEL = (TTMR == 1) & (TNVT==0) & (TTWR==5) & (TBOK==1) & (TNTR == 1)
            ENTRY_NUMS = np.arange(0,len(TEVN))
            PRESEL_ENTRIES = ENTRY_NUMS[PRESEL == 1]


            #PRESEL_EVN = TEVN[(TTMR == 1) & (TNVT==0) & (TTWR==5) & (TBOK==1) &
            #                  (TNTR == 1) ] # Keep just one MRD track 

            total_events += len(TEVN)
            if(len(PRESEL_ENTRIES)==0):continue 
            presel_events += len(PRESEL_ENTRIES)
            for this_ev_mask in PRESEL_ENTRIES:
                this_ev_ct2 = CT2[this_ev_mask]
                this_ev_pe1 = CPE1[this_ev_mask]
                this_ev_cb1 = CCB1[this_ev_mask]
                this_ev_ch1 = CH1[this_ev_mask]
                this_ev_chz =  CLHITZ[this_ev_mask]
                this_ev_chpe = CLHITPE[this_ev_mask]

                this_ev_mrd_angle = MRDAngle[this_ev_mask]
                this_ev_mrd_eloss = MRDELoss[this_ev_mask]
                this_ev_mrd_trackl = MRDTrackL[this_ev_mask]
                this_ev_mrd_side = MRDSide[this_ev_mask]
                this_ev_mrd_stop = MRDStop[this_ev_mask]
                this_ev_mrd_th = MRDTh[this_ev_mask]

                # Get new cluster charges
                this_ev_a = np.array([np.sum(np.array(i)*np.array(j)) for i,j in zip(this_ev_chz,this_ev_chpe)])

                # Get barycenter
                pre_cluster_mask = ((this_ev_pe1 > 1000) & (this_ev_pe1 < 6000) & 
                                    (this_ev_ct2 > 200) & (this_ev_ct2 < 1800) &
                                    (this_ev_ch1 > 100) & 
                                    (this_ev_ct2 < 2000.) & 
                                    (this_ev_pe1 == np.max(this_ev_pe1)) &                  
                                    (this_ev_cb1 > 0.) & (this_ev_cb1 <0.2) & (this_ev_a > 0.)) # Might need to recalculate the charge-balance?? 
        
        

                # Store the thru-mu charge 
                if len(pre_cluster_mask[pre_cluster_mask==1]) >0:
                    thru_mu_cluster_pe.append(this_ev_pe1[pre_cluster_mask][0])
                    thru_mu_mrd_angle.append(this_ev_mrd_angle[0])
                    thru_mu_mrd_tlength.append(this_ev_mrd_trackl[0])
                    thru_mu_mrd_eloss.append(this_ev_mrd_eloss[0])
                    thru_mu_mrd_side.append(this_ev_mrd_side[0])
                    thru_mu_mrd_stop.append(this_ev_mrd_stop[0])
                    thru_mu_mrd_thru.append(this_ev_mrd_th[0])
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



#-------------- Make a diagnostic figure ----------------------------------------- 
fig = plt.figure(dpi=200)
plt.hist(thru_mu_cluster_pe,bins=40,color='red',histtype='step')
plt.xlabel("cluster charge [p.e]")
plt.ylabel("Number of entries")
plt.savefig(f"../figures/throughgoing_muon_data_LM_selection.png",bbox_inches='tight')


arrays_to_root_tree(
    arrays=[thru_mu_cluster_pe,thru_mu_mrd_angle,thru_mu_mrd_tlength,
    thru_mu_mrd_eloss,thru_mu_mrd_thru,thru_mu_mrd_stop,thru_mu_mrd_side],
    branch_names=["cluster_pe", "mrd_track_angle", "mrd_track_length","mrd_track_Eloss","mrd_thru","mrd_stop","mrd_side"],
    root_filename=f"../samples/throughgoing_muons_data.root",
    tree_name="Muons",
    mode="recreate",
)