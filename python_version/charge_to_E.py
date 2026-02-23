import numpy as np



def reco_energy_michels(pe):
    '''
    Based on Steven D
    Calculation for the NC analysis
    '''
    return (pe + 2.90)/5.90


def reco_energy_muons(pe):
    '''
    Based on the energy
    scale estimate by 
    Michael Nielsony 
    '''
    pe_to_E = 0.08534
    return pe*pe_to_E


