#include <string>
#include <vector> 
#include <iostream> 
#include <fstream>

// ROOT headers
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TSystem.h"
#include "TMinuit.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLegend.h"

// ANNIE plot style
#include "rootlogon.hpp" 

#include "sample_histograms.hpp"
#include "sample_utils.hpp"






void run_selection(std::string port, std::string position, std::ofstream& out_csv){
    std::string files_path = "/exp/annie/data/users/doran/WCSim_AmBe_samples/BC_files/pmt_tilting_v1/QE_1.50_WB_ETEL_LUX_1.0_HM_WM_1.25_corrected_waveforms/";
    std::string file = files_path + "MC_AmBe_port"+port+"_z"+position+"_QE_1.50_WB_ETEL_LUX_1.0_HM_WM_1.25.root";
    TFile* f = TFile::Open(file.c_str());
    TTree* tree = (TTree*)f->Get("phaseIITankClusterTree");
    std::cout << "Number of entries in this file: " << tree->GetEntries() << std::endl;

    Double_t cluster_time, cluster_pe, cluster_cb;
    Int_t cluster_number; 

    tree->SetBranchAddress("clusterTime", &cluster_time);
    tree->SetBranchAddress("clusterPE", &cluster_pe);
    tree->SetBranchAddress("clusterChargeBalance", &cluster_cb);
    tree->SetBranchAddress("clusterNumber", &cluster_number);

    // Create output file and new tree

    int pos_label;

    if(position=="minus50"){pos_label=-50;}
    else if(position=="minus100"){pos_label=-100;}
    else if(position=="0"){pos_label=0;}
    else if(position=="50"){pos_label=50;}
    else if(position=="100"){pos_label=100;}





    std::string output_file = "../samples/mc_neutron_port_" + port + "_position_" + std::to_string(pos_label) + ".root";
    TFile *outputFile = TFile::Open(output_file.c_str(), "RECREATE");
    TTree *outTree = new TTree("NeutronMC", "Tree with selected entries");
    // Create branches in output tree
    outTree->Branch("cluster_pe", &cluster_pe, "cluster_pe/D");
    outTree->Branch("cluster_Qb", &cluster_cb, "cluster_Qb/D");
    outTree->Branch("cluster_time", &cluster_time, "cluster_time/D");


    for(int i = 0; i < tree->GetEntries(); i++){
        tree->GetEntry(i);

        bool cluster_cuts = (cluster_pe < 70.) && (cluster_time > 2000. && cluster_time < 67000.) && (cluster_cb < 0.4);
        if(cluster_cuts){
            if(cluster_number!=0){
              continue;
            }
            else{
                if(i+1 < tree->GetEntries()){

                    // Inspect next entry
                    tree->GetEntry(i+1);
                    if(cluster_number!=0){
                        continue;
                    }
                    else{
                        tree->GetEntry(i);
                        outTree->Fill();
                    }
                }
            }

        }
        else{
            continue;
        }



    }
    // Write and clean up
   
   double n_simulated_neutrons = 20000; // each sample corresponds to 20k neutrons 
   std::cout << "Number of selected neutrons: " << outTree->GetEntries() << std::endl;
   std::cout << "Efficiency of this selection: " << (outTree->GetEntries()*1.)/(n_simulated_neutrons) << std::endl;
   out_csv << port << "," << position << "," << (outTree->GetEntries()*1.)/(n_simulated_neutrons) << "\n";
   outTree->Write();
   outputFile->Close();
   f->Close();
}





void MC_neutrons_selection(){

    std::vector<std::string> port_list = {"1","3","4","5"};
    std::vector<std::string> position_list = {"0","50","100","minus50","minus100"};

    std::string out_csv = "../samples/neutron_mc_selection_efficiency.csv";
    std::ofstream file(out_csv);
    file << "port,position,efficiency\n";
    
    for(int i=0;i<port_list.size();i++){
        for(int j=0; j < position_list.size(); j++){
        std::cout << "Running selection for port: " + port_list.at(i) + ", position: " + position_list.at(j) << std::endl;
        run_selection(port_list.at(i), position_list.at(j),file);
     } 
    }


}