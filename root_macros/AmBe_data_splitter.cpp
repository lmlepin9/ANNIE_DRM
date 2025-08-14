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


void fill_in_port_pos(std::vector<TTree>,int port, Long_t pos){
    
}


void AmBe_data_splitter(){

    std::string data_file = "../samples/neutron_candidates_v5.root";
    TFile *f = TFile::Open(data_file.c_str());
    TTree *tree = (TTree*) f->Get("Neutrons");

    Double_t cluster_pe;
    Long64_t x_pos, y_pos, z_pos;

    tree->SetBranchAddress("cluster_charge",&cluster_pe);
    tree->SetBranchAddress("X_pos",&x_pos);
    tree->SetBranchAddress("Y_pos",&y_pos);
    tree->SetBranchAddress("Z_pos",&z_pos);


    std::vector<int> ports{1,3,4,5};
    std::vector<long> pos{-100,-50,0,50,100};

    std::string output_file = "../samples/AmBe_data_port_pos_separated.root";
    TFile *outputFile = TFile::Open(output_file.c_str(), "RECREATE");

    std::cout << "Creating TTrees for each port/position..." << std::endl;

    std::map<std::string,TTree*> pt_map;


    for(int i = 0; i < ports.size(); i++){
        for(int j=0; j<pos.size(); j++){
            std::string tree_name = "AmBe_data_port_" + std::to_string(ports.at(i)) + "_position_" + std::to_string(pos.at(j));
            std::cout << "Creating TTree: " << tree_name << std::endl;
            TTree* temp_tree = new TTree(tree_name.c_str(), "AmBe data tree");
            temp_tree->Branch("cluster_charge",&cluster_pe,"cluster_charge/D");
            pt_map[tree_name] = temp_tree;
        }
    }

    std::cout << "Number of entries in data tree: " << tree->GetEntries() << std::endl;
    for(int i=0; i < tree->GetEntries(); i++){
        tree->GetEntry(i);

        int temp_port;

        // Port 5
        if(x_pos==0 && z_pos==0){temp_port = 5;}

        // Port 4
        else if(x_pos==-75 && z_pos == 0){temp_port=4;}

        // Port 1
        else if(x_pos== 0 && z_pos == -75){temp_port=1;}

        // Port 3
        else if(x_pos==0 && z_pos == 102){temp_port=3;}

        std::string this_tree_name = "AmBe_data_port_" + std::to_string(temp_port) + "_position_" + std::to_string(y_pos);
        pt_map[this_tree_name]->Fill();
        continue;



    }



    // At the end we store the TTree in the output file 
    outputFile->cd();
    std::cout << "Number of trees created: " << pt_map.size() << std::endl;

    int n_total_events_stored = 0;
    for (const auto& pair : pt_map) {
        const std::string& key = pair.first;
        TTree* tree = pair.second;
        tree->Write();
        n_total_events_stored+=tree->GetEntries();
    }

    std::cout << "Sanity check, total number of entries stored: " << n_total_events_stored << std::endl;


    outputFile->Close();






}