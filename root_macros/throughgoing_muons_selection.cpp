#include <string>
#include <vector> 
#include <iostream> 
#include <fstream>

// ROOT headers
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TStopwatch.h"

// ANNIE plot style
#include "rootlogon.hpp"





int TEST_RUN = -1;
int TEST_RUN_MC = -1;


void read_data(std::string file_list){

    /*
    
    This function reads data n-tuples and creates 
    a ROOT file with the entries passing the selection
    
    */

    std::cout << "Reading data n-tuples..." << std::endl;
    if(TEST_RUN > 0){
        std::cout << "Test run, iterating over " << TEST_RUN << " files" << std::endl;

    }
    
    std::ifstream listFile(file_list);
    if(!listFile){
        std::cerr << "Error: could not open: " << file_list << std::endl;
    }



    TChain *data_tree = new TChain("data");
    // Iterate over files 
    std::string filePath;
    int file_counter = 0;
    while(std::getline(listFile, filePath)){

        if(filePath.empty()) continue;
        if(file_counter == TEST_RUN){
            break;
        }
        data_tree->Add(filePath.c_str());
        file_counter+=1;
    }


    float ct2;
    float cpe1;
    float cb1;
    int    ch1;
    int    b1;
    int    tmrd1;
    int    mrd_yes;
    int    t1;
    int    nv1;
    std::vector<double>  *hz1=nullptr;
    std::vector<double>  *hpe1=nullptr;
    std::vector<double>  *ht1=nullptr;
    std::vector<double>  *hid1=nullptr;
    int    tg1;


    data_tree->SetBranchAddress("cluster_Qb", &cb1);
    data_tree->SetBranchAddress("cluster_time_BRF", &ct2); 
    data_tree->SetBranchAddress("cluster_PE", &cpe1);
    data_tree->SetBranchAddress("cluster_Hits", &ch1);
    data_tree->SetBranchAddress("isBrightest", &b1);
    data_tree->SetBranchAddress("TankMRDCoinc", &tmrd1);
    data_tree->SetBranchAddress("MRD_activity", &mrd_yes);
    data_tree->SetBranchAddress("MRD_Track", &t1);
    data_tree->SetBranchAddress("NoVeto", &nv1);
    data_tree->SetBranchAddress("hitZ", &hz1);
    data_tree->SetBranchAddress("hitPE", &hpe1);
    data_tree->SetBranchAddress("hitT", &ht1);
    data_tree->SetBranchAddress("hitID", &hid1);
    data_tree->SetBranchAddress("MRDThrough", &tg1);

    int nEvents = data_tree->GetEntries();
    std::cout << "Number of entries in data chain: " << data_tree->GetEntries() << std::endl;


        // Create output file and new tree
    TFile *outputFile = TFile::Open("through_going_muon_data.root", "RECREATE");
    TTree *outTree = new TTree("MuonData", "Tree with selected entries");

    // Create branches in output tree
    outTree->Branch("cluster_pe", &cpe1, "cluster_pe/F");
    outTree->Branch("cluster_Qb", &cb1, "cluster_Qb/F");
    outTree->Branch("cluster_time", &ct2, "cluster_time/F");



    int selected_thru = 0; 

    for(int i=0; i < nEvents; i++){
        data_tree->GetEntry(i);
        double *hz1_v1 = hz1->data();
        double *hpe1_v1 = hpe1->data();
        double *ht1_v1 = ht1->data();
        double *hid1_v1 = hid1->data();
        float a = 0;
        float cluster_charge = 0.; 


        // Charge barycenter is calculated with all hits (?) 
        for(int j=0; j<ch1; j++){

            a+=hz1_v1[j]*hpe1_v1[j];
        }




        for(int j=0; j<ch1; j++){

            // Neglect negative hit charges and limit saturation effects 
            if(hpe1_v1[j] > 0. && hpe1_v1[j] < 350.){
                cluster_charge+=hpe1_v1[j];
            }
            else{
                continue;}
        }
 

        bool througgoing = (t1 == 1)    &&  // 1 track
                           (tmrd1 == 1) &&  // Tank-MRD coincidence
                           (nv1 == 0)   &&  // Vetoed (FMV yes)
                           (b1 == 1)    &&  // Brightest object
                           (cluster_charge >= 1000 && cluster_charge <= 6000) && // 1000 < cluster PE < 6000 (why (?))
                           (cb1 > 0. && cb1 < 0.2)    && // Cluster charge balance < 0.2
                           (ch1 >= 100.) && // More than 100 hits indicate that the muon topology is center-aligned 
                           (ct2 >= 200   && ct2 <= 1800) && // Inside spill time
                           (a>0.);   // Charge barycenter downstream 

        if(througgoing){
            outTree->Fill();
            selected_thru+=1;

        }
   }
        
   std::cout << "Number of selected throughgoing muons in data: " << selected_thru << std::endl;
   // Write and clean up
   outputFile->cd();
   outTree->Write();
   outputFile->Close();

}
