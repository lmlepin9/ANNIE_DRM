#include <string>
#include <vector> 
#include <iostream> 
#include <fstream>


// ROOT headers
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMath.h"

// ANNIE plot style 
#include "rootlogon.hpp"


int TEST_RUN_DATA = -1;



void read_data(std::string file_list){

    /*

    This function reads data n-tuples and fills
    vectors with relevant information 

    */

    std::cout << "Reading data n-tuples..." << std::endl;
    if(TEST_RUN_DATA > 0){
    std::cout << "Test run, iterating over " << TEST_RUN_DATA << " files" << std::endl;

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
    if(file_counter == TEST_RUN_DATA){
    break;
    }
    data_tree->Add(filePath.c_str());
    file_counter+=1;
    }

    data_tree->SetBranchStatus("*", 0);

    data_tree->SetBranchStatus("run_number", 1);
    data_tree->SetBranchStatus("number_of_clusters",1);
    data_tree->SetBranchStatus("event_number", 1);
    data_tree->SetBranchStatus("cluster_Qb", 1);
    data_tree->SetBranchStatus("cluster_time", 1); 
    data_tree->SetBranchStatus("cluster_PE", 1);
    data_tree->SetBranchStatus("cluster_Number", 1);
    data_tree->SetBranchStatus("cluster_Hits", 1);
    data_tree->SetBranchStatus("isBrightest", 1);
    data_tree->SetBranchStatus("TankMRDCoinc", 1);
    data_tree->SetBranchStatus("MRD_activity", 1);
    data_tree->SetBranchStatus("hadExtended", 1);
    data_tree->SetBranchStatus("NoVeto", 1);
    data_tree->SetBranchStatus("hitZ", 1);
    data_tree->SetBranchStatus("hitPE", 1);
    data_tree->SetBranchStatus("hitT", 1);
    data_tree->SetBranchStatus("hitID", 1);


    float ct2;
    float cpe1;
    float cb1;
    int   ch1;
    Long64_t    cn1;
    Long64_t    noc1;
    Long64_t    evn1; 
    int    rn1; 
    int    b1;
    int    tmrd1;
    int    mrd_yes;
    int    t1;
    int    nv1;
    int    had_ext;
    std::vector<double>  *hz1=nullptr;
    std::vector<double>  *hpe1=nullptr;
    std::vector<double>  *ht1=nullptr;
    std::vector<double>  *hid1=nullptr;


    data_tree->SetBranchAddress("run_number", &rn1);
    data_tree->SetBranchAddress("number_of_clusters", &noc1);
    data_tree->SetBranchAddress("event_number", &evn1);
    data_tree->SetBranchAddress("cluster_Qb", &cb1);
    data_tree->SetBranchAddress("cluster_time", &ct2); 
    data_tree->SetBranchAddress("cluster_PE", &cpe1);
    data_tree->SetBranchAddress("cluster_Number", &cn1);
    data_tree->SetBranchAddress("cluster_Hits", &ch1);
    data_tree->SetBranchAddress("isBrightest", &b1);
    data_tree->SetBranchAddress("TankMRDCoinc", &tmrd1);
    data_tree->SetBranchAddress("MRD_activity", &mrd_yes);
    data_tree->SetBranchAddress("hadExtended", &had_ext);
    data_tree->SetBranchAddress("NoVeto", &nv1);
    data_tree->SetBranchAddress("hitZ", &hz1);
    data_tree->SetBranchAddress("hitPE", &hpe1);
    data_tree->SetBranchAddress("hitT", &ht1);
    data_tree->SetBranchAddress("hitID", &hid1);


    int nEvents = data_tree->GetEntries();
    std::cout << "Number of entries in data chain: " << data_tree->GetEntries() << std::endl;


    int selected_muons = 0;
    int selected_michels = 0; 

            // Create output file and new tree
    TFile *outputFile = TFile::Open("michel_electron_data.root", "RECREATE");
    TTree *outTree = new TTree("MichelData", "Tree with selected entries");

    // Create branches in output tree
    outTree->Branch("cluster_pe", &cpe1, "cluster_pe/F");
    outTree->Branch("cluster_Qb", &cb1, "cluster_Qb/F");
    outTree->Branch("cluster_time", &ct2, "cluster_time/F");


    for(int i=0; i < nEvents; i++){
        if(i%100000==0){std::cout << "Processing entry: " << i << std::endl;}
        data_tree->GetEntry(i);


        double *hz1_v1 = hz1->data();
        double *hpe1_v1 = hpe1->data();
        double *ht1_v1 = ht1->data();
        double *hid1_v1 = hid1->data();
        float a = 0;
        for(int j=0; j<ch1; j++){
        a+=hz1_v1[j]*hpe1_v1[j];
        }

        bool dirt_muon = (tmrd1 == 0) &&  // No tank-MRD coincidence
                        (nv1 == 0)   &&  // Vetoed (FMV yes)
                        (b1 == 1)    &&  // Brightest object
                        (noc1 > 1) && // More than one cluster
                        (had_ext != 0) && // Extended window to look for Michels
                        (ch1 >= 100) && // at least 50 pmt hits
                        (cpe1 >= 1000 && cpe1 <= 4000) && // 1000 < cluster PE < 4000 (why (?))
                        (cb1 > 0. && cb1 < 0.2)    && // Cluster charge balance < 0.2
                        (ct2 >= 200   && ct2 <= 1750) && // Inside spill time
                        (a>0.);   // Charge barycenter downstream 

        if(dirt_muon){

            float parent_time = ct2;
            Long64_t parent_evn = evn1;
            int parent_run = rn1;
            selected_muons+=1;


            // Loop again to find michels

            bool temp_michel = false;
            for(int k = i+1; k < nEvents; k++){
                data_tree->GetEntry(k);

                // Break if no longer same event or if we found a Michel candidate
                if(evn1 != parent_evn || rn1!= parent_run){break;}
            
                    
                    

                float adj_time = ct2 - parent_time;
                //std::cout << "Adjusted time: " << adj_time << std::endl;
                //std::cout << cb2 << " " << cpe2 << std::endl;
                bool michel_electron = (adj_time >= 200. && adj_time <= 5000) &&
                                        (cpe1 > 0. && cpe1 < 800.) &&
                                        //(ch1 >= 20) &&
                                        (cb1 > 0. && cb1 < 0.2);
                if(michel_electron){


                    temp_michel=true;
                    selected_michels+=1;
                    outTree->Fill();
                    break;
                }

            }
                
    
                
            //std::cout << "--------------------" << std::endl;
            //delete sub_tree;
        }
        else{continue;}

    }

    std::cout << "Number of selected dirt muons in data: " << selected_muons << std::endl;
    std::cout << "Number of selected Michels in data: " << selected_michels << std::endl;
       // Write and clean up
   outputFile->cd();
   outTree->Write();
   outputFile->Close();

}
