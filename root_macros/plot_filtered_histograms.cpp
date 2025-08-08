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


// My includes
#include "sample_histograms.hpp"
#include "sample_utils.hpp"


void plot_filtered_histograms(){
    TFile* thru_muon_file = TFile::Open("./samples/throughgoing_muons_v1.root");
    TFile* thru_muon_data_file = TFile::Open("./samples/through_going_muon_data.root");
    TFile* michel_file = TFile::Open("./samples/Michels_v1.root");
    TFile* michel_data_file = TFile::Open("./samples/michel_electron_data.root");

    sample_features thru_muon_mc_features;
    sample_features michel_mc_features;
    sample_features thru_muon_data_features;
    sample_features michel_data_features;


    std::string thru_mu_name = "Muons";
    std::string michels_name = "Michels";
    std::string thru_mu_data_name = "MuonData";
    std::string michels_data_name = "MichelData";

    sample_histograms thru_muon_histograms(thru_mu_name,40,1000.,6000.,0.,0.2);
    sample_histograms thru_muon_data_histograms(thru_mu_data_name,40,1000.,6000.,0.,0.2);
    sample_histograms michel_histograms(michels_name,40.,0., 700.,0.,0.2);
    sample_histograms michel_data_histograms(michels_data_name,40.,0., 700.,0.,0.2);



    read_and_fill(thru_muon_file,thru_mu_name, thru_muon_mc_features);
    read_and_fill(michel_file, michels_name,  michel_mc_features);
    read_and_fill(thru_muon_data_file,thru_mu_data_name,thru_muon_data_features);
    read_and_fill(michel_data_file, michels_data_name,  michel_data_features);

    for(int i=0; i<thru_muon_mc_features.cluster_pe.size(); i++){
        thru_muon_histograms.cl_pe_hist->Fill(thru_muon_mc_features.cluster_pe.at(i));
        thru_muon_histograms.cl_qb_hist->Fill(thru_muon_mc_features.cluster_qb.at(i));
    }

    for(int i=0; i<michel_mc_features.cluster_pe.size(); i++){
        michel_histograms.cl_pe_hist->Fill(michel_mc_features.cluster_pe.at(i));
        michel_histograms.cl_qb_hist->Fill(michel_mc_features.cluster_qb.at(i));
    }

    for(int i=0; i<thru_muon_data_features.cluster_pe.size(); i++){
        thru_muon_data_histograms.cl_pe_hist->Fill(thru_muon_data_features.cluster_pe.at(i));
        thru_muon_data_histograms.cl_qb_hist->Fill(thru_muon_data_features.cluster_qb.at(i));

    }

    for(int i=0; i<michel_data_features.cluster_pe.size(); i++){
        michel_data_histograms.cl_pe_hist->Fill(michel_data_features.cluster_pe.at(i));
        michel_data_histograms.cl_qb_hist->Fill(michel_data_features.cluster_qb.at(i));
    }


    int n_thru_muon_data = thru_muon_data_features.cluster_pe.size(); 
    thru_muon_histograms.cl_pe_hist->Scale((1./thru_muon_histograms.cl_pe_hist->Integral())*n_thru_muon_data);
    thru_muon_histograms.cl_pe_hist->SetTitle("Through-going muons comparison");
    MakeRatioPlot(thru_muon_data_histograms.cl_pe_hist, thru_muon_histograms.cl_pe_hist, "./figures/through_going_muon_comparison.pdf");

    int n_michel_data = michel_data_features.cluster_pe.size();
    michel_histograms.cl_pe_hist->Scale((1./michel_histograms.cl_pe_hist->Integral())*n_michel_data);
    michel_histograms.cl_pe_hist->SetTitle("Michel electron comparison");
    MakeRatioPlot(michel_data_histograms.cl_pe_hist, michel_histograms.cl_pe_hist, "./figures/michel_electron_comparison.pdf");



    thru_muon_histograms.MakePlots();
    michel_histograms.MakePlots();
}