#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <cmath>

// ANNIE plot style
#include "rootlogon.hpp" 

#include "sample_histograms.hpp"
#include "sample_utils.hpp"

Double_t chi2(TH1D* data_hist, TH1D* mc_hist){
    int n_bins_data = data_hist->GetNbinsX();
    int n_bins_mc = mc_hist->GetNbinsX();

    if(n_bins_data!=n_bins_mc){
        std::cout << "Error! The number of bins in the mc doesn't match the number of bins in the data hist..." << std::endl;
        return -1;
    }

    double chi2_val = 0.;
    for(int i=0; i < n_bins_data; i++){
        chi2_val+=(TMath::Power((data_hist->GetBinContent(i+1) - mc_hist->GetBinContent(i+1)),2)/(data_hist->GetBinContent(i+1)));
    }
    return chi2_val;
}



Double_t qprim(Double_t *x, Double_t *par){
    Double_t xx = x[0];
    return (xx + par[0]*xx + par[1]*TMath::Power(xx,2));
}




void make_2D_model_figure() {

    TFile* thru_muon_file = TFile::Open("./samples/throughgoing_muons_v1.root");
    TFile* thru_muon_data_file = TFile::Open("./samples/through_going_muon_data.root");
    std::string thru_mu_name = "Muons";
    std::string thru_mu_data_name = "MuonData";
    sample_features thru_muon_mc_features;
    sample_features thru_muon_data_features;
    sample_histograms thru_muon_histograms(thru_mu_name,40,1000.,4500.,0.,0.2);
    sample_histograms thru_muon_data_histograms(thru_mu_data_name,40,1000.,4500.,0.,0.2);
    read_and_fill(thru_muon_file,thru_mu_name, thru_muon_mc_features);
    read_and_fill(thru_muon_data_file,thru_mu_data_name,thru_muon_data_features);

    for(int i=0; i<thru_muon_data_features.cluster_pe.size(); i++){
        thru_muon_data_histograms.cl_pe_hist->Fill(thru_muon_data_features.cluster_pe.at(i));
        thru_muon_data_histograms.cl_qb_hist->Fill(thru_muon_data_features.cluster_qb.at(i));

    }
    const int nBinsX = 500;
    const int nBinsY = 100;
    const double xMin = -0.2, xMax = 1.;
    const double yMin = -2e-6, yMax = 2e-6;

    TH2D* hist = new TH2D("hist", "2D Function Histogram;X;Y", nBinsX, xMin, xMax, nBinsY, yMin, yMax);

    for (int i = 1; i <= nBinsX; ++i) {
        for (int j = 1; j <= nBinsY; ++j) {
            double A = hist->GetXaxis()->GetBinCenter(i);
            double B = hist->GetYaxis()->GetBinCenter(j);
            double param_set[2]={A,B};
            TH1D *alt_a = new TH1D("altA", "Alt A; cluster charge [p.e]; Number of events", 40, 1000., 4500.);
            for(int i=0; i<thru_muon_mc_features.cluster_pe.size(); i++){
                double thru_mu_cluster_pe = thru_muon_mc_features.cluster_pe.at(i);
                double thru_mu_cluster_a = qprim(&thru_mu_cluster_pe,param_set);
                alt_a->Fill(thru_mu_cluster_a);
            }

            int n_thru_muon_data = thru_muon_data_histograms.cl_pe_hist->Integral();
            thru_muon_histograms.cl_pe_hist->Scale((1./thru_muon_histograms.cl_pe_hist->Integral())*thru_muon_data_histograms.cl_pe_hist->Integral());
            alt_a->Scale((1./alt_a->Integral())*n_thru_muon_data);
            hist->SetBinContent(i, j, chi2(thru_muon_data_histograms.cl_pe_hist,alt_a));
         delete alt_a;
        }
    }
    TCanvas* canvas = new TCanvas("canvas", "2D Histogram", 800, 600);
    gStyle->SetOptStat(0);
    hist->Draw("COLZ");
    canvas->SaveAs("2DHistogram.png");
}
