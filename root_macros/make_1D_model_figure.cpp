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
        chi2_val+=(TMath::Power((data_hist->GetBinContent(i+1) - mc_hist->GetBinContent(i+1)),2)/data_hist->GetBinContent(i+1));
    }
    return chi2_val;
}


Double_t llr(TH1D* data_hist, TH1D* mc_hist){
    int n_bins_data = data_hist->GetNbinsX();
    int n_bins_mc = mc_hist->GetNbinsX();

    if(n_bins_data!=n_bins_mc){
        std::cout << "Error! The number of bins in the mc doesn't match the number of bins in the data hist..." << std::endl;
        return -1;
    }

    double llr_val = 0.;
    for(int i=0; i < n_bins_data; i++){
        double M_i = data_hist->GetBinContent(i+1);
        double mu_i = mc_hist->GetBinContent(i+1);
        llr_val+=2*(mu_i - M_i + M_i*TMath::Log(M_i/mu_i));
    }
    return llr_val;

}


Double_t qprim(Double_t *x, Double_t *par){
    Double_t xx = x[0];
    return (xx + par[0]*xx);
}




void make_1D_model_figure() {

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
    const int nBinsX = 100;
    const int nBinsY = 200;
    const double xMin = -0.05, xMax = 0.05;
    const double yMin = -8e-7, yMax = -4e-7;

    
    TH1D* hist = new TH1D("chi-2", " ;A;f(A)", nBinsX, xMin, xMax);
    TH1D* llr_hist = new TH1D("llr", " ;A;f(A)", nBinsX, xMin, xMax);
    double A_min = -1.;
    double min_chi2 = 1e6; 

    double A_min_llr = -1;
    double min_llr = 1e6;

    for (int i = 1; i <= nBinsX; ++i) {
            double A = hist->GetXaxis()->GetBinCenter(i);
            double param_set[1]={A};
            TH1D *alt_a = new TH1D("altA", "Alt A; cluster charge [p.e]; Number of events", 40, 1000., 4500.);
            for(int j=0; j<thru_muon_mc_features.cluster_pe.size(); j++){
                double thru_mu_cluster_pe = thru_muon_mc_features.cluster_pe.at(j);
                double thru_mu_cluster_a = qprim(&thru_mu_cluster_pe,param_set);
                alt_a->Fill(thru_mu_cluster_a);
            }

            int n_thru_muon_data = thru_muon_data_histograms.cl_pe_hist->Integral();
            thru_muon_histograms.cl_pe_hist->Scale((1./thru_muon_histograms.cl_pe_hist->Integral())*thru_muon_data_histograms.cl_pe_hist->Integral());
            alt_a->Scale((1./alt_a->Integral())*n_thru_muon_data);
            hist->SetBinContent(i, chi2(thru_muon_data_histograms.cl_pe_hist,alt_a));
            llr_hist->SetBinContent(i,llr(thru_muon_data_histograms.cl_pe_hist,alt_a));

            if(chi2(thru_muon_data_histograms.cl_pe_hist,alt_a)<min_chi2){
                min_chi2 = chi2(thru_muon_data_histograms.cl_pe_hist,alt_a);
                A_min = A;
            }

            if(llr(thru_muon_data_histograms.cl_pe_hist,alt_a) < min_llr){
                min_llr = llr(thru_muon_data_histograms.cl_pe_hist,alt_a);
                A_min_llr = A;
            }


         delete alt_a;
        
    }
    std::cout << "Minimum chi-2: " << min_chi2 << std::endl;
    std::cout << "Best parameter A (chi2): " << A_min << std::endl;
    std::cout << "Minimum llr: " << min_llr << std::endl;
    std::cout << "Best parameter A (llr): " << A_min_llr << std::endl;
    TCanvas* canvas = new TCanvas("canvas", "1D Histogram", 800, 600);
    gStyle->SetOptStat(0);

    TLegend *l = new TLegend(0.73, 0.7, 0.85, 0.8);
    l->AddEntry(hist, "chi-2");
    l->AddEntry(llr_hist, "LLR");

    hist->Draw("E");
    llr_hist->SetLineColor(kRed);
    llr_hist->Draw("E SAME");
        l->Draw("SAME");


    canvas->SaveAs("1D_LLR_chi2_comparison.png");
}
