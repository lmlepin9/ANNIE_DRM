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



Double_t qprim(Double_t *x, Double_t *par){
    Double_t xx = x[0];
    return (xx + par[0]*xx + par[1]*TMath::Power(xx,2));
}


class MyHistoFitter{
    public:
        static TH1D *data_hist;
        static std::vector<Float_t> mc;
        static bool is_llr; 

        static void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag) {
            f = 0;
            TH1D *alternative_hist = new TH1D("alternative_hist", "Alt A; cluster charge [p.e]; Number of events", 40, 1000., 4500.);
            for(int i=0; i<mc.size(); i++){
                double cluster_pe = mc.at(i);
                double mod_cluster_pe = (cluster_pe + par[0]*cluster_pe);
                alternative_hist->Fill(mod_cluster_pe);
            }

            // Scale to compare 
            alternative_hist->Scale((1./alternative_hist->Integral())*data_hist->Integral());
            for(int i=0; i < data_hist->GetNbinsX(); i++){

                double mu_i = alternative_hist->GetBinContent(i+1);
                double M_i = data_hist->GetBinContent(i+1);


                if(!is_llr){

                    // Neyman chi-2
                    f+=(TMath::Power((mu_i - M_i),2)/M_i);
                }
                else{

                    // Poisson chi-2 (directly from LLR)
                    f+=2*(mu_i - M_i + M_i*TMath::Log(M_i/mu_i));
                }
            }
            delete alternative_hist;
        }

};



TH1D *MyHistoFitter::data_hist = nullptr; 
std::vector<Float_t> MyHistoFitter::mc;
bool MyHistoFitter::is_llr;



void fit_energy_model(){

    TCanvas *c = new TCanvas("c", "Energy response function",800,600);



    double q_val_A[4] = {1., 1.5, 2., 2.5};
    double q_val_B[4] = {7., 7.5, 8., 8.5};
    int colors[4] = {1, 2, 3, 4};

    TF1 *f1 = new TF1("f1", qprim, 0., 8000., 2);
    f1->SetParameters(q_val_A[0],q_val_B[0]);
    f1->SetLineColor(kRed);
    TF1 *f2 = new TF1("f2", qprim, 0., 8000., 2);
    f2->SetLineColor(kBlue);
    f2->SetParameters(q_val_A[1],q_val_B[1]);
    f1->Draw();
    f2->Draw("SAME");


    c->SaveAs("./figures/response_function_drawing.pdf");

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

    TH1D *alt_a = new TH1D("altA", "Alt A; cluster charge [p.e]; Number of events", 40, 1000., 4500.);
    TH1D *alt_b = new TH1D("altB", "Alt B; cluster charge [p.e]; Number of events", 40, 1000., 4500.);

    double param_set_A[2]={0.8};
    double param_set_B[2]={ 0.0224138};

    for(int i=0; i<thru_muon_mc_features.cluster_pe.size(); i++){
        double thru_mu_cluster_pe = thru_muon_mc_features.cluster_pe.at(i);
        double thru_mu_cluster_a = qprim(&thru_mu_cluster_pe,param_set_A);
        double thru_mu_cluster_b = qprim(&thru_mu_cluster_pe,param_set_B);
        thru_muon_histograms.cl_pe_hist->Fill(thru_mu_cluster_pe);
        alt_a->Fill(thru_mu_cluster_a);
        alt_b->Fill(thru_mu_cluster_b);
    }

    int n_thru_muon_data = thru_muon_data_histograms.cl_pe_hist->Integral();
    thru_muon_histograms.cl_pe_hist->Scale((1./thru_muon_histograms.cl_pe_hist->Integral())*thru_muon_data_histograms.cl_pe_hist->Integral());
    alt_a->Scale((1./alt_a->Integral())*n_thru_muon_data);
    alt_b->Scale((1./alt_b->Integral())*n_thru_muon_data);

    TCanvas *c1 = new TCanvas("c1", "Alt hists", 800,600);
    thru_muon_histograms.cl_pe_hist->SetLineColor(kBlack);
    thru_muon_data_histograms.cl_pe_hist->SetLineColor(kBlack);
    thru_muon_data_histograms.cl_pe_hist->Draw("E1");
    thru_muon_histograms.cl_pe_hist->Draw("HIST SAME");
    alt_a->SetLineColor(kRed);
    alt_a->Draw("HIST SAME");
    alt_b->SetLineColor(kBlue);
    alt_b->Draw("HIST SAME");
    c1->SaveAs("./figures/alternative_model_histograms.pdf");

    std::cout << "Calculating chi-2 for alternative models: " << std::endl;
    std::cout << "Nominal chi2: " << chi2(thru_muon_data_histograms.cl_pe_hist, thru_muon_histograms.cl_pe_hist) << std::endl;
    std::cout << "Alt A chi2: " << chi2(thru_muon_data_histograms.cl_pe_hist, alt_a) << std::endl;
    std::cout << "Alt B chi2: " << chi2(thru_muon_data_histograms.cl_pe_hist, alt_b) << std::endl;

    MyHistoFitter::data_hist = thru_muon_data_histograms.cl_pe_hist;
    MyHistoFitter::mc = thru_muon_mc_features.cluster_pe;
    MyHistoFitter::is_llr = false;

    std::cout << "Sanity check, printing the integral of the data hist: " << MyHistoFitter::data_hist->Integral() << std::endl;
    std::cout << "Sanity check, printing the size of the mc array: " << MyHistoFitter::mc.size() << std::endl;


    // Let's try the fitting with two parameters here:
    TMinuit minuit(1);
    minuit.SetFCN(MyHistoFitter::fcn);

    double arglist[10];
    int ierflg=0;
    arglist[0] = 1;
    minuit.mnexcm("SET ERR", arglist, 1, ierflg);

    arglist[0]=1;
    minuit.mnexcm("SET STR", arglist, 1, ierflg);

    Double_t vstart[1] = {0.017};
    Double_t step[1] = {1e-5};
    Double_t low_lim[1] = {0.016};
    Double_t up_lim[1] = {0.018};

    minuit.mnparm(0,"A",vstart[0],step[0],low_lim[0],up_lim[0],ierflg);

    //arglist[0] = 2000;
    //minuit.mnexcm("MIGRAD",arglist,2,ierflg);
    arglist[0] = 0; // 0 = all parameters
    minuit.mnexcm("MINOS",arglist,1,ierflg);

    double a, b, ea, eb;
    minuit.GetParameter(0, a, ea);

    std::cout << "Fit results:\n";
    std::cout << "A = " << a << " ± " << ea << "\n";


    // Retrieve parameters and errors
    double eaLow, eaHigh, eaparab, eagcc, ebLow, ebHigh, ebparab, ebgcc;


    minuit.mnerrs(0, eaLow, eaHigh,eaparab,eagcc);


    std::cout << "Fit results with MINOS errors:\n";
    std::cout << "A = " << a << " ± " << ea << " [" << eaLow << ", " << eaHigh << "]\n";


  


}