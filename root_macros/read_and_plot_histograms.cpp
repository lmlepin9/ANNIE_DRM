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


void MakeRatioPlot(TH1D *data_hist, TH1D* mc_hist, std::string fig_name){


    double max_data = data_hist->GetMaximum();
    double max_mc = mc_hist->GetMaximum();
    double y_max = std::max(max_data, max_mc);


    TCanvas *c = new TCanvas("c", "MC/Data Comparison", 800,800);
    c->Divide(1,2);

    // Create two pads, one for histograms other for the ratio
    TPad *pad1 = new TPad("pad1", "Hist plot", 0, 0.32, 1, 1.0);
    pad1->SetBottomMargin(0.02); // No x-axis labels
    pad1->Draw();
    pad1->cd();

    mc_hist->SetMaximum(1.1*y_max);
    mc_hist->GetXaxis()->SetLabelSize(0);
    mc_hist->Draw("HIST");
    data_hist->Draw("E1 SAME");
    TLegend *l = new TLegend(0.73, 0.7, 0.85, 0.8);
    l->AddEntry(data_hist, "Data");
    l->AddEntry(mc_hist, "MC");
    PreliminarySide();
    l->Draw("SAME");

    c->cd();
    TPad *pad2 = new TPad("pad2", "Ratio plot", 0, 0.05, 1, 0.3);
    pad2->SetTopMargin(0.03);
    pad2->SetBottomMargin(0.3);
    pad2->Draw();
    pad2->cd();

    TH1D *ratio = (TH1D*)mc_hist->Clone("ratio");
    ratio->Divide(data_hist);
    ratio->SetStats(0);
    ratio->SetTitle("");
    ratio->SetLineColor(kBlack);

    ratio->GetYaxis()->SetTitle("MC / Data");
    ratio->GetYaxis()->SetNdivisions(505);
    ratio->GetYaxis()->SetTitleSize(0.12);
    ratio->GetYaxis()->SetTitleOffset(0.4);
    ratio->GetYaxis()->SetLabelSize(0.10);
    ratio->GetXaxis()->SetTitleSize(0.12);
    ratio->GetXaxis()->SetLabelSize(0.10);

    ratio->SetMinimum(0.0);
    ratio->SetMaximum(2.0);
    ratio->Draw("HIST");

    // Draw a line at y=1
    TLine *line = new TLine(ratio->GetXaxis()->GetXmin(), 1.0,ratio->GetXaxis()->GetXmax(), 1.0);
    line->SetLineStyle(2);
    line->Draw("SAME");

    c->SaveAs(fig_name.c_str());

    // Recall that objects created with new don't go away after the scope ends!! 
    delete c; 
}



void read_and_plot_histograms(){
    rootlogon();
    gStyle->SetOptStat(0);

    
    TFile *f = TFile::Open("mc_legacy_thru_muon_cpe_hists.root");
    TH1D *data_hist = (TH1D*) f->Get("cpe");
    TH1D *mc_hist = (TH1D*) f->Get("cpe_mc");

    TFile *f1 = TFile::Open("world_thru_muon_cpe_hists.root");
    TH1D *mc_world_hist = (TH1D*)f1->Get("cpe_mc");
    mc_world_hist->SetLineColor(kBlue);

    TFile *f2 = TFile::Open("world_tilt_thru_muon_cpe_hists.root");
    TH1D *mc_tilt_hist = (TH1D*)f2->Get("cpe_mc");
    mc_tilt_hist->SetLineColor(kGreen);

    TFile *f3 = TFile::Open("world_tilt_shift_thru_muon_cpe_hists.root");
    TH1D *mc_ts_hist = (TH1D*)f3->Get("cpe_mc");
    mc_ts_hist->SetLineColor(6);

    TCanvas *c = new TCanvas("c", "MC/Data Comparison", 800,800);
    mc_hist->GetYaxis()->SetTitleOffset(2.);
    mc_hist->Draw("HIST");
    mc_world_hist->Draw("HIST SAME");
    mc_tilt_hist->Draw("HIST SAME");
    mc_ts_hist->Draw("HIST SAME");
    data_hist->Draw("E SAME");

    TLegend *l = new TLegend(0.7, 0.6, 0.88, 0.8);
    l->AddEntry(data_hist, "Data");
    l->AddEntry(mc_hist, "MC (Thru-mu)");
    l->AddEntry(mc_world_hist, "MC world volume (WV)");
    l->AddEntry(mc_tilt_hist, "MC (WV, tilted)");
    l->AddEntry(mc_ts_hist, "MC (WV,tilt+shift)");
    l->Draw("SAME");

    c->SaveAs("./figures/thru_muon_comparison.png");
    

}