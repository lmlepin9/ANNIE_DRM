#include <vector> 


struct sample_features{
    std::vector<Float_t> cluster_time;
    std::vector<Float_t> cluster_pe;
    std::vector<Float_t> cluster_qb;
    std::vector<int> cluster_nhits;
    std::vector<Float_t> cluster_ds_pe;
    std::vector<Float_t> cluster_us_pe;
};

void read_and_fill(TFile *f, std::string sample, sample_features& data){
    TTree* temp_tree = (TTree*) f->Get(sample.c_str());
    std::cout << "Number of entries in " << sample << " tree: " << temp_tree->GetEntries() << std::endl;

    Float_t cl_time;
    Float_t cl_pe;
    Float_t cl_qb;
    temp_tree->SetBranchAddress("cluster_time",&cl_time);
    temp_tree->SetBranchAddress("cluster_pe",&cl_pe);
    temp_tree->SetBranchAddress("cluster_Qb",&cl_qb);

    for(int i=0; i < temp_tree->GetEntries(); i ++){
        temp_tree->GetEntry(i);
        data.cluster_time.push_back(cl_time);
        data.cluster_pe.push_back(cl_pe);
        data.cluster_qb.push_back(cl_qb);

    }

}


void MakeRatioPlot(TH1D *data_hist, TH1D* mc_hist, std::string fig_name){
    rootlogon();
    gStyle->SetOptStat(0);

    data_hist->SetLineColor(kBlack);
    mc_hist->SetLineColor(kRed);




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

