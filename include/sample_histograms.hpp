class sample_histograms{
    private:
       std::string flabel;
       int fn_bins;
       float fpe_low;
       float fpe_up;
       float fcb_low;
       float fcb_up;

   public:
    sample_histograms(std::string label, 
                      int n_bins,
                      float pe_low,
                      float pe_up,
                      float cb_low,
                      float cb_up):flabel(label),fn_bins(n_bins),fpe_low(pe_low),fpe_up(pe_up),fcb_low(cb_low),fcb_up(cb_up) 
                      { }

    std::string cl_pe_hist_name = "pe_hist_" + flabel;
    std::string cl_qb_hist_name = "qb_hist_" + flabel;
    TH1D *cl_pe_hist = new TH1D(cl_pe_hist_name.c_str(),"cluster pe; cluster charge [p.e]; number of entries",fn_bins,fpe_low,fpe_up);
    TH1D *cl_qb_hist = new TH1D(cl_qb_hist_name.c_str(),"cluster Qb; charge balance [ns]; number of entries",fn_bins,fcb_low,fcb_up);
    void MakePlots(bool normalize=true);


};

void sample_histograms::MakePlots(bool normalize=true){
    rootlogon();
    gStyle->SetOptStat(0);

    /* Make cluster pe hist*/
    std::string cpe_c_name = "Cluster charge comparison - " + flabel;
    TCanvas *c = new TCanvas("c",cpe_c_name.c_str(), 800,600);
    if(normalize){cl_pe_hist->Scale(1./cl_pe_hist->Integral());}
    cl_pe_hist->Draw("HIST");
    std::string file_name = "./figures/cluster_charge_"+flabel+".pdf";
    c->SaveAs(file_name.c_str());
    delete c;
}