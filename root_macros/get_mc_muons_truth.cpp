#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <TApplication.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TMath.h>

// ANNIE plot style 
#include "rootlogon.hpp"

using namespace std;

int get_mc_muons_truth() {

    rootlogon();
    gStyle->SetOptStat(0);

    ifstream infile("./samples/thru_genie_muons.txt");

    if (!infile) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    vector<double> vertex_x, vertex_y, vertex_z;
    vector<double> momentum_x, momentum_y, momentum_z;
    vector<double> energy;

    double vx, vy, vz, mx, my, mz, e;

    while (infile >> vx >> vy >> vz >> mx >> my >> mz >> e) {
        vertex_x.push_back(vx);
        vertex_y.push_back(vy);
        vertex_z.push_back(vz);
        momentum_x.push_back(mx);
        momentum_y.push_back(my);
        momentum_z.push_back(mz);
        energy.push_back(e);
    }

    vector<double> cos_mu;
    for(int i =0; i < momentum_x.size(); i++){
        double temp_cos = (momentum_z.at(i)*1.)/TMath::Sqrt(
            TMath::Power(momentum_x.at(i),2) +
            TMath::Power(momentum_y.at(i),2) +
            TMath::Power(momentum_z.at(i),2)
        );

        cos_mu.push_back(temp_cos);
    }

    auto print_min_max = [](const vector<double>& data, const string& label) {
        double min_val = numeric_limits<double>::max();
        double max_val = numeric_limits<double>::lowest();
        for (double val : data) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        cout << label << " -> Min: " << min_val << ", Max: " << max_val << endl;
    };

    print_min_max(vertex_x, "vertex_x");
    print_min_max(vertex_y, "vertex_y");
    print_min_max(vertex_z, "vertex_z");
    print_min_max(momentum_x, "momentum_x");
    print_min_max(momentum_y, "momentum_y");
    print_min_max(momentum_z, "momentum_z");
    print_min_max(cos_mu, "cosine_muon");
    print_min_max(energy, "energy");

    auto fill_draw_save = [](const vector<double>& data, const string& name, int bins) {
        double min_val = *min_element(data.begin(), data.end());
        double max_val = *max_element(data.begin(), data.end());
        TH1D* hist = new TH1D(name.c_str(), "Through going muons", bins, min_val, max_val);
        hist->SetLineColor(kRed);
        hist->GetXaxis()->SetTitle(name.c_str());
        hist->GetYaxis()->SetTitle("Counts");
        for (double val : data) hist->Fill(val);

        TCanvas* canvas = new TCanvas(("c_" + name).c_str(), name.c_str(), 800, 600);
        hist->Draw();
        SimulationSide();
        canvas->SaveAs(("./figures/mc_thru_muons_" + name + ".pdf").c_str());
    };

    fill_draw_save(vertex_x, "Vertex_x [cm]", 100);
    fill_draw_save(vertex_y, "Vertex_y [cm]", 100);
    fill_draw_save(vertex_z, "Vertex_z [cm]", 100);
    fill_draw_save(momentum_x, "momentum_x", 100);
    fill_draw_save(momentum_y, "momentum_y", 100);
    fill_draw_save(momentum_z, "momentum_z", 100);
    fill_draw_save(cos_mu, "cos(#theta_{#mu})", 100);
    fill_draw_save(energy, "Energy [MeV]", 100);


    return 0;
}