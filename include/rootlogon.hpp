// To set this as default, you need a .rootrc file in your home directory,

// containing the following line:

// Rint.Logon: /full/path/to/rootlogon.C

// Nice trick, taken from: https://cdcvs.fnal.gov/redmine/attachments/41274/rootlogon.C

#include "TColor.h"

#include "TH1.h"

#include "TLatex.h"

#include "TROOT.h"

#include "TStyle.h"

void rootlogon()

{

  printf("Using ANNIE default plot style \n");

  // Defaults to classic style, but that's OK, we can fix it

  TStyle* annieStyle = new TStyle("annieStyle", "ANNIE Style");

  // Centre title

  annieStyle->SetTitleAlign(22);

  annieStyle->SetTitleX(.5);

  annieStyle->SetTitleY(.95);

  annieStyle->SetTitleBorderSize(0);

  // No info box

  annieStyle->SetOptStat(0);

  //set the background color to white

  annieStyle->SetFillColor(10);

  annieStyle->SetFrameFillColor(10);

  annieStyle->SetCanvasColor(10);

  annieStyle->SetPadColor(10);

  annieStyle->SetTitleFillColor(0);

  annieStyle->SetStatColor(10);

  // Don't put a colored frame around the plots

  annieStyle->SetFrameBorderMode(0);

  annieStyle->SetCanvasBorderMode(0);

  annieStyle->SetPadBorderMode(0);

  // Set the default line color for a fit function to be red

  annieStyle->SetFuncColor(kRed);

  // Marker settings

  //  annieStyle->SetMarkerStyle(kFullCircle);

  // No border on legends

  annieStyle->SetLegendBorderSize(0);

  // Disabled for violating ANNIE style guidelines

  // Scientific notation on axes

  //  TGaxis::SetMaxDigits(3);

  // Axis titles

  annieStyle->SetTitleSize(.05, "xyz");

  annieStyle->SetTitleOffset(0.9, "xyz");

  // More space for y-axis to avoid clashing with big numbers

  annieStyle->SetTitleOffset(1.0, "y");

  // This applies the same settings to the overall plot title

  annieStyle->SetTitleSize(.06, "");

  annieStyle->SetTitleOffset(.9, "");

  // Axis labels (numbering)

  annieStyle->SetLabelSize(.04, "xyz");

  annieStyle->SetLabelOffset(.005, "xyz");

  // Prevent ROOT from occasionally automatically zero-suppressing

  annieStyle->SetHistMinimumZero();

  // Thicker lines

  annieStyle->SetHistLineWidth(2);

  annieStyle->SetFrameLineWidth(2);

  annieStyle->SetFuncWidth(2);

  // Set the number of tick marks to show

  annieStyle->SetNdivisions(506, "xyz");

  // Set the tick mark style

  annieStyle->SetPadTickX(1);

  annieStyle->SetPadTickY(1);

  // Fonts

  const int kAnnieFont = 42;

  annieStyle->SetStatFont(kAnnieFont);

  annieStyle->SetLabelFont(kAnnieFont, "xyz");

  annieStyle->SetTitleFont(kAnnieFont, "xyz");

  annieStyle->SetTitleFont(kAnnieFont, ""); // Apply same setting to plot titles

  annieStyle->SetTextFont(kAnnieFont);

  annieStyle->SetLegendFont(kAnnieFont);

  // Get moodier colours for colz

  const Int_t NRGBs = 5;

  const Int_t NCont = 255;

  Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };

  Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };

  Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };

  Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };

  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);

  annieStyle->SetNumberContours(NCont);

  gROOT->SetStyle("annieStyle");

  // Uncomment this line if you want to force all plots loaded from files

  // to use this same style

  //gROOT->ForceStyle();

}

// Put a "ANNIE Preliminary" tag in the corner

void Preliminary(TString data="")

{

  TLatex* prelim = new TLatex(.9, .95, "ANNIE "+data+" Preliminary");

  prelim->SetTextColor(kBlue);

  prelim->SetNDC();

  prelim->SetTextSize(2/30.);

  prelim->SetTextAlign(32);

  prelim->Draw();

}

// Put a "ANNIE Preliminary" tag on the right

void PreliminarySide(TString data="")

{

  TLatex* prelim = new TLatex(.93, .9, "ANNIE "+data+" Preliminary");

  prelim->SetTextColor(kBlue);

  prelim->SetNDC();

  prelim->SetTextSize(2/30.);

  prelim->SetTextAngle(270);

  prelim->SetTextAlign(12);

  prelim->Draw();

}

// Put a "ANNIE Simulation" tag in the corner

void Simulation()

{

  TLatex* prelim = new TLatex(.9, .95, "ANNIE Simulation");

  prelim->SetTextColor(kGray+1);

  prelim->SetNDC();

  prelim->SetTextSize(2/30.);

  prelim->SetTextAlign(32);

  prelim->Draw();

}

// Put a "ANNIE Simulation" tag on the right

void SimulationSide()

{

  TLatex* prelim = new TLatex(.93, .9, "ANNIE Simulation");

  prelim->SetTextColor(kGray+1);

  prelim->SetNDC();

  prelim->SetTextSize(2/30.);

  prelim->SetTextAngle(270);

  prelim->SetTextAlign(12);

  prelim->Draw();

}

void CenterTitles(TH1* histo)

{

  histo->GetXaxis()->CenterTitle();

  histo->GetYaxis()->CenterTitle();

  histo->GetZaxis()->CenterTitle();

}
