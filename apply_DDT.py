#==============================================================================
# apply_DDT.py ----------------------------------------------------------------
#------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Sara Nabili -------------------------------------
#------------------------------------------------------------------------------
# Applies a DDT to a trained BDT model ----------------------------------------
#    (Designed Decorrelated Tagger, https://arxiv.org/pdf/1603.00027.pdf) -----
#------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
from time import strftime
from collections import OrderedDict
from common import create_DDT_map_dict, calculate_varDDT
import json

import numpy as np
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter
import mplhep as hep
hep.style.use("CMS") # CMS plot style

#disable tex formating while having problems
#plt.rcParams['text.usetex'] = False

np.random.seed(1001)

from common import logger, DATADIR, Columns, time_and_log, imgcat, set_mpl_fontsize, columns_to_numpy

#------------------------------------------------------------------------------
# User input global variables (to document options during running) ------------
#------------------------------------------------------------------------------

training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef', 
    ]

bkg_data_files = 'data/bkg_20240515/Summer20UL18/*.npz'
sig_data_files = 'data/signal/*mDark-10_rinv-0p3*.npz'

model_file = 'models/svjbdt_Feb28_lowmass_iterative_qcdtt_100p38.json'
ddt_map_file = 'models/ddt_Feb28_lowmass_iterative_qcdtt_100p38.json'

# Make sure this is what you want set for the specific ddt
lumi = 14026.948 + 7044.413  # PreHEM RunII 2018

# bdt cut to use when plotting signal mt
sig_bdt_cut = 0.9

# Choose the BDT cut values that you want to make for the DDT
# or that are in the DDT you are loading
bdt_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95] 
#bdt_cuts = [0.5, 0.7]
#bdt_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Choose what you want to plot
plot_2D_DDT_map = True
plot_bkg_scores_mt = True
plot_fom_significance = True
plot_sig_mt_single_BDT = True
plot_one_sig_mt_many_bdt = True

#------------------------------------------------------------------------------
# User defined functions ------------------------------------------------------
#------------------------------------------------------------------------------

def FOM(s, b):
    FOM = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
    return FOM


#------------------------------------------------------------------------------
# The Main Function -----------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    set_mpl_fontsize(18, 22, 26)

    # add ddt necessary variables in addition to the bdt input features
    all_features = training_features + ['pt', 'mt', 'rho', 'rt'] # rho is an important variable for applying the decorrelation

    # Grab the bkg data
    bkg_cols = [Columns.load(f) for f in glob.glob(bkg_data_files)]

    # Grab all the features from the files necessary for making the ddt
    bkg = []
    bkg_weight = []
    for col in bkg_cols:  
        # Grab the input features and weights
        bkg.append(col.to_numpy(all_features))
        bkg_weight.append(col.xs / col.cutflow['raw'] * lumi * col.arrays['puweight'])
    bkg = np.concatenate(bkg)
    X = bkg
    bkg_weight = np.concatenate(bkg_weight)

    # grab rt and apply signal region cuts
    rt = X[:,-1]
    X = X[:,:-1]
    X = X[rt > 1.18]
    bkg_weight = bkg_weight[rt > 1.18]
    rt = rt[rt > 1.18]

    # grab rho
    rho = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab mT
    mT = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab pT
    pT = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # _____________________________________________
    # Open the trained models and get the scores

    bkg_score = {}

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_file)
    with time_and_log(f'Calculating xgboost scores for {model_file}...'):
        bkg_score = xgb_model.predict_proba(X)[:,1]

    # _____________________________________________
    # Calculate bkg efficiencies for the DDT

    print("Calculating bkg efficiencies for various bdt cut values")
    bkg_eff=[]
    bkg_percents=[]
    bkg_Hist={}
    i = 0
    bkg_Hist_nobdt=np.histogram(bkg_score[bkg_score>0.0],weights=bkg_weight[bkg_score>0.0]*len(bkg_score))
    for bdt_cut in bdt_cuts: 
        bkg_Hist[i]=np.histogram(bkg_score[bkg_score>bdt_cut],weights=bkg_weight[bkg_score>bdt_cut]*len(bkg_score))
        bkg_eff.append(sum(bkg_Hist[i][0])/sum(bkg_Hist_nobdt[0]))
        bkg_percents.append(sum(bkg_Hist[i][0])/sum(bkg_Hist_nobdt[0])*100)
        i += 1

    # _____________________________________________
    # Create the DDT 2D map

    # Only make the map if it doesn't exist
    if not osp.exists(ddt_map_file):
        create_DDT_map_dict(mT, pT, rho, bkg_score, bkg_weight, bkg_percents, bdt_cuts, ddt_map_file)
    else: print("The DDT has already been calculated, please change the name if you want to remake the ddt map")

    # Load the dictionary from the json file
    with open(ddt_map_file, 'r') as f:
        var_dict = json.load(f)

    if plot_2D_DDT_map == True :
        for key in var_dict.keys() :
 
            print("Plotting the 2D DDT maps")
 
            # Get the 2D DDT map and bin edges for the corresponding BDT cut (key)
            bdt_cut = key
            var_map_smooth, RHO_edges, PT_edges = var_dict[key]
            var_map_smooth = np.array(var_map_smooth)
            RHO_edges = np.array(RHO_edges)
            PT_edges = np.array(PT_edges)
 
            # Plot 2D map for rho-phi plane for each BDT cut
            plt.figure(figsize=(10, 8))
            hep.cms.label(rlabel="2018 (13 TeV)")
            plt.imshow(var_map_smooth.T, extent=[RHO_edges[0], RHO_edges[-1], PT_edges[0], PT_edges[-1]], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='DDT Map value')
            plt.xlabel('$ \\rho = \\ln(\\mathrm{m}^{2}/\\mathrm{p}_{\\mathrm{T}}^{2})$')
            plt.ylabel('$\\mathrm{p}_{\\mathrm{T}}$ [GeV]')
            plt.tight_layout()
            plt.savefig(f'plots/2D_map_cut_{bdt_cut}.png')  # Save the plot
            plt.savefig(f'plots/2D_map_cut_{bdt_cut}.pdf')  # Save the plot
            plt.close()

    # _____________________________________________
    # Apply the DDT to the background

    if plot_bkg_scores_mt == True :
        print("Applying the DDT background")
        BKG_score_ddt = []
        for bdt_cut in bdt_cuts:
            BKG_score_ddt.append(calculate_varDDT(mT, pT, rho, bkg_score, bkg_weight, bdt_cut, ddt_map_file))
 
        # Plot histograms for the DDT scores for different BDT cuts
        print("Making background plots")
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
        for cuts, scores in zip(bdt_cuts, BKG_score_ddt):
            if cuts != 0.7: alpha = 0.3
            else: alpha = 1.0
            ax.hist(scores, bins=50, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
        #ax.set_title('DDT Score Distribution for Different Cuts')
        ax.set_xlabel('BKG_score_ddt')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/DDT_score.png')
        plt.savefig('plots/DDT_score.pdf')
 
        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
        for cuts, scores in zip(bdt_cuts, BKG_score_ddt):
            score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(mT[score_mask], bins=50, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})')
 
        #ax.set_title('mT for Different DDT(BDT) Thresholds')
        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/bkg_events_vs_mT.png')
        plt.savefig('plots/bkg_events_vs_mT.pdf')
        
        # log scale it
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('plots/log_bkg_events_vs_mT.png')
        plt.savefig('plots/log_bkg_events_vs_mT.pdf')
 
        # Do it again normalized to unit area
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
        for cuts, scores in zip(bdt_cuts, BKG_score_ddt):
            score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(mT[score_mask], bins=50, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})', density=True)
 
        #ax.set_title('mT for Different DDT(BDT) Thresholds')
        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/norm_bkg_events_vs_mT.png')
        plt.savefig('plots/norm_bkg_events_vs_mT.pdf')
        plt.close()
 
        print(BKG_score_ddt) 

    # _____________________________________________
    # Create Significance Plots
    if plot_fom_significance == True :

        # Grab the signal data
        sig_cols = [Columns.load(f) for f in glob.glob(sig_data_files)]
 
        # Prepare a figure
        fig = plt.figure(figsize=(10, 7))
        hep.cms.label(rlabel="2018 (13 TeV)")
        ax = fig.gca()

        best_bdt_cuts = [] #store the best bdt values
        # Iterate over the variables in the 'con' dictionary
        for sig_col in sig_cols:
            mz = sig_col.metadata['mz']
            #if mz not in [300, 350, 450, 550]:
            #  continue
            s = f'bsvj_{mz:d}_10_0.3'
         
            # Grab the input features and weights
            sig_X = sig_col.to_numpy(all_features)
            sig_weight = sig_col.xs / sig_col.cutflow['raw'] * lumi * sig_col.arrays['puweight']
      
            # Apply the signal region
            sig_rt = sig_X[:,-1]
            sig_X = sig_X[sig_rt > 1.18]
            sig_weight = sig_weight[sig_rt > 1.18]
            print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)
  
            # grab rt (after cut)
            sig_rt = sig_rt[sig_rt>1.18]
            sig_X = sig_X[:,:-1]
      
            # grab rho
            sig_rho = sig_X[:,-1]
            sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
      
            # grab mT
            sig_mT = sig_X[:,-1]
            sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
      
            # grab pT
            sig_pT = sig_X[:,-1]
            sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
 
            # _____________________________________________
            # Open the trained models and get the scores
      
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_file)
            with time_and_log(f'Calculating xgboost scores for {model_file}...'):
                sig_score = xgb_model.predict_proba(sig_X)[:,1]
      
            # _____________________________________________
            # Apply the DDT and calculate FOM
            bkg_score_ddt = []
            print("Applying the DDT and calculate FOM")

            # Iterate over the bdt cut values
            fom = [] # to store the figure of merit values
            for bdt_cut in bdt_cuts:
             
                sig_score_ddt = calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_weight, bdt_cut, ddt_map_file)
                bkg_score_ddt = calculate_varDDT(mT, pT, rho, bkg_score, bkg_weight, bdt_cut, ddt_map_file)

                # Apply ddt scores
                sig_mT_ddt = sig_mT[sig_score_ddt > 0]
                bkg_mT_ddt = mT[bkg_score_ddt > 0]
                
                # Apply mT window 
                sig_mT_mask = (sig_mT_ddt > (mz - 100)) & (sig_mT_ddt < (mz + 100) )
                bkg_mT_mask = (bkg_mT_ddt > (mz - 100)) & (bkg_mT_ddt < (mz + 100) )
                sig_mT_ddt = sig_mT_ddt[sig_mT_mask]
                bkg_mT_ddt = bkg_mT_ddt[bkg_mT_mask]
                wsig_bdt_mT_wind = sig_weight[sig_score_ddt > 0][sig_mT_mask]
                wbkg_bdt_mT_wind = bkg_weight[bkg_score_ddt > 0][bkg_mT_mask]
 
                # Calculate the figure of merit values for this bdt cut
                S = sum(np.histogram(sig_mT_ddt, bins=50, weights=wsig_bdt_mT_wind)[0])
                B = sum(np.histogram(bkg_mT_ddt, bins=50, weights=wbkg_bdt_mT_wind)[0])
                print("mZ': ", mz, " S: ", S, "B: ", B)
                fom.append(FOM(S,B))
             
            # Find the cut value corresponding to the maximum figure of merit
            fom_array = np.array(fom)
            bdt_cuts_array = np.array(bdt_cuts)
            best_bdt_cuts.append([mz, bdt_cuts_array[np.where(fom_array == max(fom_array))[0]][0]])
            print(best_bdt_cuts)
             
            # Plot the histogram of the metric variable range
            #arr = ax.hist(bdt_cut_values, bins=func.bin_cuts(bdt_cut_values), weights=fom, label=s, linewidth=2, histtype='step')
            if mz == 300: alpha = 1.0
            else: alpha = 0.3
            arr = ax.plot(bdt_cuts,fom, marker='o', label=f"m(Z') {mz}", alpha=alpha)  
            
        # grab labels that will go into the legend
        handles, labels = ax.get_legend_handles_labels()
 
        # Convert last part of labels to integers, handle errors if conversion is not possible
        masses = []
        for label in labels:
            try:
                mass = int(label.split()[-1])  # Assuming mass is the last word in the label
            except ValueError:
                mass = float('inf')  # If conversion fails, assign a large number
            masses.append(mass)

        # Sort legend items by mass (in descending order)
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: masses[labels.index(t[0])], reverse=True))

        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))          
        ax.ticklabel_format(style='sci', axis='x')
        ax.set_ylabel('FoM')
        ax.set_xlabel('BDT cut value')
        outfile = f'plots/metrics_bdt_FOM'
        print(outfile, ".pdf and .png")
        
        # Save the plot as a PDF and png file
        plt.savefig(outfile+".pdf", bbox_inches='tight')
        plt.savefig(outfile+".png", bbox_inches='tight')
        plt.close()

        # sort the best bdt cuts
        best_bdt_cuts = np.array(best_bdt_cuts)
        sort_indices = np.argsort(best_bdt_cuts[:,0]) #sort array to ascend by mz
        best_bdt_cuts = best_bdt_cuts[sort_indices]

        # Find optimal cut
        mask = (best_bdt_cuts[:,0] >= 200) & (best_bdt_cuts[:,0] <= 400)
        selected_values = best_bdt_cuts[mask, 1]
        average = np.mean(selected_values)

        # plot the best bdt cuts
        fig = plt.figure(figsize=(10, 7))
        hep.cms.label(rlabel="2018 (13 TeV)")
        ax = fig.gca()
        ax.plot(best_bdt_cuts[:,0], best_bdt_cuts[:,1], marker='o')
        ax.text(0.05, 0.95, f'Optimal Cut: {average:.2f}', transform=ax.transAxes, verticalalignment='top')
        ax.ticklabel_format(style='sci', axis='x')
        ax.set_ylabel('Best BDT Cut Value')
        ax.set_xlabel("m(Z')")
        plt.savefig("plots/best_bdt_cuts.pdf", bbox_inches='tight')
        plt.savefig("plots/best_bdt_cuts.png", bbox_inches='tight')
        plt.close()

        

    # _____________________________________________
    # Apply the DDT to different signals for one BDT cut value

    if plot_sig_mt_single_BDT == True :

        # grab signal data
        sig_cols = [Columns.load(f) for f in glob.glob(sig_data_files)]

        # make plotting objects
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
 
        # Loop over the mZ values
        for sig_col in sig_cols:
            mz = sig_col.metadata['mz']
         
            # Signal Column 
            sig_X = sig_col.to_numpy(all_features)
            sig_weight = sig_col.xs / sig_col.cutflow['raw'] * lumi * sig_col.arrays['puweight']
  
            # Apply the signal region
            sig_rt = sig_X[:,-1]
            sig_X = sig_X[sig_rt > 1.18]
            sig_weight = sig_weight[sig_rt > 1.18]
            print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)
 
            # grab rt (after cut)
            sig_rt = sig_rt[sig_rt>1.18]
            sig_X = sig_X[:,:-1]
  
            # grab rho
            sig_rho = sig_X[:,-1]
            sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
  
            # grab mT
            sig_mT = sig_X[:,-1]
            sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
  
            # grab pT
            sig_pT = sig_X[:,-1]
            sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
  
            # _____________________________________________
            # Open the trained models and get the scores
  
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_file)
            with time_and_log(f'Calculating xgboost scores for {model_file}...'):
                sig_score = xgb_model.predict_proba(sig_X)[:,1]
  
            # _____________________________________________
            # Apply the DDT  to the signal
            print("Applying the DDT signal")
            sig_score_ddt = calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_weight, sig_bdt_cut, ddt_map_file)
 
            # Make mT distributions
            if mz != 300: alpha = 0.3
            else: alpha = 1.0
            score_mask = sig_score_ddt > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(sig_mT[score_mask], weights=sig_weight[score_mask], bins=50, range=(180,650), histtype='step', label=f"m(Z')={mz}", alpha=alpha)
 
        #ax.set_title('mT for Different DDT(BDT) Thresholds')
        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/sig_events_vs_mT.png')
        plt.savefig('plots/sig_events_vs_mT.pdf')
        
        # log scale it
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('plots/log_sig_events_vs_mT.png')
        plt.savefig('plots/log_sig_events_vs_mT.pdf')
        plt.close()

    # _____________________________________________
    # Apply the DDT to one signals for different BDT cut values

    if plot_one_sig_mt_many_bdt == True :

        # grab signal data
        sig_cols = [Columns.load(f) for f in glob.glob(sig_data_files)]

        # Loop over the mZ values and only grab the mZ = 300 value
        sig_col = None
        for col in sig_cols:
            mz = col.metadata['mz']
            if mz == 300 : sig_col = col
        if sig_col == None : print("The mZ' 300 file doesn't exist, please make sure you provide it in order to make the one sig plots")

        # make plotting objects
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
 
        mz = sig_col.metadata['mz']
        
        # Signal Column 
        sig_X = sig_col.to_numpy(all_features)
        sig_weight = sig_col.xs / sig_col.cutflow['raw'] * lumi * sig_col.arrays['puweight']
  
        # Apply the signal region
        sig_rt = sig_X[:,-1]
        sig_X = sig_X[sig_rt > 1.18]
        sig_weight = sig_weight[sig_rt > 1.18]
        print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)
 
        # grab rt (after cut)
        sig_rt = sig_rt[sig_rt>1.18]
        sig_X = sig_X[:,:-1]
  
        # grab rho
        sig_rho = sig_X[:,-1]
        sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
  
        # grab mT
        sig_mT = sig_X[:,-1]
        sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
  
        # grab pT
        sig_pT = sig_X[:,-1]
        sig_X = sig_X[:,:-1] # remove it from X so that eventually it can be used for BDT scores
  
        # _____________________________________________
        # Open the trained models and get the scores
  
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_file)
        with time_and_log(f'Calculating xgboost scores for {model_file}...'):
            sig_score = xgb_model.predict_proba(sig_X)[:,1]
  
        # _____________________________________________
        # Apply the DDT  to the signal at all the cut values
        print("Applying the DDT signal")
        sig_score_ddt = []
        for bdt_cut in bdt_cuts :
            sig_score_ddt.append(calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_weight, bdt_cut, ddt_map_file) )

        # Plot histograms for the DDT scores for different BDT cuts
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
        for cuts, scores in zip(bdt_cuts, sig_score_ddt):
            if cuts != 0.7: alpha = 0.3
            else: alpha = 1.0
            ax.hist(scores, weights=sig_weight, bins=50, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
        #ax.set_title('DDT Score Distribution for Different Cuts')
        ax.set_xlabel('DDT(BDT)')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/DDT_sig_score_mz300.png')
        plt.savefig('plots/DDT_sig_score_mz300.pdf')
        plt.close()
 
        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
        for cuts, scores in zip(bdt_cuts, sig_score_ddt):
            score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(sig_mT[score_mask], weights=sig_weight[score_mask], bins=25, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})')
 
        #ax.set_title('mT for Different DDT(BDT) Thresholds')
        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/sig_mz300_events_vs_mT.png')
        plt.savefig('plots/sig_mz300_events_vs_mT.pdf')
        
        # log scale it
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('plots/log_sig_mz300_events_vs_mT.png')
        plt.savefig('plots/log_sig_mz300_events_vs_mT.pdf')
 
        # Do it again normalized to unit area
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier 
        hep.cms.label(rlabel="2018 (13 TeV)")
        for cuts, scores in zip(bdt_cuts, sig_score_ddt):
            score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(sig_mT[score_mask], weights=sig_weight[score_mask], bins=25, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})', density=True)
 
        #ax.set_title('mT for Different DDT(BDT) Thresholds')
        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/norm_sig_mz300_events_vs_mT.png')
        plt.savefig('plots/norm_sig_mz300_events_vs_mT.pdf')
        plt.close()

if __name__ == '__main__':
    main()
