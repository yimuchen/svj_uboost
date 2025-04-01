#==============================================================================
# apply_DDT.py ----------------------------------------------------------------
#------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Yi-Mu Chen, Sara Nabili -------------------------
#------------------------------------------------------------------------------
# Applies a DDT to a trained BDT model ----------------------------------------
#    (Designed Decorrelated Tagger, https://arxiv.org/pdf/1603.00027.pdf) -----
#------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
from time import strftime
from collections import OrderedDict
from common import create_DDT_map_dict, calculate_varDDT, apply_cutbased
import json
import argparse

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
import svj_ntuple_processing as svj

np.random.seed(1001)

from common import read_training_features, logger, DATADIR, Columns, time_and_log, imgcat, set_mpl_fontsize, columns_to_numpy, apply_rt_signalregion, calc_bdt_scores

#------------------------------------------------------------------------------
# Global variables and user input arguments -----------------------------------
#------------------------------------------------------------------------------

training_features = []

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some inputs.')

    # Which analysis to run (BDT or cut-based)
    parser.add_argument('--analysis_type', default='BDT-based', choices=['BDT-based', 'cut-based'], help='Apply DDT to BDT or cut-based analysis')

    parser.add_argument('--bkg_files', default='data/bkg_20241030/Summer20UL*/QCD*.npz', help='Background data files (default is QCD only DDT)')
    parser.add_argument('--sig_files', default='data/sig_20241030/sig_*/*.npz', help='Signal data files')

    # BDT and ddt model
    parser.add_argument('--bdt_file', default='models/svjbdt_obj_rev_version.json', help='BDT model file')
    parser.add_argument('--ddt_map_file', default='models/ddt_AN_v5.json', help='DDT map file')

    parser.add_argument('--lumi', type=float, default=137600, help='Luminosity')

    # The default value of 0.65 was the optimal cut value determined. If the training or selection changes,
    # the value should be adapted accordingly
    parser.add_argument('--sig_bdt_cut', type=float, default=0.67, help='BDT cut for signal plotting (current optimal cut is 0.65)')

    # Choose the BDT cut values that you want to make for the DDT
    # or that are in the DDT you are loading
    # another common set of cuts is
    # bdt_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95]
    # and another common set for plots is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    parser.add_argument('--bdt_cuts', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95], help='List of BDT cuts')

    # Allowed plots: 2D DDT maps, Background Scores vs MT, FOM significance,
    #                Signal mt spectrums for one BDT working point,  one signal mt spectrum for many BDT working points
    allowed_plots = ['2D_DDT_map', 'bkg_scores_mt', 'fom_significance', 'sig_mt_single_BDT', 'one_sig_mt_many_bdt']

    # Create the parser and add the plot argument
    parser.add_argument('--plot', nargs='*', type=str, default=[], choices=allowed_plots, help='Plots to make')

    # Add verbosity level argument
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, help="Set verbosity level: 0 (silent), 1 (default), 2 (detailed)")

    return parser.parse_args()

#------------------------------------------------------------------------------
# User defined functions ------------------------------------------------------
#------------------------------------------------------------------------------

def FOM(s, b):
    '''
    The significance for signal vs. background derived from the CLs method
    '''
    FOM = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
    return FOM

def save_plot(plt, plt_name, flag_tight_layout=True, **kwargs):
    '''
    Save the plots, save some lines of code
    '''
    if flag_tight_layout == True: plt.tight_layout()
    plt.savefig(f'plots/{plt_name}.png', **kwargs)
    plt.savefig(f'plots/{plt_name}.pdf', **kwargs)

def bdt_ddt_inputs(col, lumi, all_features):

    # Storing column features
    X = []
    weight = []

    # Apply the signal region
    if isinstance(col, list):
        col = [apply_rt_signalregion(c) for c in col]
    else:
        col = apply_rt_signalregion(col)

    # Test if more than one column
    if isinstance(col,list):
        for icol in col:
            # Grab the input features and weights
            X.append(icol.to_numpy(all_features))
            weight.append(icol.xs / icol.cutflow['raw'] * lumi * icol.arrays['puweight'])
        X = np.concatenate(X)
        weight = np.concatenate(weight)

    # If only one column
    else:
        X = col.to_numpy(all_features)
        weight = col.xs / col.cutflow['raw'] * lumi * col.arrays['puweight']

    # grab rt
    rt = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab rho
    rho = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab mT
    mT = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab pT
    pT = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    return X, pT, mT, rho, weight


#------------------------------------------------------------------------------
# The Main Function -----------------------------------------------------------
#------------------------------------------------------------------------------

def main():

    # Parse arguments and take the results
    args = parse_arguments()

    ana_type = args.analysis_type
    bkg_files = args.bkg_files
    sig_files = args.sig_files
    model_file = args.bdt_file
    ddt_map_file = args.ddt_map_file
    lumi = args.lumi
    sig_bdt_cut = args.sig_bdt_cut
    bdt_cuts = args.bdt_cuts
    plots = args.plot
    verbosity = args.verbosity

    set_mpl_fontsize(18, 22, 26)




    #--------------------------------------------------------------------------
    # For the cut-based search ------------------------------------------------
    #--------------------------------------------------------------------------
    # Grab the bkg data
    bkg_cols = [Columns.load(f) for f in glob.glob(bkg_files)]

    features_common = ["pt", "mt", "rho", "rt"]
    ana_variant = {
        "cut-based": {
            "features": ["ecfm2b1"] + features_common,
            "inputs_to_primary": lambda x: x[:, 0],
            "primary_var_label": "$M_2^{(1)}$ $>$ ",
            "cut_values": [0.09]
        },
        "BDT-based": {
            "features": read_training_features(model_file) + features_common,
            "inputs_to_primary": lambda x:  calc_bdt_scores(x, model_file=model_file),
            "primary_var_label": "BDT $>$",
            "cut_values": bdt_cuts
        }
    }
    X, pT, mT, rho, bkg_weight = bdt_ddt_inputs(bkg_cols, lumi, ana_variant[ana_type]["features"])
    primary_var = ana_variant[ana_type]["inputs_to_primary"](X)
    bkg_eff=[]
    bkg_percents=[]
    bkg_Hist={}
    bkg_Hist_nobdt=np.histogram(primary_var[primary_var>0.0],weights=bkg_weight[primary_var>0.0]*len(primary_var))
    for i, cut_val in enumerate(ana_variant[ana_type]["cut_values"]):
        bkg_Hist[i]=np.histogram(primary_var[primary_var>cut_val],weights=bkg_weight[primary_var>cut_val]*len(primary_var))
        bkg_eff.append(sum(bkg_Hist[i][0])/sum(bkg_Hist_nobdt[0]))
        bkg_percents.append(sum(bkg_Hist[i][0])/sum(bkg_Hist_nobdt[0])*100)

    # _____________________________________________
    # Create the DDT 2D map

    # Only make the map if it doesn't exist
    if not osp.exists(ddt_map_file):
        create_DDT_map_dict(mT, pT, rho, primary_var, bkg_weight, bkg_percents, ana_variant[ana_type]["cut_values"], ddt_map_file)
    else: print("The DDT has already been calculated, please change the name if you want to remake the ddt map")

    # Load the dictionary from the json file
    with open(ddt_map_file, 'r') as f:
        var_dict = json.load(f)

    if '2D_DDT_map' in plots :
        for key in var_dict.keys() :

            if verbosity > 0 : print("Plotting the 2D DDT maps")

            # Get the 2D DDT map and bin edges for the corresponding BDT cut (key)
            var_map_smooth, RHO_edges, PT_edges = var_dict[key]
            var_map_smooth = np.array(var_map_smooth)
            RHO_edges = np.array(RHO_edges)
            PT_edges = np.array(PT_edges)

            # Plot 2D map for rho-phi plane for each BDT cut
            plt.figure(figsize=(10, 8))
            hep.cms.label(rlabel="(13 TeV)")
            plt.imshow(var_map_smooth.T, extent=[RHO_edges[0], RHO_edges[-1], PT_edges[0], PT_edges[-1]], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='DDT Map value')
            plt.xlabel('$ \\rho = \\ln(\\mathrm{m}^{2}/\\mathrm{p}_{\\mathrm{T}}^{2})$')
            plt.ylabel('$\\mathrm{p}_{\\mathrm{T}}$ [GeV]')
            save_plot(plt,f'2D_map_{ana_type}_{key}')
            plt.close()

    if 'bkg_scores_mt' in plots :
        if verbosity > 0 : print("Applying the DDT background")
        primary_var_ddt = []
        for cut_val in ana_variant[ana_type]["cut_values"]:
            primary_var_ddt.append(calculate_varDDT(mT, pT, rho, primary_var, bkg_weight, cut_val, ddt_map_file))

        # Plot histograms for the DDT scores for different BDT cuts
        if verbosity > 0 : print("Making background plots")
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier
        hep.cms.label(rlabel="(13 TeV)")
        for cuts, scores in zip(ana_variant[ana_type]['cut_values'], primary_var_ddt):
            if cuts != 0.6: alpha = 0.3
            else: alpha = 1.0
            ax.hist(scores, bins=50, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
        ax.set_xlabel('BKG_score_ddt')
        ax.set_ylabel('Events')
        ax.legend()
        save_plot(plt,'DDT_score')
        plt.close()

        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier
        hep.cms.label(rlabel="(13 TeV)")
        var_label = ana_variant[ana_type]["primary_var_label"]
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(mT[score_mask], bins=50, range=(180,650), histtype='step', label=f'DDT({var_label} {cuts})')

        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        save_plot(plt,f'bkg_events_vs_mT_{ana_type}')

        # log scale it
        ax.set_yscale('log')
        save_plot(plt,f'log_bkg_events_vs_mT_{ana_type}')

        # Do it again normalized to unit area
        fig, ax = plt.subplots(figsize=(10, 8))
        # Options to make the plot fancier
        hep.cms.label(rlabel="(13 TeV)")
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(mT[score_mask], bins=50, range=(180,650), histtype='step', label=f'DDT({var_label} {cuts})', density=True)

        ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
        ax.set_ylabel('Events')
        ax.legend()
        save_plot(plt,f'norm_bkg_events_vs_mT_{ana_type}')
        plt.close()

        if verbosity > 1 : print(primary_var_ddt)

    # Thi
    if ana_type == 'BDT-based' :

        # _____________________________________________
        # Create Significance Plots
        if 'fom_significance' in plots :

            # Group files by mass point
            masses = ['mMed-200', 'mMed-250', 'mMed-300', 'mMed-350', 'mMed-400', 'mMed-450', 'mMed-500', 'mMed-550']
            files_by_mass = {mass: [] for mass in masses}
            for f in glob.glob(sig_files):
                for mass in masses:
                    if mass in f : files_by_mass[mass].append(f)
            # Load and combine data for each mass point
            sig_cols = []
            for mass, files in files_by_mass.items():
                mass_cols = []
                for yr_file in files :
                    mass_cols.append(Columns.load(yr_file) )
                sig_cols.append(svj.concat_columns(mass_cols))

            if verbosity > 1 : print(sig_cols)

            # Prepare a figure
            fig = plt.figure(figsize=(10, 7))
            hep.cms.label(rlabel="(13 TeV)") # full run 2
            ax = fig.gca()

            best_bdt_cuts = [] #store the best bdt values
            # Iterate over the variables in the 'con' dictionary
            for sig_col in sig_cols:
                mz = sig_col.metadata['mz']
                s = f'bsvj_{mz:d}_10_0.3'

                # Grab the input features and weights
                sig_X, sig_pT, sig_mT, sig_rho, sig_weight = bdt_ddt_inputs(sig_col, lumi, ana_variant[ana_type]['features'])
                if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

                # _____________________________________________
                # Open the trained models and get the scores

                sig_score = calc_bdt_scores(sig_X, model_file=model_file)

                # _____________________________________________
                # Apply the DDT and calculate FOM
                bkg_score_ddt = []
                if verbosity > 0 : print("Applying the DDT and calculate FOM")

                # Iterate over the bdt cut values
                fom = [] # to store the figure of merit values
                for cut_val in ana_variant[ana_type]['cut_values']:

                    sig_score_ddt = calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_weight, cut_val, ddt_map_file)
                    bkg_score_ddt = calculate_varDDT(mT, pT, rho, primary_var, bkg_weight, cut_val, ddt_map_file)

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
                    if verbosity > 0 : print("mZ': ", mz, " S: ", S, "B: ", B)
                    fom.append(FOM(S,B))

                # Find the cut value corresponding to the maximum figure of merit
                fom_array = np.array(fom)
                bdt_cuts_array = np.array(ana_variant[ana_type]['cut_values'])
                best_bdt_cuts.append([mz, bdt_cuts_array[np.where(fom_array == max(fom_array))[0]][0]])
                if verbosity > 1 : print(best_bdt_cuts)

                # Plot the histogram of the metric variable range
                if mz == 300: alpha = 1.0
                else: alpha = 0.3
                arr = ax.plot(ana_variant[ana_type]['cut_values'],fom, marker='o', label=f"m(Z') {mz}", alpha=alpha)

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
            if verbosity > 0 : print("plotting the FOM for the BDT cuts")

            # Save the plot as a PDF and png file
            # cannot use layout_tight, will cause saving errors
            save_plot(plt, 'metrics_bdt_FOM', flag_tight_layout=False, bbox_inches='tight')
            plt.close()

            # sort the best bdt cuts
            best_bdt_cuts = np.array(best_bdt_cuts)
            sort_indices = np.argsort(best_bdt_cuts[:,0]) #sort array to ascend by mz
            best_bdt_cuts = best_bdt_cuts[sort_indices]

            # Find optimal cut
            mask = (best_bdt_cuts[:,0] >= 200) & (best_bdt_cuts[:,0] <= 400)
            selected_values = best_bdt_cuts[mask, 1]
            average = np.mean(selected_values)
            optimal_bdt_cut = min(ana_variant[ana_type]['cut_values'], key=lambda x: abs(x - average)) # Finding the nearest value in ana_variant[ana_type]['cut_values'] to the calculated average

            # plot the best bdt cuts
            fig = plt.figure(figsize=(10, 7))
            hep.cms.label(rlabel="(13 TeV)") # full run 2
            ax = fig.gca()
            ax.plot(best_bdt_cuts[:,0], best_bdt_cuts[:,1], marker='o')
            ax.text(0.05, 0.10, f'Optimal Cut: {optimal_bdt_cut:.2f}', transform=ax.transAxes, verticalalignment='top')
            ax.ticklabel_format(style='sci', axis='x')
            ax.set_ylabel('Best BDT Cut Value')
            ax.set_xlabel("m(Z')")
            # cannot use layout_tight, will cause saving errors
            save_plot(plt, 'best_bdt_cuts', flag_tight_layout=False, bbox_inches='tight')
            plt.close()



        # _____________________________________________
        # Apply the DDT to different signals for one BDT cut value

        if 'sig_mt_single_BDT' in plots :

            # grab signal data
            sig_cols = [Columns.load(f) for f in glob.glob(sig_files)]

            # make plotting objects
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")

            # Loop over the mZ values
            for sig_col in sig_cols:
                mz = sig_col.metadata['mz']

                # Signal Column
                sig_X, sig_pT, sig_mT, sig_rho, sig_weight = bdt_ddt_inputs(sig_col, lumi, ana_variant[ana_type]['features'])
                if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

                # _____________________________________________
                # Open the trained models and get the scores

                sig_score = calc_bdt_scores(sig_X, model_file=model_file)

                # _____________________________________________
                # Apply the DDT  to the signal
                if verbosity > 0 : print("Applying the DDT signal")
                sig_score_ddt = calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_weight, sig_bdt_cut, ddt_map_file)

                # Make mT distributions
                if mz != 300: alpha = 0.3
                else: alpha = 1.0
                score_mask = sig_score_ddt > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[score_mask], weights=sig_weight[score_mask], bins=50, range=(180,650), histtype='step', label=f"m(Z')={mz}", alpha=alpha)

            ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'sig_events_vs_mT')

            # log scale it
            ax.set_yscale('log')
            save_plot(plt,'log_sig_events_vs_mT')
            plt.close()

        # _____________________________________________
        # Apply the DDT to one signals for different BDT cut values

        if 'one_sig_mt_many_bdt' in plots :

            # grab signal data
            sig_cols = [Columns.load(f) for f in glob.glob(sig_files)]

            # Loop over the mZ values and only grab the mZ = 300 value
            sig_col = None
            for col in sig_cols:
                mz = col.metadata['mz']
                if mz == 300 : sig_col = col
            if sig_col == None :
                raise FileNotFoundError("The mZ' 300 file doesn't exist. Please make sure you provide it in order to make the one sigma plots.")


            # make plotting objects
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")

            mz = sig_col.metadata['mz']

            # Signal Column
            sig_X, sig_pT, sig_mT, sig_rho, sig_weight = bdt_ddt_inputs(sig_col, lumi, ana_variant[ana_type]['features'])
            if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

            # _____________________________________________
            # Open the trained models and get the scores

            sig_score = calc_bdt_scores(sig_X, model_file=model_file)

            # _____________________________________________
            # Apply the DDT  to the signal at all the cut values
            if verbosity > 0 : print("Applying the DDT signal")
            sig_score_ddt = []
            for cut_val in ana_variant[ana_type]['cut_values'] :
                sig_score_ddt.append(calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_weight, cut_val, ddt_map_file) )

            # Plot histograms for the DDT scores for different BDT cuts
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")
            for cuts, scores in zip(ana_variant[ana_type]['cut_values'], sig_score_ddt):
                if cuts != 0.7: alpha = 0.3
                else: alpha = 1.0
                ax.hist(scores, weights=sig_weight, bins=50, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
            ax.set_xlabel('DDT(BDT)')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'DDT_sig_score_mz300')
            plt.close()

            # Apply DDT > 0.0 for the different BDT score transformations
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")
            for cuts, scores in zip(ana_variant[ana_type]['cut_values'], sig_score_ddt):
                score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[score_mask], weights=sig_weight[score_mask], bins=25, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})')

            ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'sig_mz300_events_vs_mT')

            # log scale it
            ax.set_yscale('log')
            save_plot(plt,'log_sig_mz300_events_vs_mT')

            # Do it again normalized to unit area
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")
            for cuts, scores in zip(ana_variant[ana_type]['cut_values'], sig_score_ddt):
                score_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[score_mask], weights=sig_weight[score_mask], bins=25, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})', density=True)

            ax.set_xlabel('$\\mathrm{m}_{\\mathrm{T}}$')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'norm_sig_mz300_events_vs_mT')
            plt.close()

if __name__ == '__main__':
    main()
