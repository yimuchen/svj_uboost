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
from common import create_DDT_map_dict, calculate_varDDT
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
import seutils as se

np.random.seed(1001)

from common import read_training_features, logger, lumis, DATADIR, Columns, time_and_log, imgcat, set_mpl_fontsize, columns_to_numpy, calc_bdt_scores, expand_wildcards, signal_xsecs, MTHistogram, get_event_weight, mask_isolated_bins, SELECTION_RT_SIGNAL_REGION

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
    parser.add_argument('--ddt_map_file', default='models/bdt_ddt_AN_v6.json', help='DDT map file')
    parser.add_argument('--rt_ddt_file', default="models/rt_ddt_map.json", help='DDT map file for RT selection')

    # The default value of 0.65 was the optimal cut value determined. If the training or selection changes,
    # the value should be adapted accordingly
    parser.add_argument('--sig_bdt_cut', type=float, default=0.67, help='BDT cut for signal plotting (current optimal cut is 0.67)')

    # Choose the BDT cut values that you want to make for the DDT
    # or that are in the DDT you are loading
    # another common set of cuts is
    # bdt_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95]
    # and another common set for plots is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    parser.add_argument('--bdt_cuts', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95], help='List of BDT cuts')

    # Allowed plots: 2D DDT maps, Background Scores vs MT, FOM significance,
    #                Signal mt spectrums for one BDT working point,  one signal mt spectrum for many BDT working points
    allowed_plots = ['RT_DDT_map', '2D_DDT_map', 'bkg_scores_mt', 'fom_significance', 'sig_mt_single_BDT', 'one_sig_mt_many_bdt']

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

def simple_fig():
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.cms.label(rlabel="(13 TeV)")
    return fig, ax


def save_plot(plt, plt_name, flag_tight_layout=True, **kwargs):
    '''
    Save the plots, save some lines of code
    '''
    if flag_tight_layout == True: plt.tight_layout()
    plt.savefig(f'plots/{plt_name}.png', **kwargs)
    plt.savefig(f'plots/{plt_name}.pdf', **kwargs)


def bdt_ddt_inputs(input_files: list[str], all_features: list[str], rt_ddt_file: str|None = None):
    def _get_mask(x, w):
        mt = x[:,-3]
        # Creating bins from the main histogram of interest, we don't really care about the actual
        # range as long as it ls larger than the defined region-of-interest of the fit
        mt_binw, mt_min, mt_max = MTHistogram.default_binning
        mt_min = np.around(mt_min / 2, mt_binw)
        mt_max = np.around(mt_max * 2, mt_binw)
        mt_edges = np.arange(mt_min, mt_max, mt_binw)
        bin_idx = np.digitize(mt, mt_edges) # Getting which bin the item should be in
        bin_idx[bin_idx==0] = 1 # put underflow into first bin
        bin_idx[bin_idx==len(mt_edges)] = len(mt_edges)-1 # put overflow into last bin
        bin_idx = bin_idx - 1 # shift to be consistent with histogram indexing
        bin_count, _ = np.histogram(mt, bins=mt_edges) # Getting the number of entries in each bin
        mask_bin = mask_isolated_bins(bin_count)
        return mask_bin[bin_idx] # Extracting to per-event masking via array index

    # Extract and filter
    X_list, W_list = [], []
    for input_file in input_files:
        col = Columns.load(input_file)
        x = col.to_numpy(all_features)
        w = get_event_weight(col, lumis[str(col.metadata["year"])])
        if len(x) == 0: # Skipping length 0 arrays, as this messes up the masking creating routine
            continue
        # Only construct mask for background sample
        mask = _get_mask(x,w) if col.metadata['sample_type'] == 'bkg' else np.ones_like(w, dtype=bool)
        X_list.append(x[mask])
        W_list.append(w[mask])

    X = np.concatenate(X_list)
    weight = np.concatenate(W_list)

    # Grab tail variables
    rT, rho, mT, pT = X[:, -1], X[:, -2], X[:, -3], X[:, -4]
    X = X[:, :-4]

    return X, pT, mT, rT, weight

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
    rt_ddt_file = args.rt_ddt_file
    sig_bdt_cut = args.sig_bdt_cut
    bdt_cuts = args.bdt_cuts
    plots = args.plot
    verbosity = args.verbosity

    set_mpl_fontsize(18, 22, 26)

    #--------------------------------------------------------------------------
    # For the cut-based search ------------------------------------------------
    #--------------------------------------------------------------------------
    # Grab the bkg data
    features_common = ["pt", "mt", "rho", "rt"]
    ana_variant = {
        "cut-based": {
            "features": ["ecfm2b1"] + features_common,
            "inputs_to_primary": lambda x: x[:, 0],
            "primary_var_label": "$M_2^{(1)}$ $>$ ",
            "cut_values":  [0.08,0.09,0.10,0.11,0.12]
            #"cut_values":  [np.round(x,4) for x in np.linspace(0.07 , 0.17, 41)]
        },
        "BDT-based": {
            "features": read_training_features(model_file) + features_common,
            "inputs_to_primary": lambda x:  calc_bdt_scores(x, model_file=model_file),
            "primary_var_label": "BDT $>$",
            "cut_values": bdt_cuts
        }
    }

    # Efficiencies of the BDT/ECF are evaluated without RT cut. The optimal cut would determined with the FOM
    # evaluation. Here we select a top level evaluation
    X, pT, mT, rT, bkg_weight = bdt_ddt_inputs(expand_wildcards([bkg_files]), ana_variant[ana_type]["features"])
    primary_var = ana_variant[ana_type]["inputs_to_primary"](X)

    # Calculating the DDT files for RT selection, since we there is a single efficiency
    if not osp.exists(rt_ddt_file):
        logger.info("Creating DDT map file for RT selection")
        eff = np.sum(bkg_weight[rT > SELECTION_RT_SIGNAL_REGION]) / np.sum(bkg_weight)
        create_DDT_map_dict(mT, pT, rT, bkg_weight, [eff *100], [SELECTION_RT_SIGNAL_REGION], rt_ddt_file, smear=0.2)

    # Only make the 2D DDT map if it doesn't exist
    if not osp.exists(ddt_map_file):
        rt_ddt = calculate_varDDT(mT, pT, rT, str(SELECTION_RT_SIGNAL_REGION), rt_ddt_file)
        rt_mask = rt_ddt > 0.0
        bkg_percents = []
        bkg_sum = np.sum(bkg_weight[(primary_var > 0.0) & rt_mask])
        for cut_val in ana_variant[ana_type]["cut_values"]:
            bkg_percents.append((np.sum(bkg_weight[(primary_var > cut_val) & rt_mask]) / bkg_sum)*100)
        create_DDT_map_dict(mT[rt_mask], pT[rt_mask], primary_var[rt_mask], bkg_weight[rt_mask], bkg_percents, ana_variant[ana_type]["cut_values"], ddt_map_file)
    else: print("The DDT has already been calculated, please change the name if you want to remake the ddt map")

    # Load the dictionary from the json file
    with open(ddt_map_file, 'r') as f:
        var_dict = json.load(f)

    def plot_single_ddt(var_map_smooth, MT_PT_edges, PT_edges, output_name):
        var_map_smooth = np.array(var_map_smooth)
        MT_PT_edges = np.array(MT_PT_edges)
        PT_edges = np.array(PT_edges)
        # Plot 2D map for mt-pt plane for each BDT cut
        plt.figure(figsize=(10, 8))
        hep.cms.label(rlabel="(13 TeV)")
        plt.imshow(
            var_map_smooth.T,
            extent=[MT_PT_edges[0], MT_PT_edges[-1], PT_edges[0], PT_edges[-1]],
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        plt.colorbar(label='DDT Map value')
        plt.xlabel('$\\frac{m_{\\mathrm{T}}}{p_{\\mathrm{T}}}$')
        plt.ylabel('$p_{\\mathrm{T}}$ [GeV]')
        save_plot(plt, output_name)
        plt.close()

    if 'RT_DDT_map' in plots:
        if verbosity > 0 : print("Making RT DDT plot")
        with open(rt_ddt_file, 'r') as f:
            rt_var_dict =json.load(f)
        plot_single_ddt(*rt_var_dict[str(SELECTION_RT_SIGNAL_REGION)], '2D_map_RT_DDT')

    if '2D_DDT_map' in plots:
        if verbosity > 0 : print("Making variable DDT plots")
        for key in var_dict.keys():
            plot_single_ddt(*var_dict[key], f'2D_map_{ana_type}_{key}')

    if 'bkg_scores_mt' in plots :
        if verbosity > 0 : print("Applying the DDT background")
        rt_ddt = calculate_varDDT(mT, pT, rT, str(SELECTION_RT_SIGNAL_REGION), rt_ddt_file)
        primary_var_ddt = [calculate_varDDT(mT, pT, primary_var, cut_val, ddt_map_file) for cut_val in ana_variant[ana_type]["cut_values"]]

        # Plot histograms for the DDT scores for different BDT cuts
        if verbosity > 0 : print("Making background plots")
        fig, ax = simple_fig()
        for cuts, scores in zip(ana_variant[ana_type]['cut_values'], primary_var_ddt):
            if cuts != 0.6: alpha = 0.3
            else: alpha = 1.0
            ax.hist(scores, bins=60, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
        ax.set_xlabel('BKG_score_ddt')
        ax.set_ylabel('Events')
        ax.legend()
        save_plot(plt,'DDT_score')
        plt.close()

        def save_mt_fig(ax, name, save_log=False):
            # Common method for fixing files for MT, related items
            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.legend()
            save_plot(plt, name)
            if save_log:
                ax.set_yscale('log') # Saving a log version of the plot
                save_plot(plt,f'log_{name}')
            plt.close()

        # Common label
        var_label = ana_variant[ana_type]["primary_var_label"]
        RT_mask = rt_ddt > 0.0

        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = simple_fig()
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            SR_mask = (scores > 0.0) & RT_mask
            ax.hist(mT[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT({var_label} {cuts})', weights=bkg_weight[SR_mask])
        save_mt_fig(ax, f'bkg_events_vs_mT_{ana_type}', save_log=True)

        # Do it again normalized to unit area
        fig, ax = simple_fig()
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            SR_mask = (scores > 0.0) & RT_mask # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            ax.hist(mT[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT({var_label} {cuts})', weights=bkg_weight[SR_mask], density=True)
        ax.set_ylabel('Events')
        save_mt_fig(ax, f'norm_bkg_events_vs_mT_{ana_type}', save_log=True)

        # Apply DDT > 0.0 for the different BDT score transformations
        fig,ax = simple_fig()
        mT_bins = np.linspace(180, 650, 48)
        bin_centers = 0.5 * (mT_bins[:-1] + mT_bins[1:])
        bin_widths = np.diff(mT_bins)
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            SR_mask = (scores > 0.0)& RT_mask # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            mT_before, _ = np.histogram(mT, bins=mT_bins, weights=bkg_weight)
            mT_after, _ = np.histogram(mT[SR_mask], bins=mT_bins, weights=bkg_weight[SR_mask])
            with np.errstate(divide='ignore', invalid='ignore') :
                mT_eff = mT_after / mT_before
                mT_eff[mT_after == 0] = np.nan
            ax.plot(bin_centers, mT_eff, drawstyle='steps-mid', label=f'DDT({var_label} {cuts})')
        ax.set_ylabel('Bkg efficiency')
        save_mt_fig(ax, f'bkg_eff_vs_mT_{ana_type}')

        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = simple_fig()
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            SR_mask = (scores > 0.0) & RT_mask # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            mT_before, _ = np.histogram(mT, bins=mT_bins, weights=bkg_weight)
            mT_after, _ = np.histogram(mT[SR_mask], bins=mT_bins, weights=bkg_weight[SR_mask])
            with np.errstate(divide='ignore', invalid='ignore') :
                mT_eff = mT_after / mT_before
                mT_eff[mT_after == 0] = np.nan
            mT_eff_area =  np.nansum(mT_eff * bin_widths)
            mT_norm_eff = mT_eff / mT_eff_area
            ax.plot(bin_centers, mT_norm_eff, drawstyle='steps-mid', label=f'DDT({var_label} {cuts})')
        ax.set_ylabel('norm bkg efficiency')
        save_mt_fig(ax,f'norm_bkg_eff_vs_mT_{ana_type}')

        # Plot ratio of events above and below DDT > 0 in mT bins as step histograms
        fig, ax = simple_fig()
        all_ratios = []
        for cuts, scores in zip(ana_variant[ana_type]["cut_values"], primary_var_ddt):
            mask_above = (scores > 0) & RT_mask
            mask_below = ~RT_mask
            num_above, _ = np.histogram(mT[mask_above], bins=mT_bins, weights=bkg_weight[mask_above])
            num_below, _ = np.histogram(mT[mask_below], bins=mT_bins, weights=bkg_weight[mask_below])
            ratio = np.divide(num_above, num_below, out=np.zeros_like(num_above, dtype=float), where=num_below > 0)
            all_ratios.append(ratio)
            ax.step(bin_centers, ratio, where='mid', label=f'DDT({var_label} {cuts})')
        ax.set_ylabel(r'Ratio: $\mathrm{DDT} > 0 \,/\, \mathrm{DDT} < 0$')
        save_mt_fig(ax, f'bkg_ddt_ratio_vs_mT_{ana_type}')

        # Plot normalized ratio: divide each ratio curve by its average to see shape only
        fig, ax = simple_fig()
        for cuts, ratio in zip(ana_variant[ana_type]["cut_values"], all_ratios):
            # Compute average, ignoring empty bins
            avg = np.nanmean(ratio[ratio > 0])  # or use np.mean with a mask
            normalized_ratio = np.divide(ratio, avg, out=np.zeros_like(ratio), where=avg > 0)
            ax.step(bin_centers, normalized_ratio, where='mid', label=f'DDT({var_label} {cuts})')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylim(0.0, 2.0)
        ax.set_ylabel(r'$(\mathrm{DDT}>0/\mathrm{DDT}<0) \,/\, \langle\mathrm{DDT}>0/\mathrm{DDT}<0\rangle$')
        save_mt_fig(ax, f'bkg_ddt_ratio_vs_mT_normalized_{ana_type}')


        if verbosity > 1 : print(primary_var_ddt)
    # _____________________________________________
    # Create Significance Plots
    if 'fom_significance' in plots :

        # Group files by mass point
        files_by_mass = {
            mass: [
                f for f in expand_wildcards([sig_files])
                if f'mMed-{mass}' in f and 'mDark-10' in f and 'rinv-0p3' in f
            ]
            for mass in signal_xsecs.keys()
        }
        # Prepare a figure
        fig = plt.figure(figsize=(10, 7))
        hep.cms.label(rlabel="(13 TeV)") # full run 2
        ax = fig.gca()

        best_bdt_cuts = [] #store the best bdt values
        # Iterate over the variables in the 'con' dictionary
        for mz, mz_files in files_by_mass.items():
            s = f'bsvj_{mz:d}_10_0.3'

            # Grab the input features and weights
            sig_X, sig_pT, sig_mT, sig_rho, sig_weight = bdt_ddt_inputs(mz_files, ana_variant[ana_type]['features'])
            if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

            # _____________________________________________
            # Extracting the primary variable for comparison
            sig_primary_var = ana_variant[ana_type]["inputs_to_primary"](sig_X)

            # _____________________________________________
            # Apply the DDT and calculate FOM
            bkg_score_ddt = []
            if verbosity > 0 : print("Applying the DDT and calculate FOM")

            # Iterate over the bdt cut values
            fom = [] # to store the figure of merit values
            for cut_val in ana_variant[ana_type]['cut_values']:
                def _get_ddt_yield(mT, pT, rho, var, weight):
                    score_ddt = calculate_varDDT(mT, pT, rho, var, cut_val, ddt_map_file)
                    SR_mask = score_ddt > 0
                    mt_mask = (mT > (mz - 100)) & (mT < (mz + 100))
                    mt_fill = mT[SR_mask & mt_mask]
                    weight_fill = weight[SR_mask & mt_mask]
                    return sum(np.histogram(mt_fill, bins=50, weights=weight_fill)[0])

                # Calculate the figure of merit values for this bdt cut
                S = _get_ddt_yield(sig_mT, sig_pT, sig_rho, sig_primary_var, sig_weight)
                B = _get_ddt_yield(mT, pT, rho, primary_var, bkg_weight)
                F = FOM(S,B)
                if verbosity > 0 : print("mZ': ", mz, "cut:" , cut_val, " S: ", S, "B: ", B, "FOM:" , F)
                fom.append(F)

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
        ax.set_xlabel('BDT cut value' if ana_type == "BDT-based" else "ECF cut value")
        if verbosity > 0 : print(f"plotting the FOM for the {ana_type} cuts")

        # Save the plot as a PDF and png file
        # cannot use layout_tight, will cause saving errors
        save_plot(plt, f'metrics_{ana_type}_FOM', flag_tight_layout=False, bbox_inches='tight')
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
        ax.set_ylabel('Best BDT Cut Value' if ana_type == "BDT-based" else "ECF cut value")
        ax.set_xlabel("$m(\\mathrm{Z'})$ [GeV]")
        # cannot use layout_tight, will cause saving errors
        save_plot(plt, f'best_{ana_type}_cuts', flag_tight_layout=False, bbox_inches='tight')
        plt.close()


    # _____________________________________________
    # BDT only section
    if ana_type == 'BDT-based' :
        # _____________________________________________
        # Apply the DDT to different signals for one BDT cut value

        if 'sig_mt_single_BDT' in plots :
            # make plotting objects
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")

            # Loop over the mZ values
            for sig_file in expand_wildcards([sig_files]):
                sig_col = Columns.load(sig_file)
                mz = sig_col.metadata['mz']

                # Signal Column
                sig_X, sig_pT, sig_mT, sig_rho, sig_weight = bdt_ddt_inputs([sig_file], ana_variant[ana_type]['features'])
                if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

                # _____________________________________________
                # Open the trained models and get the scores
                sig_score = ana_variant[ana_type]["inputs_to_primary"](sig_X)

                # _____________________________________________
                # Apply the DDT  to the signal
                if verbosity > 0 : print("Applying the DDT signal")
                sig_score_ddt = calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, sig_bdt_cut, ddt_map_file)

                # Make mT distributions
                if mz != 300: alpha = 0.3
                else: alpha = 1.0
                SR_mask = sig_score_ddt > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[SR_mask], weights=sig_weight[SR_mask], bins=47, range=(180,650), histtype='step', label=f"m(Z')={mz}", alpha=alpha)

            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
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
            sig_X, sig_pT, sig_mT, sig_rho, sig_weight = bdt_ddt_inputs(sig_col, ana_variant[ana_type]['features'])
            if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

            # _____________________________________________
            # Open the trained models and get the scores

            sig_score = calc_bdt_scores(sig_X, model_file=model_file)

            # _____________________________________________
            # Apply the DDT  to the signal at all the cut values
            if verbosity > 0 : print("Applying the DDT signal")
            sig_score_ddt = []
            for cut_val in ana_variant[ana_type]['cut_values'] :
                sig_score_ddt.append(calculate_varDDT(sig_mT, sig_pT, sig_rho, sig_score, cut_val, ddt_map_file) )

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
                SR_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[SR_mask], weights=sig_weight[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})')

            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
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
                SR_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[SR_mask], weights=sig_weight[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})', density=True)

            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'norm_sig_mz300_events_vs_mT')
            plt.close()

if __name__ == '__main__':
    main()
