from common import calculate_varDDT, apply_rt_signalregion, get_event_weight, Columns, expand_wildcards, MTHistogram
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import pickle
import scipy.stats

def bdt_ddt_inputs(input_files: list[str], lumi):
    all_features = ["ecfm2b1", "pt", "mt", "rho", "rt"]
    def _get_cols(file: str):
        return apply_rt_signalregion(Columns.load(file))

    def _get_features(col):
        return col.to_numpy(all_features)

    def _get_mask(x, w):
        mt = x[:,-3]
        # Creating bins from the main histogram of interest, we don't really care about the actual
        # range as long as it ls larger than the defined region-of-interest of the fit
        mt_binw, mt_min, mt_max = MTHistogram.default_binning
        mt_min = np.around(mt_min / 2, mt_binw)
        mt_max = np.around(mt_max * 2, mt_binw)
        mt_edges = np.arange(mt_min, mt_max, mt_binw)
        bin_idx = np.digitize(mt, mt_edges) # Getting which bin the item should be in
        bin_count, _ = np.histogram(mt, bins=mt_edges) # Getting the number of entries in each bin
        bin_count = np.concatenate([[0], bin_count, [0]]) # Adding overflow bin to have bin_count match np.digitize ourput
        mask_bin = [True, ] # Constructing the array for which bin should be masked
        for i in range(1, len(bin_count)-1):
            if bin_count[i-1] == 0 and bin_count[i+1] == 0:
                mask_bin.append(False) # Mask if neighboring bins are both empty
            else:
                mask_bin.append(True)
        mask_bin.append(False) # Always mask overflow bin
        mask_bin = np.array(mask_bin)
        return mask_bin[bin_idx] # Extracting to per-event masking via array index

    cols = [_get_cols(f) for f in tqdm.tqdm(input_files, desc="Loading file") ]

    # Extract and filter
    X_list, W_list = [], []
    for col in tqdm.tqdm(cols, desc="Post processing"):
        x = _get_features(col)
        w = get_event_weight(col, lumi)
        if len(x) == 0: # Skipping length 0 arrays, as this messes up the masking creating routine
            continue
        # Only construct mask for background sample
        mask = _get_mask(x,w) if col.metadata['sample_type'] == 'bkg' else np.ones_like(w, dtype=bool)
        X_list.append(x[mask])
        W_list.append(w[mask])

    X = np.concatenate(X_list)
    weight = np.concatenate(W_list)

    # Grab tail variables
    rt, rho, mT, pT = X[:, -1], X[:, -2], X[:, -3], X[:, -4]
    X = X[:, :-4]

    return X, pT, mT, rho, weight

def make_plot():
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.cms.label(rlabel="(13 TeV)")

    if not os.path.exists("./cut-based.cache.pkl"):
        X, pT, mT, rho, bkg_weight = bdt_ddt_inputs(expand_wildcards(['root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Summer*/*.npz']), 137600)
        pickle.dump((X, pT, mT, rho , bkg_weight), open("./cut-based.cache.pkl", "wb"))
    else:
        X, pT, mT, rho, bkg_weight = pickle.load(open("./cut-based.cache.pkl", "rb"))

    frac = np.sum(bkg_weight[X[:, 0] > 0.11]) / np.sum(bkg_weight)
    nevt = np.sum(bkg_weight) **2 / np.sum(bkg_weight **2)
    error = scipy.stats.binomtest(int(frac*nevt), int(nevt), p=frac).proportion_ci(0.68)
    lo, hi = (error.low - frac)/frac, (error.high -frac)/frac # Converting to relative error
    lo, hi = lo / (1-frac), hi / (1-frac) # Scaling by different denominator
    lo, hi = lo + frac/(1-frac), hi + frac/(1-frac)
    ax.fill_between([180, 650], lo, hi, color="gray", alpha=0.5)
    ax.plot([180,650],[frac/(1-frac), frac/(1-frac)], color='k')

    for idx, (ddt_map, label) in enumerate([
            ("./models/cutbased_ddt_map_allbkg_nofilter.json", "No smoothing"),
            ("./models/cutbased_ddt_map_allbkg_smooth=1.0.json", "Smoothing $\sigma=1.0$"),
            ("./models/cutbased_ddt_map_allbkg_smooth=0.5.json", "Smoothing $\sigma=0.5$"),
            #("./models/cutbased_ddt_map_allbkg_smooth=1.0_nozeroDDT.json", "Set raw DDT to original"),
        ]):
        primary_var_ddt = calculate_varDDT(mT, pT, rho, X[:,0], bkg_weight, 0.11, ddt_map)

        # Plot ratio of events above and below DDT > 0 in mT bins as step histograms
        bin_edges = np.linspace(180, 650, 51)  # 50 bins
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1]-bin_edges[0]
        mask_above = primary_var_ddt > 0
        mask_below = primary_var_ddt < 0

        num_above, _ = np.histogram(mT[mask_above], bins=bin_edges, weights=bkg_weight[mask_above])
        num_below, _ = np.histogram(mT[mask_below], bins=bin_edges, weights=bkg_weight[mask_below])

        sum_s, _ = np.histogram(mT, bins=bin_edges, weights=bkg_weight)
        sum_s2, _ = np.histogram(mT, bins=bin_edges, weights=bkg_weight**2)

        ratio = np.divide(num_above, num_below, out=np.zeros_like(num_above, dtype=float), where=num_below > 0)
        frac = num_above / sum_s
        nevt = sum_s**2 / sum_s2
        err = np.sqrt((frac)*(1-frac)/nevt)

        ax.errorbar(bin_centers+bin_width*idx/10.0, ratio, yerr=err/(1-frac), marker='o', label=label)

    ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
    ax.set_ylabel(r'Ratio: $\mathrm{DDT} > 0 \,/\, \mathrm{DDT} < 0$')
    ax.legend(title="ECF cut=0.11", ncol=2)
    ax.set_ylim(bottom=0.5, top=0.7)
    fig.savefig('bkg_ddt_ratio_vs_mT_filter_comparison.pdf')
    plt.close()


make_plot()
