import os, os.path as osp, argparse, glob, json
from time import strftime

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from common import logger, DATADIR, filter_pt, filter_ht, Columns, time_and_log, columns_to_numpy, read_training_features, set_matplotlib_fontsizes, imgcat
from training import reweight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='.json file to the trained model')
    parser.add_argument('-d', '--debug', action='store_true', help='Uses only small part of data set for testing')
    parser.add_argument('--ref', type=str, default='data/train_signal/madpt300_mz350_mdark10_rinv0.1.npz', help='path to the npz file for the reference distribution for reweighting.')
    parser.add_argument('--downsample', type=float, default=.4)
    args = parser.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    training_features = read_training_features(args.model)

    # __________________________________________________________
    # Load data

    train_signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*.npz')]
    test_signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*.npz')]

    train_bkg_cols = [
        Columns.load(f) for f in
        glob.glob(DATADIR+'/train_bkg/Summer20UL18/QCD_*.npz')
        + glob.glob(DATADIR+'/train_bkg/Summer20UL18/TTJets_*.npz')
        ]
    test_bkg_cols = [
        Columns.load(f) for f in
        glob.glob(DATADIR+'/test_bkg/Summer20UL18/QCD_*.npz')
        + glob.glob(DATADIR+'/test_bkg/Summer20UL18/TTJets_*.npz')
        ]

    def filter_bkg(bkg_cols):
        bkg_cols = filter_pt(bkg_cols, 300.)
        bkg_cols = filter_ht(bkg_cols, 400., 'wjets')
        # Filter out wjets inclusive bin - it's practically the HT<100 bin, and it's giving problems
        bkg_cols = [c for c in bkg_cols if not(c.metadata['bkg_type']=='wjets' and 'htbin' not in c.metadata)]
        return bkg_cols

    train_bkg_cols = filter_bkg(train_bkg_cols)
    test_bkg_cols = filter_bkg(test_bkg_cols)

    if args.debug:
        # Use very small portion of data for debugging
        train_signal_cols = train_signal_cols[:2]
        train_bkg_cols = train_bkg_cols[:4]
        test_signal_cols = test_signal_cols[:2]
        test_bkg_cols = test_bkg_cols[:4]

    all_cols = train_signal_cols + train_bkg_cols + test_signal_cols + test_bkg_cols

    # Reweighting to mT
    reference_col = Columns.load(osp.abspath(args.ref))
    reweight(reference_col, all_cols, 'mt')

    # __________________________________________________________
    # Score

    with time_and_log('Scoring...'):
        X_train, y_train, weight_train = columns_to_numpy(
            train_signal_cols, train_bkg_cols, training_features,
            weight_key='reweight', downsample=args.downsample
            )
        weight_train *= 100. # Was done in training for stability; probably not needed
        score_train = model.predict_proba(X_train)[:,1]

        X_test, y_test, weight_test = columns_to_numpy(
            test_signal_cols, test_bkg_cols, training_features,
            weight_key='reweight', downsample=args.downsample
            )
        weight_test *= 100. # Was done in training for stability; probably not needed
        score_test = model.predict_proba(X_test)[:,1]


    # __________________________________________________________
    # Make histograms


    n_bins = 40
    score_axis = np.linspace(0., 1., n_bins)

    hist_sig_train, _ = np.histogram(score_train[y_train==1], score_axis, weights=weight_train[y_train==1])
    hist_bkg_train, _ = np.histogram(score_train[y_train==0], score_axis, weights=weight_train[y_train==0])
    hist_sig_test, _ = np.histogram(score_test[y_test==1], score_axis, weights=weight_test[y_test==1])
    hist_bkg_test, _ = np.histogram(score_test[y_test==0], score_axis, weights=weight_test[y_test==0])
    
    set_matplotlib_fontsizes()
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca()

    from scipy.stats import ks_2samp
    ks_sig = ks_2samp(hist_sig_test, hist_sig_train)
    ks_bkg = ks_2samp(hist_bkg_test, hist_bkg_train)

    for h, label in [
        (hist_sig_train, 'sig_train'),
        (hist_sig_test, f'sig_test, ks={ks_sig.statistic:.3f}, p-val={ks_sig.pvalue:.3f}'),
        (hist_bkg_train, 'bkg_train'),
        (hist_bkg_test, f'bkg_test, ks={ks_bkg.statistic:.3f}, p-val={ks_bkg.pvalue:.3f}'),
        ]:
        ax.step(score_axis[:-1], h, where='pre', label=label)

    ax.legend()
    ax.set_xlabel('BDT Score')
    ax.set_ylabel('A.U.')
    plt.savefig('overfit.png', bbox_inches='tight')
    imgcat('overfit.png')



if __name__ == '__main__':
    main()
