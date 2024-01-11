import os, os.path as osp, argparse, glob, json
from time import strftime

import numpy as np
import xgboost as xgb

import common
from common import logger, DATADIR, filter_pt, filter_ht, Columns, time_and_log, columns_to_numpy, read_training_features, Scripter, mask_cutbased

scripter = Scripter()


class Histogram:
    """
    Histogram container class.

    Keeps track of binning, values, errors, and metadata.
    Designed to be easily JSON-serializable.
    """
    @classmethod
    def from_dict(cls, dict):
        inst = cls.__new__(cls)
        inst.binning = np.array(dict['binning'])
        inst.vals = np.array(dict['vals'])
        inst.errs = np.array(dict['errs'])
        inst.metadata = dict['metadata'].copy()
        return inst

    def __init__(self, binning, vals=None, errs=None):
        self.binning = binning
        self.vals = np.zeros(self.nbins) if vals is None else vals
        self.errs = np.sqrt(self.vals) if errs is None else errs
        self.metadata = {}

    @property
    def nbins(self):
        return len(self.binning)-1

    def json(self):
        # Convert anything that remotely looks like a float to python float.
        for k, v in self.metadata.items():
            try:
                self.metadata[k] = float(v)
            except ValueError:
                pass
        return dict(
            type = 'Histogram',
            binning = list(self.binning),
            vals = list(self.vals),
            errs = list(self.errs),
            metadata = self.metadata.copy()
            )

    def __repr__(self):
        d = np.column_stack((self.vals, self.errs))
        return (
            f'<H n={self.nbins} int={self.vals.sum():.3f}'
            f' binning={self.binning[0]:.1f}-{self.binning[-1]:.1f}'
            f' vals/errs=\n{d}'
            '>'
            )

    def copy(self):
        the_copy = Histogram(self.binning.copy(), self.vals.copy(), self.errs.copy())
        the_copy.metadata = self.metadata.copy()
        return the_copy

    def __add__(self, other):
        """Add another Histogram or a numpy array to this histogram. Returns new object."""
        ans = self.copy()
        if isinstance(other, Histogram):
            ans.vals = self.vals + other.vals
            ans.errs = np.sqrt(self.errs**2 + other.errs**2)
        elif hasattr(other, 'shape') and self.vals.shape == other.shape:
            # Add a simple np histogram on top of it
            ans.vals += other
            ans.errs = np.sqrt(self.errs**2 + other)
        return ans

    def __radd__(self, other):
        if other == 0:
            return self.copy()
        raise NotImplemented

    @property
    def norm(self):
        return self.vals.sum()


def repr_dict(d, depth=0):
    s = []
    for key, val in d.items():
        if isinstance(val, np.ndarray):
            print(f'WARNING: {key} is type ndarray! {val=}')
        s.append(depth*'  ' + repr(key))
        if hasattr(val, 'items') and len(val):
            if val.get('type', '') == 'Histogram':
                s[-1] += f' (histogram; norm={sum(val["vals"]):.4f})'
            else:
                s.append(repr_dict(val, depth+1))
    return '\n'.join(s)


def filter_bad_bkgs(bkgs):
    # Filter empty backgrounds
    bkgs = [c for c in bkgs if len(c)]
    # Filter out QCD with pT<300
    # Only singular events pass the preselection, which creates spikes in the final bkg dist
    bkgs = filter_pt(bkgs, 300.)
    # Same story for wjets with HT<400
    bkgs = filter_ht(bkgs, 400., 'wjets')
    # Filter out wjets inclusive bin - it's practically the HT<100 bin,
    # and it's giving problems
    bkgs = [c for c in bkgs if not(c.metadata['bkg_type']=='wjets' and 'htbin' not in c.metadata)]
    return bkgs


@scripter
def cutbased():
    mt_axis = common.MT_BINS
    lumi = common.pull_arg('--lumi', type=float, default=137.2).lumi
    systfile = common.pull_arg('-s', '--systfile', type=str).systfile
    outfile = common.pull_arg('-o', '--outfile', type=str, default=strftime('histograms_cutbased_%Y%m%d.json')).outfile
    lumi *= 1e3 # Convert to nb-1 for easier multiplication with xs (which is in nb)
    npzfiles = common.pull_arg('npzfiles', nargs='+', type=str).npzfiles

    signals = [] ; bkgs = []
    for c in (Columns.load(f) for f in npzfiles):
        c.mask = mask_cutbased(c)
        if 'mz' in c.metadata:
            signals.append(c)
        else:
            bkgs.append(c)

    bkgs = filter_bad_bkgs(bkgs)

    out = {}
    out['version'] = 3
    out['mt'] = list(mt_axis)

    out['histograms'] = {}
    out['histograms']['0.000'] = {
        'qcd' : Histogram(mt_axis),
        'ttjets' : Histogram(mt_axis),
        'wjets' : Histogram(mt_axis),
        'zjets' : Histogram(mt_axis)
        }

    # Backgrounds
    for c in bkgs:
        mt = c.arrays['mt']
        mt_dist = np.histogram(mt[c.mask], mt_axis)[0] / len(mt)
        mt_dist *= c.xs * c.presel_eff * lumi
        out['histograms']['0.000'][c.metadata['bkg_type']] += mt_dist
    out['histograms']['0.000']['bkg'] = sum(out['histograms']['0.000'].values())
    # Convert to json
    out['histograms']['0.000'] = {k: h.json() for k, h in out['histograms']['0.000'].items()}

    if systfile:
        common.logger.info(f'Loading systematics from {systfile}')
        with open(systfile, 'r') as f:
            systs = json.load(f)

    # Signals
    for c in signals:
        mt = c.arrays['mt']
        mt_dist = np.histogram(mt[c.mask], mt_axis)[0] / len(mt) * c.effxs * lumi
        histogram = Histogram(mt_axis, mt_dist)
        histogram.metadata.update(c.metadata)
        key = f"mz{c.metadata['mz']}_mdark{c.metadata['mdark']}_rinv{c.metadata['rinv']:.1f}"
        out['histograms']['0.000'][key] = histogram.json()

        if systfile:
            # Histograms don't have the correct normalization yet
            # Normalize them to the current signal
            central_norm = np.array(systs['central']['vals']).sum()
            for name, hist in systs.items():
                if name in ['central', 'selection']: continue
                hist = Histogram.from_dict(hist)
                hist.vals *= histogram.norm / central_norm
                hist.metadata.update(histogram.metadata)
                hist.metadata['systname'] = name
                out['histograms']['0.000'][f'SYST_{key}_{name}'] = hist.json()                

    logger.info(f'Dumping the following dict tree to {outfile}:\n{repr_dict(out)}')
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=4)


@scripter
def bdt():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='.json file to the trained model')
    parser.add_argument('-d', '--debug', action='store_true', help='Uses only small part of data set for testing')
    parser.add_argument('--lumi', type=float, default=137.2, help='Luminosity (in fb-1)')
    parser.add_argument('-o', '--outfile', type=str, default=strftime('histograms_%b%d.json'), help='Output file for the histograms')
    args = parser.parse_args()
    lumi = args.lumi * 1e3 # Convert to nb-1 for easier multiplication with xs (which is in nb)

    model = xgb.XGBClassifier()
    model.load_model(args.model)

    training_features = read_training_features(args.model)
    DATADIR = '/home/snabili/hadoop/BKG/Ultra_Legacy/HADD_BKGCutbase'
    #signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/signal_notruthcone/*.npz')]
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/signal_notruth/*.npz')]
    bkg_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/bkg/Summer20UL18/*.npz')]
    bkg_cols = [c for c in bkg_cols if len(c)] # Filter empty backgrounds
    bkg_cols = filter_pt(bkg_cols, 300.)
    bkg_cols = filter_ht(bkg_cols, 400., 'wjets')
    # Filter out wjets inclusive bin - it's practically the HT<100 bin,
    # and it's giving problems
    bkg_cols = [c for c in bkg_cols if not(c.metadata['bkg_type']=='wjets' and 'htbin' not in c.metadata)]

    if args.debug:
        signal_cols = signal_cols[:2]
        bkg_cols = bkg_cols[:4]

    n_bins = 100
    mt_axis = np.linspace(100., 1000., n_bins+1)
    
    with time_and_log(f'Scoring all backgrounds and signals'):
        for c in bkg_cols:
            if not len(c): continue
            c.arrays['bdtscore'] = model.predict_proba(c.to_numpy(training_features))[:,1]
        for c in signal_cols:
            if not len(c): continue
            c.arrays['bdtscore'] = model.predict_proba(c.to_numpy(training_features))[:,1]

    out = {}
    out['version'] = 2
    out['mt'] = list(mt_axis)
    histograms = {}
    out['histograms'] = histograms

    for bdtcut in .1*np.arange(10):
        logger.info(f'bdtcut={bdtcut}')
        mt_dist_per_bkg_type = {
            'qcd' : Histogram(mt_axis),
            'ttjets' : Histogram(mt_axis),
            'wjets' : Histogram(mt_axis),
            'zjets' : Histogram(mt_axis)
            }
        for c in bkg_cols:
            # Take BDT eff and fraction inside bins into account in one go
            mt = c.arrays['mt']
            mt_dist = np.histogram(mt[c.arrays['bdtscore'] > bdtcut], mt_axis)[0] / len(mt)
            e = mt_dist.sum()
            mt_dist *= c.xs * c.presel_eff * lumi
            logger.debug(
                f'{c}:\n    bdtcut_eff * mt_range_eff = {e:.3f}'
                f', xs={c.xs:8.2f}, presel_eff={c.presel_eff:.3f}, lumi={lumi:.2f}'
                f', n@137.2={mt_dist.sum():.2f}'
                )
            mt_dist_per_bkg_type[c.metadata['bkg_type']] += mt_dist

        bdtcutkey = f'{bdtcut:.3f}'
        out['histograms'][bdtcutkey] = {}
        for bkg, hist in mt_dist_per_bkg_type.items():
            out['histograms'][bdtcutkey][bkg] = hist.json()
        out['histograms'][bdtcutkey]['bkg'] = sum(mt_dist_per_bkg_type.values()).json()

        # Signals
        for c in signal_cols:
            mt = c.arrays['mt']
            mt_dist = np.histogram(mt[c.arrays['bdtscore'] > bdtcut], mt_axis)[0] / len(mt) * c.effxs * lumi
            histogram = Histogram(mt_axis, mt_dist)
            histogram.metadata.update(c.metadata)
            key = f"mz{c.metadata['mz']}_mdark{c.metadata['mdark']}_rinv{c.metadata['rinv']:.1f}"
            out['histograms'][bdtcutkey][key] = histogram.json()

    logger.info(f'Dumping the following dict tree to {args.outfile}:\n{repr_dict(out)}')
    with open(args.outfile, 'w') as f:
        json.dump(out, f)


@scripter
def ls():
    jsonfile = common.pull_arg('jsonfile', type=str).jsonfile
    with open(jsonfile, 'r') as f:
        out = json.load(f)
    print(repr_dict(out))

    print(out['histograms']['0.0']['bsvj_200_10_0.1'])


if __name__ == '__main__':
    scripter.run()
