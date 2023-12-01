import os, os.path as osp, sys, json, re
from time import strftime

THIS_DIR = osp.dirname(osp.abspath(__file__))
MAIN_DIR = osp.dirname(THIS_DIR)
sys.path.append(MAIN_DIR)

import common
from common import mask_cutbased
from produce_histograms import Histogram, repr_dict
from cutflow_table import format_table

import svj_ntuple_processing as svj
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak


from syst import BINS, NORMALIZE, DST, MTHistogram, basename, Plot

scripter = common.Scripter()


def apply_jes(arrays, var, match_type='both'):
    from jes_numba import calc_x_jes

    if not var in ['up', 'down', 'central']:
        raise Exception('var should be up, down, or central')
    if var == 'central': return

    try:
        match_type = dict(both=3, partial=1, full=2)[match_type]
    except KeyError:
        common.logger.error('Possible choices for match_type are both, partial, or full')

    a = arrays.array

    # Before correcting, save the current pT and phi (needed for MET correction)
    pt_before = a['Jets.fCoordinates.fPt']
    phi_before = a['Jets.fCoordinates.fPhi']

    for conesize in [.4, .8, 1.5]:
        conesizestr = '' if conesize == .4 else f'AK{10*conesize:.0f}'
        x_jes = calc_x_jes(
            a[f'Jets{conesizestr}.fCoordinates.fPt'],
            a[f'Jets{conesizestr}.fCoordinates.fEta'],
            a[f'Jets{conesizestr}.fCoordinates.fPhi'],
            a[f'GenJets{conesizestr}.fCoordinates.fPt'],
            a[f'GenJets{conesizestr}.fCoordinates.fEta'],
            a[f'GenJets{conesizestr}.fCoordinates.fPhi'],
            a['GenParticles.fCoordinates.fEta'],
            a['GenParticles.fCoordinates.fPhi'],
            a['GenParticles_PdgId'],
            a['GenParticles_Status'],
            do_match_type = match_type,
            drsq_comp= conesize**2
            )
        
        a[f'x_jes_{10*conesize:.0f}'] = x_jes
        if var == 'up':
            a[f'Jets{conesizestr}.fCoordinates.fPt'] = a[f'Jets{conesizestr}.fCoordinates.fPt'] * (1+x_jes)
        elif var == 'down':
            jes_down = 1-x_jes
            jes_down = ak.where(jes_down<0., 0., jes_down)
            a[f'Jets{conesizestr}.fCoordinates.fPt'] = a[f'Jets{conesizestr}.fCoordinates.fPt'] * jes_down


    # Correct MET
    a['MET_precorr'] = a['MET']
    a['METPhi_precorr'] = a['METPhi']
    px_before = np.cos(phi_before) * pt_before
    py_before = np.sin(phi_before) * pt_before
    px_after = np.cos(a[f'Jets.fCoordinates.fPhi']) * a[f'Jets.fCoordinates.fPt']
    py_after = np.sin(a[f'Jets.fCoordinates.fPhi']) * a[f'Jets.fCoordinates.fPt']
    dpx = ak.sum(px_after - px_before, axis=-1)
    dpy = ak.sum(py_after - py_before, axis=-1)
    met_x = np.cos(a['METPhi']) * a['MET'] - dpx
    met_y = np.sin(a['METPhi']) * a['MET'] - dpy
    a['MET'] = np.sqrt(met_x**2 + met_y**2)
    a['METPhi'] = np.arctan2(met_y, met_x) # Should be between -pi .. pi


@scripter
def skim_jes():
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    common.logger.info(f'Preprocessing JES info for {rootfile}')
    svj.BRANCHES_GENONLY.extend([
        'PDFweights', 'PSweights',
        'GenJets.fCoordinates.fPt',
        'GenJets.fCoordinates.fEta',
        'GenJets.fCoordinates.fPhi',
        'GenJets.fCoordinates.fE',
        'GenJetsAK8.fCoordinates.fPt',
        'GenJetsAK8.fCoordinates.fEta',
        'GenJetsAK8.fCoordinates.fPhi',
        'GenJetsAK8.fCoordinates.fE',
        'GenJetsAK15.fCoordinates.fPt',
        'GenJetsAK15.fCoordinates.fEta',
        'GenJetsAK15.fCoordinates.fPhi',
        'GenJetsAK15.fCoordinates.fE',
        ])

    for var in ['up', 'down']:
        for match_type in ['both', 'full', 'partial']:
            common.logger.info(f'{var=}, {match_type=}')
            arrays = svj.open_root(rootfile)
            common.logger.info(f'Loaded, applying jes')
            apply_jes(arrays, var, match_type)
            common.logger.info(f'Done, applying presel')
            arrays = svj.filter_preselection(arrays)
            common.logger.info(f'Done, to columns')
            cols = svj.bdt_feature_columns(arrays)
            cols.arrays['x_jes_1'] = arrays.array['x_jes_15'][:,0].to_numpy()
            cols.arrays['x_jes_2'] = arrays.array['x_jes_15'][:,1].to_numpy()
            cols.arrays['x_jes_3'] = ak.fill_none(ak.firsts(arrays.array['x_jes_15'][:,2:]), -100.).to_numpy()
            cols.arrays['MET_precorr'] = arrays.array['MET_precorr'].to_numpy()
            cols.arrays['METPhi_precorr'] = arrays.array['METPhi_precorr'].to_numpy()
            common.logger.info(f'Saving')
            cols.save(f'{DST}/{basename(arrays.metadata)}_jes{var}_{match_type}.npz')


def load_columns(skimfiles):
    """
    Organizes and loads columns from the skim files
    """
    out = {}
    for skim in skimfiles:
        if 'central' in skim:
            out['central'] = svj.Columns.load(skim)
            out['central'].metadata['name'] = 'Central'
            continue

        for truth_match_type in ['both', 'full', 'partial']:
            if truth_match_type in skim: break
        else:
            continue
        
        for direction in ['jesup', 'jesdown']:
            if direction in skim: break
        else:
            continue
        direction = direction.replace('jes','')

        col = svj.Columns.load(skim)
        col.metadata['truth_match_type'] = truth_match_type
        col.metadata['direction'] = direction
        col.metadata['name'] = f'{truth_match_type} {direction}'

        out.setdefault(truth_match_type, {})        
        out[truth_match_type][direction] = col

    # common.logger.info(f'Organized skims:\n{repr_dict(out)}')
    return (
        out['central'],
        out['both']['up'],
        out['full']['up'],
        out['partial']['up'],
        out['both']['down'],
        out['full']['down'],
        out['partial']['down']
        )


@scripter
def produce_histograms():
    selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
    skimfiles = common.pull_arg('skimfiles', nargs='+', type=str).skimfiles
    central, both_up, full_up, partial_up, both_down, full_down, partial_down = load_columns(skimfiles)
    all = [central, both_up, full_up, partial_up, both_down, full_down, partial_down]

    out = {}
    for c in all:
        sel = mask_cutbased(c) if selection=='cutbased' else None
        h = MTHistogram(c.arrays['mt'][sel])
        h.metadata.update(c.metadata)
        out[c.metadata['name']] = h.json()
    
    outfile = f'{selection}_{basename(central.metadata)}_jes_hists.json'
    common.logger.info(f'Dumping the following to {outfile}:\n{repr_dict(out)}')
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=4)


@scripter
def plot():
    selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt'])
    histfile = common.pull_arg('histfile', type=str).histfile

    with open(histfile, 'r') as f:
        hists = json.load(f)

    for truth_match_type in ['partial', 'full']:
        plot = Plot(selection)

        central = MTHistogram.from_dict(hists['Central'])
        up = MTHistogram.from_dict(hists[truth_match_type + ' up'])
        down = MTHistogram.from_dict(hists[truth_match_type + ' down'])

        plot.plot_hist(central, label='central')
        plot.plot_hist(up, central, label='up')
        plot.plot_hist(down, central, label='down')

        plot.legend_title = truth_match_type.capitalize() + ' truth match'
        outfile = f'plots/jes_{truth_match_type}.png'
        plot.save(outfile)


@scripter
def debug_plots():
    selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
    skimfiles = common.pull_arg('skimfiles', nargs='+', type=str).skimfiles
    central, both_up, full_up, partial_up, both_down, full_down, partial_down = load_columns(skimfiles)
    all = [central, both_up, full_up, partial_up, both_down, full_down, partial_down]

    # Compute the selection mask (cutbased or bdt)
    for c in all:
        c.sel = mask_cutbased(c) if selection=='cutbased' else None

    if c.sel is None:
        raise Exception

    # MET histograms before and after correction
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(24,16))
    bins = np.linspace(0., 400., 100)
    for ax, c in [
        (ax1, both_up),
        (ax2, full_up),
        (ax3, partial_up),
        (ax4, both_down),
        (ax5, full_down),
        (ax6, partial_down),
        ]:
        met_before = c.arrays['MET_precorr'][c.sel]
        met_after = c.arrays['met'][c.sel]
        for label, met in [('before', met_before), ('after', met_after)]:
            vals = np.histogram(met, bins=bins)[0]
            ax.step(bins[:-1], vals, label=label)
        ax.legend(title=c.metadata['name'])
        ax.set_xlabel('MET (GeV)')
    plt.savefig('tmp.png', bbox_inches='tight')
    common.imgcat('tmp.png')

    # METPhi histograms before and after correction
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(24,16))
    bins = np.linspace(-np.pi, np.pi, 100)
    for ax, c in [
        (ax1, both_up),
        (ax2, full_up),
        (ax3, partial_up),
        (ax4, both_down),
        (ax5, full_down),
        (ax6, partial_down),
        ]:
        met_before = c.arrays['METPhi_precorr'][c.sel]
        met_after = c.arrays['metphi'][c.sel]
        for label, met in [('before', met_before), ('after', met_after)]:
            vals = np.histogram(met, bins=bins)[0]
            ax.step(bins[:-1], vals, label=label)
        ax.legend(title=c.metadata['name'])
        ax.set_xlabel('MET Phi')
    plt.savefig('tmp.png', bbox_inches='tight')
    common.imgcat('tmp.png')

    # pT_subl histograms before and after correction
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(24,16))
    bins = np.linspace(200., 900., 100)
    for ax, c in [
        (ax1, both_up),
        (ax2, full_up),
        (ax3, partial_up),
        (ax4, both_down),
        (ax5, full_down),
        (ax6, partial_down),
        ]:
        pt_before = central.arrays['pt'][central.sel]
        pt_after = c.arrays['pt'][c.sel]
        for label, pt in [('before', pt_before), ('after', pt_after)]:
            vals = np.histogram(pt, bins=bins)[0]
            ax.step(bins[:-1], vals, label=label)
        ax.legend(title=c.metadata['name'])
        ax.set_xlabel(r'$p_{T}^{subl}$ (GeV)')
    plt.savefig('tmp.png', bbox_inches='tight')
    common.imgcat('tmp.png')

    # Histograms of 1+x_jes_{1,2,3}
    bins = np.linspace(0., 3., 100)
    density = True
    for density in [True, False]:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24,8))
        for ax, title, c in [
            (ax1, 'both', both_up),
            (ax2, 'full', full_up),
            (ax3, 'partial', partial_up),
            ]:
            c.ptratio_1 = c.arrays['x_jes_1'][c.sel & (c.arrays['x_jes_1']!=0.)] + 1.
            c.ptratio_1_hist = np.histogram(c.ptratio_1, bins)[0]
            c.ptratio_2 = c.arrays['x_jes_2'][c.sel & (c.arrays['x_jes_2']!=0.)] + 1.
            c.ptratio_2_hist = np.histogram(c.ptratio_2, bins)[0]
            c.ptratio_3 = c.arrays['x_jes_3'][c.sel & (c.arrays['x_jes_3']!=0.) & (c.arrays['x_jes_3']!=-100.)] + 1.
            c.ptratio_3_hist = np.histogram(c.ptratio_3, bins)[0]

            ax.set_title(title)
            ax.step(bins[:-1], c.ptratio_1_hist / (c.ptratio_1_hist.sum() if density else 1.) , label='J1')
            ax.step(bins[:-1], c.ptratio_2_hist / (c.ptratio_2_hist.sum() if density else 1.) , label='J2')
            ax.step(bins[:-1], c.ptratio_3_hist / (c.ptratio_3_hist.sum() if density else 1.) , label='J3')
            ax.set_xlabel(r'$p_{T}^{gen} / p_{T}^{reco}$')
            ax.legend()
        plt.savefig('tmp.png', bbox_inches='tight')
        common.imgcat('tmp.png')


if __name__ == '__main__':
    scripter.run()