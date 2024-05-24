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


from syst import BINS, NORMALIZE, DST, MTHistogram, basename, Plot, use_dc_binning

if __name__ == '__main__' and common.pull_arg('--dc', action='store_true').dc:
    use_dc_binning()
    NORMALIZE = True
    common.logger.info(f'Using binning for DC: {BINS}')
else:
    common.logger.info(f'Using binning for plots: {BINS}')


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

    arrays = arrays.copy()
    a = arrays.array

    # Before correcting, save the current pT and phi (needed for MET correction)
    pt_before = a['Jets.fCoordinates.fPt']
    eta_before = a['Jets.fCoordinates.fEta']
    phi_before = a['Jets.fCoordinates.fPhi']
    energy_before = a['Jets.fCoordinates.fE']

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
            corr = 1+x_jes
        elif var == 'down':
            jes_down = 1-x_jes
            corr = ak.where(jes_down<0., 0., jes_down)

        a[f'Jets{conesizestr}.fCoordinates.fPt'] = corr * a[f'Jets{conesizestr}.fCoordinates.fPt']
        a[f'Jets{conesizestr}.fCoordinates.fE'] = corr * a[f'Jets{conesizestr}.fCoordinates.fE']


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

    return arrays

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


@scripter
def skim_for_truth_stats():
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    do_presel = common.pull_arg('-p', '--presel', action='store_true').presel

    common.logger.info(f'Preprocessing JES info for {rootfile}')
    svj.BRANCHES_GENONLY.extend([
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

    for truth_match_type in ['partial', 'full', 'both']:
        common.logger.info(f'Loading {rootfile}')
        arrays = svj.open_root(rootfile)

        if do_presel:
            common.logger.info('Applying preselection...')
            arrays = svj.filter_preselection(arrays)

        common.logger.info(f'Applying JES for truth match type {truth_match_type}')
        apply_jes(arrays, 'up', truth_match_type)
        
        sel_corrected = arrays.array['x_jes_15'] != 0.
        match_count = ak.sum(sel_corrected, axis=-1)

        for n, count in zip(*np.unique(match_count, return_counts=True)):
            perc = 100.*count / len(match_count)
            print(f'{n} matching jet(s): {count:7} ({perc:.2f}%)')

        genparticles_pt = arrays.array['GenParticles.fCoordinates.fPt']
        genparticles_eta = arrays.array['GenParticles.fCoordinates.fEta']
        genparticles_phi = arrays.array['GenParticles.fCoordinates.fPhi']
        genparticles_pdgid = arrays.array['GenParticles_PdgId']
        genparticles_status = arrays.array['GenParticles_Status']

        select_dq = ((np.abs(genparticles_pdgid)==4900101) & (genparticles_status==71))
        select_z = genparticles_pdgid == 4900023

        out = ak.Array({
            'match_count' : match_count,
            'xjes' : arrays.array['x_jes_15'],
            'pt' : arrays.array['JetsAK15.fCoordinates.fPt'],
            'eta' : arrays.array['JetsAK15.fCoordinates.fEta'],
            'phi' : arrays.array['JetsAK15.fCoordinates.fPhi'],
            'eta_gen' : arrays.array['GenJetsAK15.fCoordinates.fEta'],
            'phi_gen' : arrays.array['GenJetsAK15.fCoordinates.fPhi'],
            'pt_dq' : genparticles_pt[select_dq],
            'eta_dq' : genparticles_eta[select_dq],
            'phi_dq' : genparticles_phi[select_dq],
            'pt_z' : genparticles_pt[select_z],
            'eta_z' : genparticles_eta[select_z],
            'phi_z' : genparticles_phi[select_z],
            })
        outfile = 'truth_match_' + truth_match_type + '.parquet'
        if do_presel:
            outfile = outfile.replace('.parquet', '_presel.parquet')
        else:
            outfile = outfile.replace('.parquet', '_nopresel.parquet')
        common.logger.info(f'Dumping to {outfile}')
        ak.to_parquet(out, outfile)


@scripter
def study_truth_match():
    pqfiles = common.pull_arg('pqfiles', type=str, nargs='+').pqfiles
    pt_cut = common.pull_arg('--ptcut', type=float).ptcut
    full = ak.from_parquet([f for f in pqfiles if 'full' in f][0])
    partial = ak.from_parquet([f for f in pqfiles if 'partial' in f][0])
    both = ak.from_parquet([f for f in pqfiles if 'both' in f][0])

    for c, truth_match_type in zip([partial, full, both], ['partial', 'full', 'both']):
        N = len(c)
        print(f'\n{truth_match_type=}, {N=}')

        for match_count, count in zip(*np.unique(c['match_count'], return_counts=True)):
            print(f'  # of matching jets=={match_count}: {count:>8} occurences ({100.*count/len(c["match_count"]):.2f}%)')    

        xjes = c['xjes']
        if pt_cut:
            is_matched = (xjes != 0.) & (c['pt'] > pt_cut)
        else:
            is_matched = (xjes != 0.)

        match_1 = ak.sum(ak.fill_none(ak.firsts(is_matched), False))
        match_2 = ak.sum(ak.fill_none(ak.firsts(is_matched[:,1:]), False))
        match_3 = ak.sum(ak.fill_none(ak.firsts(is_matched[:,2:]), False))
        match_4plus = ak.sum(ak.sum(is_matched[:,3:], axis=-1))

        # match_2 = ak.sum(ak.fill_none(ak.firsts(xjes[:,1:]), 0.) != 0.)
        # match_3 = ak.sum(ak.fill_none(ak.firsts(xjes[:,2:]), 0.) != 0.)

        # if pt_cut:
        #     match_4plus = ak.sum((ak.sum(xjes[:,3:], axis=-1) != 0.) & (ak.sum(c['pt'][:,3:] > pt_cut, axis=-1)>0))
        # else:
        #     match_4plus = ak.sum(ak.sum(xjes[:,3:], axis=-1) != 0.)

        print('')
        print(f'  freq of J1 matching:  {match_1:>7}  ({100.*match_1/N:.2f}%)')
        print(f'  freq of J2 matching:  {match_2:>7}  ({100.*match_2/N:.2f}%)')
        print(f'  freq of J3 matching:  {match_3:>7}  ({100.*match_3/N:.2f}%)')
        print(f'  freq of J4+ matching: {match_4plus:>7}  ({100.*match_4plus/N:.2f}%)')


@scripter
def study_truth_match_counts():
    pqfiles = common.pull_arg('pqfiles', type=str, nargs='+').pqfiles
    full = ak.from_parquet([f for f in pqfiles if 'full' in f][0])
    partial = ak.from_parquet([f for f in pqfiles if 'partial' in f][0])
    both = ak.from_parquet([f for f in pqfiles if 'both' in f][0])

    import jes_numba

    c = both

    for match_count, count in zip(*np.unique(c['match_count'], return_counts=True)):
        print(f'{match_count=}: {count:>8} occurences  {100.*count/len(c["match_count"]):.2f}%')    

    for i in np.nonzero(c['match_count'] > 2)[0]:
        xjes = c["xjes"][i]
        pt = c["pt"][i]
        eta = c["eta"][i]
        phi = c["phi"][i]
        eta_gen = c["eta_gen"][i]
        phi_gen = c["phi_gen"][i]
        eta_z = c["eta_z"][i]
        phi_z = c["phi_z"][i]
        eta_dq = c["eta_dq"][i]
        phi_dq = c["phi_dq"][i]
        
        gen_index = jes_numba.gen_reco_matching(eta, phi, eta_gen, phi_gen)

        matched_eta_gen = eta_gen.to_numpy()[gen_index]
        matched_eta_gen[gen_index==-1] = 0.
        matched_phi_gen = phi_gen.to_numpy()[gen_index]
        matched_phi_gen[gen_index==-1] = 0.

        dr_z = svj.calc_dr(
            eta_z.to_numpy(), phi_z.to_numpy(),
            matched_eta_gen, matched_phi_gen
            )

        print(f"{i:>5}")
        print(f"  pt corr:      {list(pt)}")
        print(f"  xjes:         {list(xjes)}")
        print(f"  eta:          {list(eta)}")
        print(f"  phi:          {list(phi)}")
        print(f"  mtch eta gen: {list(matched_eta_gen)}")
        print(f"  mtch phi gen: {list(matched_phi_gen)}")
        print(f"  eta_z:        {list(eta_z)}")
        print(f"  phi_z:        {list(phi_z)}")
        print(f"  eta_dq:       {list(eta_dq)}")
        print(f"  phi_dq:       {list(phi_dq)}")
        print(f"  dr_z:         {list(dr_z)}")



def load_columns(skimfiles):
    """
    Organizes and loads columns from the skim files
    """
    out = {}
    for skim in skimfiles:
        if 'central' in skim:
            out['central'] = svj.Columns.load(skim)
            out['central'].metadata['name'] = 'central'
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
        col.metadata['name'] = f'jes_{truth_match_type}_{direction}'

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
    merge_into = common.pull_arg('-m', '--merge', type=str).merge
    central, both_up, full_up, partial_up, both_down, full_down, partial_down = load_columns(skimfiles)
    all = [central, both_up, full_up, partial_up, both_down, full_down, partial_down]

    if merge_into:
        common.logger.info(f'Merging JES hists into {merge_into}')
        with open(merge_into, 'r') as f:
            out = json.load(f)
    else:
        out = {}

    sel = mask_cutbased(central) if selection=='cutbased' else None
    central = MTHistogram(central.arrays['mt'][sel])

    for c in all:
        if merge_into and 'both' not in c.metadata['name']: continue
        sel = mask_cutbased(c) if selection=='cutbased' else None
        h = MTHistogram(c.arrays['mt'][sel])
        if NORMALIZE: h.vals /= central.norm
        h.metadata.update(c.metadata)
        out[c.metadata['name']] = h.json()
    
    if merge_into:
        outfile = merge_into
    else:
        outfile = f'{selection}_{basename(central.metadata)}_jes_hists.json'

    common.logger.info(f'Dumping the following to {outfile}:\n{repr_dict(out)}')
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=4)


@scripter
def plot():
    selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
    histfile = common.pull_arg('histfile', type=str).histfile

    with open(histfile, 'r') as f:
        hists = json.load(f)

    for truth_match_type in ['partial', 'full', 'both']:
        plot = Plot(selection)

        central = MTHistogram.from_dict(hists['Central'])
        up = MTHistogram.from_dict(hists[truth_match_type + ' up'])
        down = MTHistogram.from_dict(hists[truth_match_type + ' down'])

        plot.plot_hist(central, label='central')
        plot.plot_hist(up, central, label='up')
        plot.plot_hist(down, central, label='down')

        plot.legend_title = truth_match_type.capitalize() + ' truth match'
        if truth_match_type == 'both':
            plot.legend_title = 'Truth matched'
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
    bins = np.linspace(0., 3., 60)
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
            c.pt_ratio = np.concatenate((c.ptratio_1, c.ptratio_2, c.ptratio_3))
            c.pt_ratio_hist = np.histogram(c.pt_ratio, bins)[0]

            ax.set_title(title)
            ax.step(bins[:-1], c.ptratio_1_hist / (c.ptratio_1_hist.sum() if density else 1.) , label='J1')
            ax.step(bins[:-1], c.ptratio_2_hist / (c.ptratio_2_hist.sum() if density else 1.) , label='J2')
            ax.step(bins[:-1], c.ptratio_3_hist / (c.ptratio_3_hist.sum() if density else 1.) , label='J3')
            ax.set_xlabel(r'$p_{T}^{gen} / p_{T}^{reco}$')
            ax.legend()
        plt.savefig('tmp.png', bbox_inches='tight')
        common.imgcat('tmp.png')

    # Compare partial with full; if similar enough, we can do 1 syst unc
    import scipy.stats
    p = scipy.stats.ks_2samp(partial_up.pt_ratio, full_up.pt_ratio).pvalue
    print(f'p-value KS partial/full: {p}')


    emd = scipy.stats.wasserstein_distance(
        partial_up.pt_ratio_hist / partial_up.pt_ratio_hist.sum(),
        full_up.pt_ratio_hist / full_up.pt_ratio_hist.sum(),
        )
    print(f'EMD partial/full: {emd:.2e}')

    plot = Plot(selection)
    partial = Histogram(bins, partial_up.pt_ratio_hist / partial_up.pt_ratio_hist.sum())
    full = Histogram(bins, full_up.pt_ratio_hist / full_up.pt_ratio_hist.sum())
    plot.plot_hist(full, label='Full')
    plot.plot_hist(partial, full, label='Partial')
    plot.top.set_ylabel('A.U.')
    plot.bot.set_ylabel('Partial/Full')
    plot.bot.set_xlabel(r'$p_{T}^{gen} / p_{T}^{reco}$')
    plot.legend_title = f'EMD={emd:.2e}'
    plot.save('plots/pt_jes_partial_full.png')


if __name__ == '__main__':
    scripter.run()
