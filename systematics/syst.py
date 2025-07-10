import os, os.path as osp, sys, json, re
from time import strftime

THIS_DIR = osp.dirname(osp.abspath(__file__))
MAIN_DIR = osp.dirname(THIS_DIR)
sys.path.append(MAIN_DIR)

import common
from common import apply_cutbased, MTHistogram, Histogram
from produce_histograms import repr_dict
from cutflow_table import format_table

import tqdm
import svj_ntuple_processing as svj
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

scripter = common.Scripter()

DST = osp.join(THIS_DIR, 'skims')

# Better for studying the systs
BINS = np.linspace(180, 600, 40)
NORMALIZE = False

def use_dc_binning():
    # Better for datacards
    global BINS
    global NORMALIZE
    BINS = common.MT_BINS
    NORMALIZE = True

if __name__ == '__main__':
    if common.pull_arg('--dc', action='store_true').dc:
        use_dc_binning()
        common.logger.info(f'Using binning for DC: {BINS}')
    else:
        common.logger.info(f'Using binning for plots: {BINS}')


cms_style = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    # 
    "mathtext.fontset": "custom",
    "mathtext.rm": "helvetica",
    "mathtext.bf": "helvetica:bold",
    "mathtext.sf": "helvetica",
    "mathtext.it": "helvetica:italic",
    "mathtext.tt": "helvetica",
    "mathtext.cal": "helvetica",
    # 
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    'text.usetex' : True,    
    }

def set_mpl_fontsize(small=16, medium=22, large=26):
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title
    from matplotlib.pyplot import style as plt_style
    plt_style.use(cms_style)
    plt.rc('text', usetex=True)
    plt.rc(
        'text.latex',
        preamble=(
            r'\usepackage{helvet} '
            r'\usepackage{sansmath} '
            r'\sansmath '
            )    
        )

def put_on_cmslabel(ax, text='Simulation Preliminary', year=2018):
    fontsize = 27
    ax.text(
        .0, 1.005,
        r'\textbf{CMS}\,\fontsize{21pt}{3em}\selectfont{}{\textit{'+text+'}}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=fontsize
        )
    ax.text(
        1.0, 1.005,
        '{} (13 TeV)'.format(year),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=int(19./23. * fontsize)
        )

import matplotlib.pyplot as plt
set_mpl_fontsize()



def basename(meta):
    return (
        f'mz{meta["mz"]:.0f}_rinv{meta["rinv"]:.1f}_mdark{meta["mdark"]:.0f}'
        )


@scripter
def skim_jec_jer():
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    array = svj.open_root(rootfile, load_gen=True, load_jerjec=True)
    for var_name, appl in [
        ('jer_up',   svj.apply_jer_up),
        ('jer_down', svj.apply_jer_down),
        ('jec_up',   svj.apply_jec_up),
        ('jec_down', svj.apply_jec_down),
        ]:
        variation = appl(array)
        variation = svj.filter_preselection(variation)
        cols = svj.bdt_feature_columns(variation)
        cols.save(f'{DST}/{basename(array.metadata)}_{var_name}.npz')


@scripter
def skim_scale():
    """
    Creates the skim with scale weights
    """
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    array = svj.open_root(rootfile)

    # Compute normalizations before applying cuts
    w = array.array['ScaleWeights'].to_numpy()
    print(w.shape)
    w = w[:,np.array([0,1,2,3,4,6,8])] # Throw away the mur/muf .5/2 and 2/.5 variations
    array.metadata['norm_central'] = w[:,0].sum()
    array.metadata['norm_up'] = np.max(w, axis=-1).sum()
    array.metadata['norm_down'] = np.min(w, axis=-1).sum()
    array.metadata['factor_up'] = array.metadata['norm_central'] / array.metadata['norm_up']
    array.metadata['factor_down'] = array.metadata['norm_central'] / array.metadata['norm_down']
    svj.logger.info(
        'Scale unc:'
        f"\n    norm_central = {array.metadata['norm_central']:.5f}"
        f"\n    norm_up      = {array.metadata['norm_up']:.5f}"
        f"\n    norm_down    = {array.metadata['norm_down']:.5f}"
        f"\n    factor_up    = {array.metadata['factor_up']:.5f}"
        f"\n    factor_down  = {array.metadata['factor_down']:.5f}"
        )
    
    common.logger.info('Running preselection now')
    array = svj.filter_preselection(array)
    cols = svj.bdt_feature_columns(array, save_scale_weights=True)
    cols.save(f'{DST}/{basename(array.metadata)}_scaleunc.npz')



@scripter
def plot_pu_weight_dist():
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    svj.BRANCHES_GENONLY.extend(['puWeight'])
    array = svj.open_root(rootfile)

    w = array.array['puWeight'].to_numpy()
    with common.quick_ax(outfile='puweight.png') as ax:
        ax.hist(w)
        ax.set_xlabel('puWeight')

    with common.quick_ax(outfile='puweight.png') as ax:
        ax.hist(array.array['Weight'].to_numpy())
        ax.set_xlabel('Weight')



@scripter
def skim_central():
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    svj.BRANCHES_GENONLY.extend([
        'PDFweights', 'PSweights',
        'GenJetsAK15.fCoordinates.fPt',
        'GenJetsAK15.fCoordinates.fEta',
        'GenJetsAK15.fCoordinates.fPhi',
        'GenJetsAK15.fCoordinates.fE',
        'puSysUp', 'puSysDown'
        ])
    array = svj.open_root(rootfile)

    # ______________________________
    # Work before applying preselection

    # PDF weights
    common.logger.info('Calculating PDFweight norm factors')
    pdf_weights = array.array['PDFweights'].to_numpy()
    pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
    # mu and sigma _per event_
    mu = np.mean(pdf_weights, axis=1)
    sigma = np.std(pdf_weights, axis=1)
    # Normalization factors for the weights
    pdfw_norm_up   = np.mean(mu+sigma)
    pdfw_norm_down = np.mean(mu-sigma)

    # ______________________________
    # Apply preselection and save needed vars

    common.logger.info('Running preselection now')
    array = svj.filter_preselection(array)
    cols = svj.bdt_feature_columns(array, save_scale_weights=True)

    # Save PDF normalization and weights
    cols.metadata['pdfw_norm_up'] = pdfw_norm_up
    cols.metadata['pdfw_norm_down'] = pdfw_norm_down
    cols.arrays['pdf_weights'] = array.array['PDFweights'].to_numpy()

    # Save PS weights
    ps_weights = array.array['PSweights'].to_numpy()
    cols.arrays['ps_isr_up'] = ps_weights[:,6]
    cols.arrays['ps_isr_down'] = ps_weights[:,8]
    cols.arrays['ps_fsr_up'] = ps_weights[:,7]
    cols.arrays['ps_fsr_down'] = ps_weights[:,9]

    # Save PU weights
    cols.arrays['pu_central'] = array.array['puWeight'].to_numpy()
    cols.arrays['pu_sys_up'] = array.array['puSysUp'].to_numpy()
    cols.arrays['pu_sys_down'] = array.array['puSysDown'].to_numpy()

    cols.save(f'{DST}/{basename(array.metadata)}_central.npz')



@scripter
def produce():
    selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
    skims = common.pull_arg('skims', type=str, nargs='+').skims
    def get_by_tag(tag):
        return [s for s in skims if 'central' in s][0]
        
    common.logger.warning('Using full Run2 lumi for 2018 now! Correct this when running on all years!')
    lumi = 137.2*1e3 # Use full Run2 lumi for now

    central_skim = get_by_tag('central')

    central = svj.Columns.load(central_skim)
    if selection=='cutbased': central = apply_cutbased(central)
    mt = central.to_numpy(['mt']).ravel()
    w = common.get_event_weight(central)

    # Scale
    scale_weight = central.to_numpy(['scaleweights'])[sel][:, np.array([0,1,2,3,4,6,8])]
    weight_up = w * np.max(scale_weight, axis=-1)
    weight_down = w * np.min(scale_weight, axis=-1)
    mth_scale_up = MTHistogram(mt, weight_up)
    mth_scale_down = MTHistogram(mt, weight_down)

    # JEC/JER/JES
    def mth_jerjecjes(tag):
        col = svj.Columns.load(get_by_tag(tag))
        if selection=='cutbased': col = apply_cutbased(col)
        mt = col.to_numpy(['mt']).flatten()
        w = common.get_event_weight(col)
        return MTHistogram(mt, w)
    jer_up = mth_jerjecjes('jer_up')
    jer_down = mth_jerjecjes('jer_down')
    jec_up = mth_jerjecjes('jec_up')
    jec_down = mth_jerjecjes('jec_down')
    jes_up = mth_jerjecjes('jesup_both')
    jes_down = mth_jerjecjes('jesdown_both')

    # PS
    ps_weights = w[:,None] * central.to_numpy(['ps_isr_up', 'ps_isr_down',
                                       'ps_fsr_up', 'ps_fsr_down'])[sel]
    mth_isr_up   = MTHistogram(mt, ps_weights[:,0])
    mth_isr_down = MTHistogram(mt, ps_weights[:,1])
    mth_fsr_up   = MTHistogram(mt, ps_weights[:,2])
    mth_fsr_down = MTHistogram(mt, ps_weights[:,3])

    # PU
    pu_weights = central.to_numpy(['pu_sys_up', 'pu_sys_down'])[sel]
    mth_pu_up = MTHistogram(mt, pu_weights[:,0])
    mth_pu_down = MTHistogram(mt, pu_weights[:,1])

    # PDF
    pdf_weights = central.to_numpy(['pdf_weights'])[sel]
    pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
    mu_pdf = np.mean(pdf_weights, axis=1)
    sigma_pdf = np.std(pdf_weights, axis=1)
    pdfw_up = (mu_pdf+sigma_pdf) / central.metadata['pdfw_norm_up']
    pdfw_down = (mu_pdf-sigma_pdf) / central.metadata['pdfw_norm_down']
    mth_pdf_up = MTHistogram(mt, w*pdfw_up)
    mth_pdf_down = MTHistogram(mt, w*pdfw_down)

    # MC stats
    mth_central = MTHistogram(mt, w)
    mc_stat_err = np.sqrt(np.histogram(mt, bins=mth_central.binning, weights=w**2)[0])

    mc_stat_up = []
    mc_stat_down = []
    for i in range(mth_central.nbins):
        mth = mth_central.copy()
        mth.vals[i] += mc_stat_err[i]
        mc_stat_up.append(mth)
        mth = mth_central.copy()
        mth.vals[i] -= mc_stat_err[i]
        mc_stat_down.append(mth)

    


        






@scripter
def produce_scale_hist(selection=None, skimfile=None):
    dump = False
    if selection is None:
        dump = True
        selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
        skimfile = common.pull_arg('skim', type=str).skim

    mur_muf = [
        (1., 1.), # 0
        (1., 2.), # 1
        (1., .5), # 2
        (2., 1.), # 3
        (2., 2.), # 4
        (2., .5), # 5 <-
        (.5, 1.), # 6
        (.5, 2.), # 7 <-
        (.5, .5), # 8
        ]
    mur_muf_titles = [ rf'$\mu_{{R}}={mur:.1f}$ $\mu_{{F}}={muf:.1f}$' for mur, muf in mur_muf ]
    col = svj.Columns.load(skimfile)
    if selection=='cutbased': col = apply_cutbased(col)

    mt = col.to_numpy(['mt']).ravel()
    scale_weight = col.to_numpy(['scaleweights'])

    mt = mt[sel]
    scale_weight = scale_weight[sel]

    # Drop column 5 and 7: the .5/2 and 2/.5 variations
    scale_weight = scale_weight[:, np.array([0,1,2,3,4,6,8])]
    del mur_muf[7]; del mur_muf[5]
    del mur_muf_titles[7]; del mur_muf_titles[5]

    assert scale_weight.shape == (len(mt), 7)

    weight_up = np.max(scale_weight, axis=-1)
    weight_down = np.min(scale_weight, axis=-1)

    central = MTHistogram(mt)
    up = MTHistogram(mt, weight_up)
    down = MTHistogram(mt, weight_down)

    if NORMALIZE:
        up.vals /= central.norm
        down.vals /= central.norm
        central.vals /= central.norm

    out = {
        'selection' : selection,
        'central' : central.json(),
        'up' : up.json(),
        'down' : down.json(),
        }
    if dump:
        outfile = f'scale_{osp.basename(skimfile).replace(".npz", "")}.json'
        with open(outfile, 'w') as f:
            json.dump(out, f)
    return out

@scripter
def produce_jecjer_hist(selection=None, skimfiles=None):
    dump = False
    if selection is None:
        dump = True
        selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
        skimfiles = common.pull_arg('skimfiles', type=str, nargs='+').skimfiles

    files = {}
    files['jer_up'] = [f for f in skimfiles if 'jer_up' in f][0]
    files['jer_down'] = [f for f in skimfiles if 'jer_down' in f][0]
    files['jec_up'] = [f for f in skimfiles if 'jec_up' in f][0]
    files['jec_down'] = [f for f in skimfiles if 'jec_down' in f][0]
    files['central'] = [f for f in skimfiles if 'central' in f][0]

    out = {'selection': selection}
    for var, file in files.items():
        col = svj.Columns.load(file)
        if selection=='cutbased': col = apply_cutbased(col)
        mt = col.to_numpy(['mt']).flatten()
        out[var] = MTHistogram(mt)

    for key in out:
        if key == 'selection': continue
        if NORMALIZE: out[key].vals /= out['central'].norm
        out[key] = out[key].json()

    if dump:
        outfile = f'jerjec_{osp.basename(files["central"]).replace(".npz", "")}.json'
        with open(outfile, 'w') as f:
            json.dump(out, f, indent=4)
    return out



@scripter
def produce_hists(selection=None, skimfile=None):
    """
    Produces syst histograms for:
    - PU
    - PS
    - PDF weights

    Does *not* do:
    - JES
    - JER/JEC
    - Scale
    """
    dump = False
    if selection is None:
        dump = True
        selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
        skimfile = common.pull_arg('skim', type=str).skim
    col = svj.Columns.load(skimfile)
    if selection=='cutbased': col = apply_cutbased(col)

    mt = col.to_numpy(['mt']).flatten()
    central = MTHistogram(mt)
    central.metadata.update(col.metadata)
    out = {'selection' : selection, 'central' : central}

    # PS
    ps_isr_up = col.to_numpy(['ps_isr_up'])[sel,0]
    ps_isr_down = col.to_numpy(['ps_isr_down'])[sel,0]
    ps_fsr_up = col.to_numpy(['ps_fsr_up'])[sel,0]
    ps_fsr_down = col.to_numpy(['ps_fsr_down'])[sel,0]
    out.update({
        'isr_up' : MTHistogram(mt, ps_isr_up),
        'isr_down' : MTHistogram(mt, ps_isr_down),
        'fsr_up' : MTHistogram(mt, ps_fsr_up),
        'fsr_down' : MTHistogram(mt, ps_fsr_down),
        })

    # PU
    pu_central = col.to_numpy(['pu_central'])[sel,0]
    pu_sys_up = col.to_numpy(['pu_sys_up'])[sel,0]
    pu_sys_down = col.to_numpy(['pu_sys_down'])[sel,0]
    out.update({
        'pu_up' : MTHistogram(mt, pu_sys_up/pu_central),
        'pu_down' : MTHistogram(mt, pu_sys_down/pu_central)
        })
    
    # PDF
    pdf_weights = col.to_numpy(['pdf_weights'])[sel]
    pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
    mu_pdf = np.mean(pdf_weights, axis=1)
    sigma_pdf = np.std(pdf_weights, axis=1)
    pdfw_up = (mu_pdf+sigma_pdf) / col.metadata['pdfw_norm_up']
    pdfw_down = (mu_pdf-sigma_pdf) / col.metadata['pdfw_norm_down']
    out.update({
        'pdf_up' : MTHistogram(mt, pdfw_up),
        'pdf_down' : MTHistogram(mt, pdfw_down)
        })


    central_norm = out['central'].norm
    for key in out:
        if key == 'selection': continue
        if NORMALIZE: out[key].vals /= central_norm
        out[key] = out[key].json()

    if dump:
        outfile = f'systs_{selection}_{osp.basename(skimfile).replace(".npz", "")}{"_normalized" if NORMALIZE else ""}.json'
        with open(outfile, 'w') as f:
            json.dump(out, f)
    return out


@scripter
def produce_all():
    selection = common.pull_arg('selection', type=str, choices=['cutbased', 'bdt']).selection
    skimfiles = common.pull_arg('skims', type=str, nargs='+').skims

    central = [f for f in skimfiles if 'central' in osp.basename(f)][0]
    out = produce_hists(selection, central)

    scale_skimfile = [f for f in skimfiles if 'scale' in osp.basename(f)][0]
    scale = produce_scale_hist(selection, scale_skimfile)
    out['scale_up'] = scale['up']
    out['scale_down'] = scale['down']

    jerjec = produce_jecjer_hist(selection, skimfiles)
    out['jer_up'] = jerjec['jer_up']
    out['jer_down'] = jerjec['jer_down']
    out['jec_up'] = jerjec['jec_up']
    out['jec_down'] = jerjec['jec_down']

    outfile = strftime(f'syst_{basename(out["central"]["metadata"])}_{selection}_%b%d.json')
    if NORMALIZE: outfile = outfile.replace('.json', '_normalized.json')
    common.logger.info(f'Dumping the following to {outfile}:\n{repr_dict(out)}')
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=4)


class Plot:
    def __init__(self, selection):
        self.selection = selection
        self.fig, (self.top, self.bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))
        self.top.set_yscale('log')
        put_on_cmslabel(self.top)
        self.top.set_ylabel('Event count')
        self.bot.set_ylabel('Ratio to central')
        self.bot.set_xlabel(r'$m_{T}$ (GeV)')
        self.legend_title = None

    def plot_hist(self, hist, denominator=None, label=None):
        ratio = np.ones(hist.nbins) if denominator is None else hist.vals / denominator.vals
        l = self.top.step(hist.binning[:-1], hist.vals, where='post', label=label)[0]
        self.bot.step(hist.binning[:-1], ratio, where='post', label=label, color=l.get_color())

    def save(self, outfile='tmp.png', pdf=True):
        self.top.text(
            0.02, 0.02,
            'Cut-based' if self.selection=='cutbased' else 'BDT',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=self.top.transAxes,
            usetex=True,
            fontsize=25
            )
        self.top.legend(title=self.legend_title)
        outdir = osp.dirname(osp.abspath(outfile))
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
        plt.savefig(outfile.replace('.png', '.pdf'), bbox_inches="tight")
        common.imgcat(outfile)


@scripter
def plot(histfile=None):
    if histfile is None: histfile = common.pull_arg('histfile', type=str).histfile
    with open(histfile, 'r') as f:
        hists = json.load(f)

    central = MTHistogram.from_dict(hists['central'])

    for label, up, down in [
        ('PU', hists['pu_up'], hists['pu_down']),
        ('PS: ISR', hists['isr_up'], hists['isr_down']),
        ('PS: FSR', hists['fsr_up'], hists['fsr_down']),
        ('PDF', hists['pdf_up'], hists['pdf_down']),
        ]:
        up = MTHistogram.from_dict(up)
        down = MTHistogram.from_dict(down)
        plot = Plot(hists['selection'])
        plot.plot_hist(central, label='Central')
        plot.plot_hist(up, central, 'Up')
        plot.plot_hist(down, central, 'Down')
        plot.legend_title = label
        outfile = 'plots/' + re.sub(r'\W+', '', label.replace(' ', '_').lower()) + '.png'
        plot.save(outfile)


@scripter
def plot_scale(histfile=None):
    if histfile is None: histfile = common.pull_arg('histfile', type=str).histfile
    with open(histfile, 'r') as f:
        hists = json.load(f)
    plot = Plot(hists['selection'])
    central = MTHistogram.from_dict(hists['central'])
    up = MTHistogram.from_dict(hists['up'])
    down = MTHistogram.from_dict(hists['down'])
    plot.plot_hist(central, label='Central')
    plot.plot_hist(up, central, 'Up')
    plot.plot_hist(down, central, 'Down')
    plot.save('plots/scale.png')


@scripter
def plot_jecjer(histfile=None):
    if histfile is None: histfile = common.pull_arg('histfile', type=str).histfile
    with open(histfile, 'r') as f:
        hists = json.load(f)

    jer_up = MTHistogram.from_dict(hists['jer_up'])
    jer_down = MTHistogram.from_dict(hists['jer_down'])
    jec_up = MTHistogram.from_dict(hists['jec_up'])
    jec_down = MTHistogram.from_dict(hists['jec_down'])
    central = MTHistogram.from_dict(hists['central'])

    plot = Plot(hists['selection'])
    plot.plot_hist(central, label='Central')
    plot.plot_hist(jer_up, central, 'Up')
    plot.plot_hist(jer_down, central, 'Down')
    plot.legend_title = 'JER'
    plot.save('plots/jer.png')

    plot = Plot(hists['selection'])
    plot.plot_hist(central, label='Central')
    plot.plot_hist(jec_up, central, 'Up')
    plot.plot_hist(jec_down, central, 'Down')
    plot.legend_title = 'JEC'
    plot.save('plots/jec.png')


@scripter
def total_yield():
    histfiles = common.pull_arg('histfiles', type=str, nargs='+').histfiles
    hists = {}
    for f in histfiles:
        if 'jerjec' in osp.basename(f):
            key = 'jerjec'
        elif 'scale' in osp.basename(f):
            key = 'scale'
        else:
            key = 'central'
        with open(f, 'r') as fp:
            hists[key] = json.load(fp)

    table = []

    jer_up = MTHistogram.from_dict(hists['jerjec']['jer_up'])
    jer_down = MTHistogram.from_dict(hists['jerjec']['jer_down'])
    jec_up = MTHistogram.from_dict(hists['jerjec']['jec_up'])
    jec_down = MTHistogram.from_dict(hists['jerjec']['jec_down'])
    central = MTHistogram.from_dict(hists['jerjec']['central'])
    table.append(('jer', jer_up.norm/central.norm-1, jer_down.norm/central.norm-1))
    table.append(('jec', jec_up.norm/central.norm-1, jec_down.norm/central.norm-1))

    central = MTHistogram.from_dict(hists['scale']['central'])
    up = MTHistogram.from_dict(hists['scale']['up'])
    down = MTHistogram.from_dict(hists['scale']['down'])
    table.append(('scale', up.norm/central.norm-1, down.norm/central.norm-1))

    central = MTHistogram.from_dict(hists['central']['central'])
    for label, up, down in [
        ('pu', 'pu_up', 'pu_down'),
        ('ps_isr', 'isr_up', 'isr_down'),
        ('ps_fst', 'fsr_up', 'fsr_down'),
        ('pdf', 'pdf_up', 'pdf_down'),
        ]:
        up = MTHistogram.from_dict(hists['central'][up])
        down = MTHistogram.from_dict(hists['central'][down])
        table.append((label, up.norm/central.norm-1, down.norm/central.norm-1))

    print(hists['selection'])
    print(format_table(table, ndec=4))


@scripter
def total_yield_2():
    histfile = common.pull_arg('histfile', type=str).histfile
    with open(histfile, 'r') as f:
        hists = json.load(f)

    central = MTHistogram.from_dict(hists['central'])

    systs = set()
    for k in hists.keys():
        if k.endswith('_up') or k.endswith('_down'):
            systs.add(k.replace('_up','').replace('_down',''))
    
    table = []
    for syst in systs:
        up = MTHistogram.from_dict(hists[f'{syst}_up'])
        down = MTHistogram.from_dict(hists[f'{syst}_down'])
        table.append((syst, up.norm-1, down.norm-1))

    table.sort()

    print(hists['selection'])
    print(format_table(table, ndec=4))
    


@scripter
def ls():
    jsonfile = common.pull_arg('jsonfile', type=str).jsonfile
    with open(jsonfile, 'r') as f:
        d = json.load(f)
    print(repr_dict(d))
    

if __name__ == '__main__': scripter.run()
