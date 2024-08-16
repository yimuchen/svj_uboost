import os, os.path as osp, sys, json, re, math
from time import strftime
from collections import defaultdict

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

import svj_ntuple_processing as svj
import common
from hadd_skims import expand_wildcards

THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(THIS_DIR, 'systematics'))

scripter = common.Scripter()

def change_bin_width():
    """
    Changes MT binning based on command line options
    """
    binw = common.pull_arg('--binw', type=float).binw
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    if binw is not None:
        # Testing different bin widths
        left = mtmin
        right = mtmax
        common.MTHistogram.bins = left + binw * np.arange(math.ceil((right-left)/binw)+1)
        common.MTHistogram.non_standard_binning = True
        common.logger.warning(f'Changing bin width to {binw}; MT binning: {common.MTHistogram.bins}')

def check_rebin(hist,name):
    if not common.MTHistogram.non_standard_binning:
        return hist
    msg = []
    if hist.binning[0]>common.MTHistogram.bins[0]:
        msg.append(f'left {hist.binning[0]}>{common.MTHistogram.bins[0]}')
    if hist.binning[-1]<common.MTHistogram.bins[-1]:
        msg.append(f'right {hist.binning[-1]}>{common.MTHistogram.bins[-1]}')
    orig_width = int(hist.binning[1] - hist.binning[0])
    new_width = int(common.MTHistogram.bins[1] - common.MTHistogram.bins[0])
    rebin_factor = int(new_width/orig_width)
    rebin_mod = orig_width % new_width
    if rebin_mod!=0:
        msg.append(f'rebin {orig_width} % {new_width} = {rebin_mod}')
    if len(msg)>0:
        msg = ', '.join(msg)
        common.logger.warning(f'Hist {name} inconsistent with requested binning ({msg})')
    return hist.rebin(rebin_factor).cut(common.MTHistogram.bins[0],common.MTHistogram.bins[-1])

def rebin_dict(hists):
    if not common.MTHistogram.non_standard_binning:
        return hists
    for key in hists:
        if isinstance(hists[key],list):
            for i,entry in enumerate(hists[key]):
                hists[key][i] = check_rebin(entry,f'{key}[{i}]')
        elif isinstance(hists[key],dict):
            for k,v in hists[key].items():
                hists[key][k] = check_rebin(v,f'{key}[{k}]')
        else:
            hists[key] = check_rebin(hists[key],key)
    return hists

def rebin_name(outfile):
    if common.MTHistogram.non_standard_binning:
        binw = int(common.MTHistogram.bins[1] - common.MTHistogram.bins[0])
        left = common.MTHistogram.bins[0]
        right = common.MTHistogram.bins[-1]
        outfile = outfile.replace(
            '.json',
            f'_binw{binw:02d}_range{left:.0f}-{right:.0f}.json'
        )
    return outfile

def rebin_file(file):
    file2 = rebin_name(file)
    if file2==file:
        return file
    with open(file,'r') as f:
        mths = json.load(f, cls=common.Decoder)
    mths = rebin_dict(mths)
    with open(file2,'w') as f:
        json.dump(mths, f, cls=common.Encoder, indent=4)
    return file2

def basename(meta):
    """
    Generates a name based on signal metadata.
    """
    return (
        f'mz{meta["mz"]:.0f}_rinv{meta["rinv"]:.1f}_mdark{meta["mdark"]:.0f}'
        )

@scripter
def build_histogram(args=None):
    if args is None:
        change_bin_width()
        # Read from sys.argv
        selection = common.pull_arg('selection', type=str).selection
        lumi = common.pull_arg('--lumi', type=float, default=None, help='Luminosity in pb-1 (overrides defaults)').lumi
        year = common.pull_arg('--year', type=str, default=None, help='year (overrides metadata)').year
        fullyear = common.pull_arg('--fullyear', action="store_true", help='treat 2018 as one year instead of splitting into pre and post').fullyear
        skimfile = common.pull_arg('skimfile', type=str).skimfile
    else:
        # Use passed input
        selection, lumi, year, fullyear, skimfile = args

    def filter_bkg(cols):
        bkgs = [cols]
        # Filter empty backgrounds
        bkgs = [c for c in bkgs if len(c)]
        # Filter out QCD with pT<300
        # Only singular events pass the preselection, which creates spikes in the final bkg dist
        bkgs = common.filter_pt(bkgs, 300)
        # Same story for wjets with HT<400
        bkgs = common.filter_ht(bkgs, 400, 'wjets')
        # Filter out wjets inclusive bin - it's practically the HT<100 bin,
        # and it's giving problems
        bkgs = [c for c in bkgs if not (c.metadata['bkg_type']=='wjets' and 'htbin' not in c.metadata)]
        return bkgs[0] if len(bkgs)==1 else None

    def get_variation(var):
        return skimfile.replace(".npz",f"_{var}.npz")

    # apply final selections
    def apply_selection(cols,year):
        metadata = cols.metadata
        # HEM veto
        startHEM = 319077
        if year=="2018PRE":
            if metadata["sample_type"]=="data":
                cols = cols.select(cols.arrays['run']<startHEM)
        elif year=="2018POST":
            if metadata["sample_type"]=="data":
                cols = cols.select(cols.arrays['run']>=startHEM)
            cols = common.apply_hemveto(cols)
        # signal region
        if selection=='cutbased':
            cols = common.apply_cutbased(cols)
        elif selection.startswith('bdt='):
            wp = split_bdt(selection)
            cols = common.apply_bdtbased(cols,wp,lumi)
        elif selection=='preselection':
            pass
        else:
            raise common.InvalidSelectionException(sel=selection)
        return cols

    mths = {}
    central = svj.Columns.load(skimfile)
    metadata = central.metadata
    if metadata["sample_type"]=="bkg":
        central = filter_bkg(central)
        if central is None:
            return [""]

    if year is None: year = str(metadata["year"])
    else:
        metadata["year"] = year
        lumi = common.lumis[year]

    # process 2018 samples twice as PRE and POST
    if year=="2018" and not fullyear:
        outfiles1 = build_histogram((selection,None,"2018PRE",False,skimfile))
        outfiles2 = build_histogram((selection,None,"2018POST",False,skimfile))
        return outfiles1+outfiles2

    common.logger.info(f'Selection: {selection}')
    central = apply_selection(central,year)
    mt = central.to_numpy(['mt']).ravel()

    if metadata["sample_type"]=="data":
        w = None
    else:
        w = central.to_numpy(['puweight']).ravel()
        event_weight = common.get_event_weight(central,lumi)
        w *= event_weight
        # save event weight
        metadata['event_weight'] = common.get_single_event_weight(event_weight)
    mth_central = common.MTHistogram(mt, w)
    mth_central.metadata.update(metadata)
    mths['central'] = mth_central

    if metadata["sample_type"]=="sig":
        # Scale
        good_scales = np.array([0,1,2,3,4,6,8])
        scale_weight = central.to_numpy(['scaleweights'])
        scale_weight = scale_weight[ak.num(scale_weight,axis=1)>np.max(good_scales)]
        scale_weight = scale_weight[:, good_scales]
        if len(scale_weight):
            weight_up = w * np.max(scale_weight, axis=-1) * central.metadata['scale_factor_up']
            weight_down = w * np.min(scale_weight, axis=-1) * central.metadata['scale_factor_down']
            mths['scale_up'] = common.MTHistogram(mt, weight_up)
            mths['scale_down'] = common.MTHistogram(mt, weight_down)

        # JEC/JER/JES
        def mth_jerjecjes(tag):
            col = svj.Columns.load(get_variation(tag))
            col = apply_selection(col,year)
            mt = col.to_numpy(['mt']).flatten()
            w = col.to_numpy(['puweight']).flatten()
            event_weight = common.get_event_weight(col,lumi)
            w *= event_weight
            return common.MTHistogram(mt, w)
        # JER, JEC treated as uncorrelated between years (but 2018PRE, 2018POST always correlated)
        sysyear = get_sysyear(year)
        mths[f'jer{sysyear}_up'] = mth_jerjecjes('jer_up')
        mths[f'jer{sysyear}_down'] = mth_jerjecjes('jer_down')
        mths[f'jec{sysyear}_up'] = mth_jerjecjes('jec_up')
        mths[f'jec{sysyear}_down'] = mth_jerjecjes('jec_down')
        mths['jes_up'] = mth_jerjecjes('jesup_both')
        mths['jes_down'] = mth_jerjecjes('jesdown_both')

        # PS
        ps_weights = w[:,None] * central.to_numpy(['ps_isr_up', 'ps_isr_down', 'ps_fsr_up', 'ps_fsr_down'])
        mths['isr_up']   = common.MTHistogram(mt, ps_weights[:,0])
        mths['isr_down'] = common.MTHistogram(mt, ps_weights[:,1])
        mths['fsr_up']   = common.MTHistogram(mt, ps_weights[:,2])
        mths['fsr_down'] = common.MTHistogram(mt, ps_weights[:,3])

        # PU
        # also uncorrelated between years
        pu_weights = central.to_numpy(['puweight', 'pu_sys_up', 'pu_sys_down'])
        mths[f'pu{sysyear}_up'] = common.MTHistogram(mt, w / pu_weights[:,0] * pu_weights[:,1])
        mths[f'pu{sysyear}_down'] = common.MTHistogram(mt, w / pu_weights[:,0] * pu_weights[:,2])

        # PDF
        pdf_weights = central.to_numpy(['pdf_weights'])
        # set massive unphysical weights to physical max
        pdf_max = np.max(pdf_weights, where=pdf_weights<100, initial=1)
        pdf_weights = np.clip(pdf_weights,a_min=None,a_max=pdf_max)
        pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
        mu_pdf = np.mean(pdf_weights, axis=1)
        sigma_pdf = np.std(pdf_weights, axis=1)
        pdfw_up = (mu_pdf+sigma_pdf) / central.metadata['pdfw_norm_up']
        pdfw_down = (mu_pdf-sigma_pdf) / central.metadata['pdfw_norm_down']
        mths['pdf_up'] = common.MTHistogram(mt, w*pdfw_up)
        mths['pdf_down'] = common.MTHistogram(mt, w*pdfw_down)

        # MC stats
        mc_stat_err = np.sqrt(np.histogram(mt, bins=mth_central.binning, weights=w**2)[0])
        for i in range(mth_central.nbins):
            mth = mth_central.copy()
            mth.vals[i] += mc_stat_err[i]
            mths[f'mcstat{i}_{sysyear}_up'] = mth
            mth = mth_central.copy()
            mth.vals[i] -= mc_stat_err[i]
            mths[f'mcstat{i}_{sysyear}_down'] = mth

    # save cutflow after applying final selection & after doing any copying (to avoid duplication)
    mths['central'].cutflow = central.cutflow.copy()
    outdir = f'hists_{strftime("%Y%m%d")}'
    os.makedirs(outdir, exist_ok=True)
    process = osp.basename(skimfile).replace(".npz","")
    if metadata["sample_type"]=="data":
        # keep data era info to avoid overwriting
        process += '_'+osp.dirname(skimfile)
    outfile = f'{outdir}/{process}_sel-{selection}_year-{year}.json'
    common.logger.info(f'Dumping histograms to {outfile}')
    with open(outfile, 'w') as f:
        json.dump(mths, f, cls=common.Encoder, indent=4)
    return [outfile]

@scripter
def build_all_histograms():
    change_bin_width()
    # Read from sys.argv
    selection = common.pull_arg('selection', type=str).selection
    fullyear = common.pull_arg('--fullyear', action="store_true", help='treat 2018 as one year instead of splitting into pre and post').fullyear
    skimdir = common.pull_arg('skimdir', type=str).skimdir

    skims = expand_wildcards(skimdir)
    for skim in skims:
        build_histogram((selection, None, None, fullyear, skim))

@scripter
def merge_histograms():
    change_bin_width()
    selection = common.pull_arg('selection', type=str).selection
    histdir = common.pull_arg('histdir', type=str).histdir
    cat = common.pull_arg('--cat', type=str, required=True, choices=['sig','bkg','data']).cat
    years = common.pull_arg('--years', type=str, default=["2016","2017","2018PRE","2018POST"], nargs='*').years
    if histdir[-1]!='/': histdir += '/'

    def get_files(samples,years):
        files = []
        for year in years:
            for sample in samples:
                files += expand_wildcards(histdir+f'{sample}*_sel-{selection}_year-{year}.json')
        return files

    def get_hists(file):
        with open(file,'r') as f:
            return json.load(f, cls=common.Decoder)

    def add_hists(hist,htmp):
        if hist is None:
            hist = htmp
        else:
            # once two hists (& their cutflows) are added, event_weight no longer needed -> set to 1
            hist += htmp
            if 'event_weight' in hist.metadata:
                hist.metadata['event_weight'] = 1.0
        return hist

    outdir = histdir.replace("hists","merged")
    os.makedirs(outdir, exist_ok=True)
    def write(hists,proc):
        if selection in proc:
            outfile = f'{outdir}/{proc}.json'
        else:
            outfile = f'{outdir}/{proc}_sel-{selection}.json'
        hists = rebin_dict(hists)
        outfile = rebin_name(outfile)
        common.logger.info(f'Dumping merged histograms to {outfile}')
        with open(outfile, 'w') as f:
            json.dump(hists, f, cls=common.Encoder, indent=4)

    lumi_total = sum(common.lumis[year] for year in years)
    def assign_metadata(hist):
        hist.metadata['selection'] = selection
        hist.metadata['year'] = years
        hist.metadata['lumi'] = lumi_total

    samples = {
        "data": ["JetHT", "HTMHT"],
        "bkg": ['QCD', 'TTJets', 'ZJets', 'WJets'],
        "sig": ['SVJ'],
    }
    files = get_files(samples[cat],years)

    default = 'central'
    if cat=="data":
        # just add them all up
        mths = {
            cat : None,
        }
        for file in files:
            mths[cat] = add_hist(mths[cat],get_hists(file)[default])
        assign_metadata(mths[cat])
        write(mths,cat)

    elif cat=="bkg":
        # add up but keep components
        mths = {
            cat : None
        }
        for b in samples["bkg"]:
            b = b.lower()
            mths[b+'_individual'] = []
            mths[b] = None
        for file in files:
            tmp = get_hists(file)[default]
            bkg = next((b for b in samples["bkg"] if b in file)).lower()
            mths[bkg+'_individual'].append(tmp) # Save individual histogram
            mths[bkg] = add_hists(mths[bkg],tmp) # Add up per background category
            mths[cat] = add_hists(mths[cat],tmp) # Add up all
        assign_metadata(mths[cat])
        write(mths,cat)

    elif cat=="sig":
        # just add years
        signals = defaultdict(list)
        for file in files:
            signals['_'.join(file.split('/')[1].split('_')[:-1])].append(file)
        for signal,sigfiles in signals.items():
            sighists = {year: get_hists(next((f for f in sigfiles if year in f))) for year in years}
            keys = list(sorted(set([key for y,h in sighists.items() for key in h])))
            mths = {}
            for key in keys:
                mths[key] = None
                # handle uncorrelated systematics (vary one year at a time)
                if '20' in key:
                    getter = lambda h: h.get(key,h[default])
                # correlated systematics must be present in all years
                else:
                    getter = lambda h: h[key]
                for year,sighist in sighists.items():
                    mths[key] = add_hists(mths[key],getter(sighist))
            assign_metadata(mths[default])
            write(mths,signal)

# __________________________________________
# Plotting

def reorderLegend(ax,order,title):
    handles, labels = ax.get_legend_handles_labels()
    lhzip = list(zip(labels,handles))
    mapping = [labels.index(o) for o in order]
    labels, handles = zip(*[lhzip[i] for i in mapping])
    ax.legend(handles, labels, title=title)

class Plot:
    def __init__(self, meta):
        self.selection = meta['selection']
        self.fig, (self.top, self.bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))
        self.top.set_yscale('log')
        common.put_on_cmslabel(self.top, year = meta['lumi'] if 'lumi' in meta else meta['year'])
        self.top.set_ylabel('Event count')
        self.bot.set_ylabel('Ratio to central')
        self.bot.set_xlabel(r'$m_{T}$ (GeV)')
        self.legend_title = None

    def plot_hist(self, hist, denominator=None, label=None):
        ratio = np.ones(hist.nbins) if denominator is None else hist.vals / denominator.vals
        l = self.top.step(hist.binning[:-1], hist.vals, where='post', label=label)[0]
        self.bot.step(hist.binning[:-1], ratio, where='post', label=label, color=l.get_color())

    def set_xlim(self, *args, **kwargs):
        self.bot.set_xlim(*args, **kwargs)
        self.top.set_xlim(*args, **kwargs)

    def save(self, outfile='tmp.png', pdf=True, legend_order=None):
        self.top.text(
            0.02, 0.02,
            'Cut-based' if self.selection=='cutbased' else 'BDT',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=self.top.transAxes,
            usetex=True,
            fontsize=25
            )
        if legend_order is None:
            self.top.legend(title=self.legend_title)
        else:
            reorderLegend(self.top,legend_order,title=self.legend_title)
        outdir = osp.dirname(osp.abspath(outfile))
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
        plt.savefig(outfile.replace('.png', '.pdf'), bbox_inches="tight")
        common.imgcat(outfile)

def get_sysyear(year):
    return year[:4]

def get_systs_uncorrelated():
    uncorrelated = [
        'jer',
        'jec',
        'pu',
        'stat',
    ]
    return uncorrelated

# years = None -> generic case (uncorrelated systs not expanded)
def get_systs(names=False,years=["2016","2017","2018"],smooth=False):
    uncorrelated = get_systs_uncorrelated()
    syst_names = {
        'scale': "Scales",
        'jer': "JER",
        'jec': "JEC",
        'jes': "JES",
        'isr': "ISR (parton shower)",
        'fsr': "FSR (parton shower)",
        'pu': "Pileup reweighting",
        'pdf': "PDF",
    }
    if smooth:
        syst_names.update({
            'stat': "MC statistical (fit)",
        })
        uncorrelated.pop('stat')
    else:
        syst_names.update({
            'stat': "MC statistical",
        })
    # expand uncorrelated systs
    if years is not None:
        if not isinstance(years,list): years = [years]
        # convert to sysyears
        years = [get_sysyear(year) for year in years]
        syst_names2 = {k:v for k,v in syst_names.items() if k not in uncorrelated}
        for unc in uncorrelated:
            for year in years:
                syst_names2[unc+year] = syst_names[unc]+f' ({year})'
        syst_names = syst_names2
    if names: return syst_names
    else: return list(syst_names.keys())

# should always be run before any rebinning
def make_stat_combined(mths,sysyear):
    if f'stat{sysyear}_up' not in mths.keys():
        stat_up = mths['central'].copy()
        stat_down = mths['central'].copy()
        i = 0
        while f'mcstat{i}_{sysyear}_up' in mths.keys():
            stat_up.vals[i] = mths[f'mcstat{i}_{sysyear}_up'].vals[i]
            stat_down.vals[i] = mths[f'mcstat{i}_{sysyear}_down'].vals[i]
            i += 1

        mths[f'stat{sysyear}_up'] = stat_up
        mths[f'stat{sysyear}_down'] = stat_down
    return mths

@scripter
def plot_systematics():
    change_bin_width()
    yrange = common.pull_arg('--yrange', type=float, nargs=2, default=None).yrange
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)
    meta = mths['central'].metadata
    years = meta['year']
    if not isinstance(years,list): years = [years]
    systs = get_systs(years=years,smooth="smooth" in json_file)
    for year in years:
        sysyear = get_sysyear(year)
        mths = make_stat_combined(mths,sysyear)

    mths = rebin_dict(mths)
    n = mths['central'].vals.sum()
    common.logger.info(f'central integral: {n}')
    common.logger.info(f'central metadata:\n{mths["central"].metadata}')

    model_str = osp.basename(json_file).replace(".json","")
    outdir = f'plots_{strftime("%Y%m%d")}_{model_str}'
    os.makedirs(outdir, exist_ok=True)

    for syst in systs:
        plot = Plot(meta)
        plot.plot_hist(central, label='Central')
        plot.plot_hist(mths[f'{syst}_up'], central, f'{syst} up')
        plot.plot_hist(mths[f'{syst}_down'], central, f'{syst} down')
        if yrange is not None:
            plot.bot.set_ylim(yrange[0],yrange[1])
        plot.save(f'{outdir}/{syst}.png')

@scripter
def plot_bkg():
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    rebin = common.pull_arg('--rebin', type=int, default=1).rebin
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)

    sig_json_file = common.pull_arg('sigjsonfile', type=str, nargs='*').sigjsonfile
    do_signal = len(sig_json_file) > 0

    h = mths['bkg'].rebin(rebin).cut(mtmin,mtmax)
    binning = h.binning
    nbins = h.nbins
    zero = np.zeros(nbins)

    with common.quick_ax() as ax:
        # for bkg in ['zjets', 'wjets', 'ttjets', 'qcd']:
        for bkg in ['qcd', 'ttjets', 'wjets', 'zjets']:
            ax.fill_between(h.binning[:-1], zero, h.vals, step='post', label=bkg)
            h.vals -= mths[bkg].rebin(rebin).cut(mtmin,mtmax).vals

        if do_signal:
            with open(sig_json_file[0]) as f:
                sig = json.load(f, cls=common.Decoder)['central']
                sig = sig.rebin(rebin).cut(mtmin,mtmax)
                ax.step(
                    sig.binning[:-1], sig.vals, '--k',
                    where='post', label=basename(sig.metadata)
                    )

        ax.set_yscale('log')
        ax.legend()
        ax.set_ylabel('Event count')
        ax.set_xlabel(r'$m_{T}$ (GeV)')
        common.put_on_cmslabel(ax)
        ax.text(
            0.02, 0.02,
            'Cut-based' if h.metadata['selection']=='cutbased' else 'BDT',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
            usetex=True,
            fontsize=25
            )

def get_xye(hist):
    x = hist.binning
    x = (x[:-1]+x[1:])/2 # bin centers
    y = hist.vals
    errs = hist.errs
    return x,y,errs

def do_loess(hist,span,do_gcv=False):
    from uloess import loess
    x, y, errs = get_xye(hist)
    # safety check for empty bins w/ empty errs
    for i,yy in enumerate(y):
        if yy==0:
            errs[i] = 1
    # 1sigma interval
    pred, conf_int, gcv = loess(x, y, errs, deg=2, alpha=0.683, span=span)
    if do_gcv:
        return gcv
    else:
        return pred, conf_int

@scripter
def smooth_shapes():
    span_val = common.pull_arg('--span', type=float, default=0.25, help="span value").span
    do_opt = common.pull_arg('--optimize', type=int, default=0, help="optimize span value using n values").optimize
    target = common.pull_arg('--target', type=str, default='central', help="optimize only based on target hist").target
    debug = common.pull_arg('--debug', default=False, action="store_true", help="debug optimization").debug
    var = common.pull_arg('--variation', type=str, default=None, help="MT variation to debug").variation
    save = common.pull_arg('--save', default=False, action="store_true", help="save debug variation output").save
    mtmin = common.pull_arg('--mtmin', type=float, default=None).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=None).mtmax
    norm = not common.pull_arg('--unnorm', default=False, action="store_true", help="fit unnormalized shape").unnorm
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)
    common.logger.info(f'central metadata:\n{mths["central"].metadata}')

    # loop over central and systematics
    year = mths["central"].metadata["year"]
    if not isinstance(year,str): year = str(int(year))
    variations = get_systs(years=year)
    variations = [v for v in variations if not v.startswith('stat')]
    variations = [var+'_up' for var in variations]+[var+'_down' for var in variations]
    variations = ['central']+variations

    # find optimization target
    if len(target)>0:
        if target not in variations:
            raise ValueError("Unknown target {} (known: {})".format(target, ', '.join(variations)))
        # put target first
        variations = [target]+[v for v in variations if v!=target]

    cut_args = {}
    if mtmin is not None: cut_args["xmin"] = mtmin
    if mtmax is not None: cut_args["xmax"] = mtmax
    mths_new = {}
    save_all = True
    if var is not None:
        target = var
        variations = [var]
        save_all = False
    for var in variations:
        hist = mths[var]
        hyield = hist.vals.sum()
        common.logger.info(f'{var} integral: {hyield}')

        meta = hist.metadata

        # normalize shape to unit area, then scale prediction by original yield
        if norm: hist = hist*(1./hyield)

        if do_opt>0 and (var==target or len(target)==0):
            span_min = 0.1 # if span is too small, no points are included
            spans = np.linspace(span_min,1.,do_opt,endpoint=False) # skip 1
            gcvs = np.array([do_loess(hist, span, do_gcv=True) for span in spans])
            span_val = spans[np.argmin(gcvs)]
            if debug: print('\n'.join(['{} {}'.format(span,gcv) for span,gcv in zip(spans,gcvs)]))
            print("Optimal span ({}): {}".format(var,span_val))

        pred, conf = do_loess(hist,span=span_val)

        hsmooth = hist.copy()
        hsmooth.vals = pred
        hsmooth.errs = conf[1] - pred

        # cuts applied *after* interpolation
        if norm:
            hsmooth = hsmooth*hyield
            hsmooth = hsmooth.cut(**cut_args)
        mths_new[var] = hsmooth
        if var=='central':
            inames = ['down','up']
            # computed after normalization to original yield (above)
            ivals = [hsmooth.vals-hsmooth.errs, hsmooth.vals+hsmooth.errs]
            for iname,ival in zip(inames,ivals):
                hstat = hsmooth.copy()
                # avoid errors going negative
                hstat.vals = np.clip(ival,0,None)
                hstat.errs = np.zeros_like(hstat.vals)
                mths_new[f'stat_{iname}'] = hstat

    outdir = os.path.dirname(json_file).replace("hists","smooth").replace("merged","smooth")
    os.makedirs(outdir, exist_ok=True)
    if save_all:
        # copy any other contents from original input
        # omitting mcstat uncertainties, which are replaced by overall confidence interval
        # and applying cuts
        for key in mths.keys():
            if key not in mths_new.keys() and 'mcstat' not in key:
                if isinstance(mths[key],list):
                    mths_new[key] = []
                    for entry in mths[key]:
                        mths_new[key].append(entry.cut(**cut_args))
                else:
                    mths_new[key] = mths[key].cut(**cut_args)
        outfile = outdir+'/'+osp.basename(json_file).replace(".json","_smooth.json")
    elif save:
        outfile = outdir+'/'+osp.basename(json_file).replace(".json","_smooth_{}.json".format(var))
    if save_all or save:
        with open(outfile, 'w') as f:
            json.dump(mths_new, f, indent=4, cls=common.Encoder)

@scripter
def plot_smooth():
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    var = common.pull_arg('--variation', type=str, default='central', help="MT variation to plot (or 'all')").variation
    noratio = common.pull_arg('--no-ratio', default=False, action="store_true", help="skip ratio").no_ratio
    names = common.pull_arg('--names', type=str, nargs='*', default=[], help="legend names for files").names
    json_files = common.pull_arg('jsonfiles', type=str, nargs='+').jsonfiles

    if len(names)==0:
        names = ["sample {}".format(i+1) for i in range(len(json_files))]
    elif len(names)>0 and len(names)!=len(json_files):
        raise ValueError("Mismatch between length of names ({}) and length of files ({})".format(len(names),len(files)))

    vars = [var]
    if var=='all':
        vars = get_systs()
        vars = [var+'_up' for var in vars]+[var+'_down' for var in vars]
        vars = ['central']+vars

    mths = []
    for json_file in json_files:
        with open(json_file) as f:
            mths.append(json.load(f, cls=common.Decoder))
            omit = [var for var in vars if var not in mths[-1].keys()]
            vars = [var for var in vars if var in mths[-1].keys()]
            if len(omit)>0: print("Omitting keys missing in {}: {}".format(json_file,', '.join(omit)))

    model_str = osp.basename(json_file).replace(".json","")
    outdir = f'plot_smooth_{strftime("%Y%m%d")}_{model_str}'
    os.makedirs(outdir, exist_ok=True)

    for var in vars:
        meta = mths[0]['central'].metadata
        plot = Plot(meta)
        legend_order = []
        h_denom = None

        for i,(mth,name) in enumerate(zip(mths,names)):
            hist = mth[var].cut(mtmin,mtmax)

            x, y, e = get_xye(hist)
            if 'smooth' in name:
                ys_up = y+e
                ys_dn = y-e
                e = None
            legend_order.append(name)
            line = plot.top.errorbar(x,y,yerr=e,label=legend_order[-1])
            line = line[0]

            if 'smooth' in name:
                plot.top.fill_between(x,ys_dn,ys_up,alpha=0.33,color=line.get_color())

            if not noratio:
                if h_denom is None:
                    h_denom = y
                    plot.bot.set_ylabel('Ratio to {}'.format(name))
                else:
                    plot.bot.plot(x,y/h_denom,color=line.get_color())

        plot.save(f'{outdir}/{var}.png',legend_order=legend_order)

def get_yield(hist):
    return hist.vals.sum()

def pct_diff(central,syst):
    return abs(1-syst/central)*100

def printSigFigs(num,fig,maxdec):
    # use scientific notation to get sig figs, then convert back to regular notation
    sci = "{0:>.{1}e}".format(num,fig-1)
    val = sci.split('e')[0]
    expo = int(sci.split('e')[1])
    pindex = val.find('.')
    if expo==0:
        result = val
    elif expo>0:
        if pindex==-1:
            result = val
        else:
            # move decimal point
            rval = val.replace('.','')
            pindex2 = pindex+expo
            # trailing zeroes
            if pindex2 >= len(rval): result = rval+'0'*(pindex2-len(rval))
            else: result = rval[0:pindex2]+'.'+rval[pindex2:]
    else: # expo<0
        rval = val if pindex==-1 else val.replace('.','')
        # add leading zeroes
        result = "0."+'0'*(abs(expo)-1)+rval
    # recursive truncation
    if '.' in result and len(result.split('.')[1])>maxdec:
        if fig==1:
            rnum = round(num,abs(expo)-1)
            if rnum==0.0: return "0.0"
            else: return printSigFigs(rnum,fig,maxdec)
        else: return printSigFigs(num,fig-1,maxdec)
    else:
        if set(result.replace('.',''))==set([0]): return "0.0"
        else: return result

@scripter
def systematics_table():
    change_bin_width()
    qtyrange = common.pull_arg('--qtyrange', metavar=("qty min max"), default=[], type=str, action='append', nargs=3).qtyrange
    minimum = common.pull_arg('--minimum', type=float, default=0.01, help="minimum value to display, smaller values rounded to 0").minimum
    skimdir = common.pull_arg('skimdir', type=str).skimdir
    skims = expand_wildcards(skimdir)

    # set up qty range limitations
    qtyfilters = []
    for iq,qr in enumerate(qtyrange):
        qtyfilters.append(lambda md: float(qtyrange[iq][1])<=md[qtyrange[iq][0]] and md[qtyrange[iq][0]]<=float(qtyrange[iq][2]))

    # needs to be kept in sync w/ boostedsvj/svj_limits/boosted_fits.py:gen_datacard()
    flat_systs = {
        'lumi': 1.6,
        'trigger_cr': 2.0,
        'trigger_sim': 2.1,
    }

    unc_systs = get_systs_uncorrelated()
    systs = get_systs(names=True,years=None)
    systs.update({
        'lumi': "Luminosity",
        'trigger_cr': "Trigger (CR)",
        'trigger_sim': "Trigger (MC)",
    })

    # indexing: [year][syst]
    syst_effects = defaultdict(lambda: defaultdict(lambda: (1e10, 0.0)))
    def update_effect(year,syst,effect):
        syst_effects[year][syst] = (min(effect, syst_effects[year][syst][0]), max(effect, syst_effects[year][syst][1]))
    for skim in skims:
        with open(skim) as f:
            mths = json.load(f, cls=common.Decoder)
        meta = mths['central'].metadata
        year = meta['year']
        if not isinstance(year,str): year = str(int(year))

        mths = make_stat_combined(mths,get_sysyear(year))
        mths = rebin_dict(mths)
        central = mths['central']
        central_yield = get_yield(central)
        #common.logger.info(f'central metadata:\n{meta}')

        passed = True
        for qf in qtyfilters:
            if not qf(meta):
                passed = False
                break
        if not passed:
            continue

        total = 0
        for syst in sorted(systs.keys()):
            syst_effect = 0
            asyst = syst
            # uncorrelated systs stored with sysyear naming (for datacard creation)
            if syst in unc_systs: asyst += get_sysyear(year)
            if f'{asyst}_up' in mths:
                syst_up_yield = get_yield(mths[f'{asyst}_up'])
                syst_dn_yield = get_yield(mths[f'{asyst}_down'])
                syst_effect = max(pct_diff(central_yield,syst_up_yield),pct_diff(central_yield,syst_dn_yield))
            elif syst in flat_systs:
                syst_effect = flat_systs[syst]
            else:
                #common.logger.warning(f'could not find systematic: {syst}')
                continue
            update_effect(year,syst,syst_effect)
            total += syst_effect**2
        total = np.sqrt(total)
        update_effect(year,"total",total)

    # keep correct order
    all_years = ["2016","2017","2018","2018PRE","2018POST"]
    years = [y for y in all_years if y in list(syst_effects.keys())]
    # add overall
    if len(years)>1:
        years.append("Overall")
        for syst in list(systs.keys())+["total"]:
            syst_effects["Overall"][syst] = (min([syst_effects[year][syst][0] for year in years]), max([syst_effects[year][syst][1] for year in years]))

    # print settings
    sigfig = 2
    maxdec = int(abs(np.log10(minimum)))

    print(" & ".join(["Systematic"]+years)+r" \\")
    print(r"\hline")
    def print_syst_row(syst):
        cols = [systs[syst]]
        for year in years:
            tmin = syst_effects[year][syst][0]
            tmax = syst_effects[year][syst][1]
            smin = printSigFigs(tmin,sigfig,maxdec)
            smax = printSigFigs(tmax,sigfig,maxdec)
            # don't bother to display a range if values are equal within precision
            if abs(tmax-tmin)>minimum and smin != smax:
                trange = smin+"--"+smax
            else:
                trange = smax
            cols.append(trange)
        print(" & ".join(cols)+r" \\")
    for syst in sorted(systs):
        print_syst_row(syst)
    print(r"\hline")
    systs["total"] = "total"
    print_syst_row("total")

@scripter
def acc():
    qty = common.pull_arg('--qty', type=str, required=True, help="signal qty").qty
    skimdir = common.pull_arg('skimdir', type=str).skimdir
    skims = expand_wildcards(skimdir)
    def get_from_cutflow(cutflow,index):
        return float(list(cutflow.items())[index][1])
    for skim in skims:
        with open(skim) as f:
            mths = json.load(f, cls=common.Decoder)
        meta = mths['central'].metadata
        cutflow = mths['central'].cutflow
        print(meta[qty],get_from_cutflow(cutflow,-1)/get_from_cutflow(cutflow,0)*100)

@scripter
def merge(args=None):
    if args is None:
        outfile = common.pull_arg('-o', '--outfile', type=str).outfile
        json_files = common.pull_arg('jsonfiles', type=str, nargs='+').jsonfiles
    else:
        outfile, json_files = args

    d = {}
    for json_file in json_files:
        common.logger.info(f'Merging {json_file}')
        with open(json_file) as f:
            mths = json.load(f, cls=common.Decoder)
            d.update(mths)
    
    if not outfile:
        for f in json_files:
            f = osp.abspath(f)
            if 'mz' not in f: continue
            outfile = osp.basename(osp.abspath(f)) + '_withbkg.json'
            break
        else:
            outfile = 'out.npz'

    common.logger.info(f'Dumping to {outfile}')
    with open(outfile, 'w') as f:
        json.dump(d, f, indent=4, cls=common.Encoder)


@scripter
def ls():
    infile = common.pull_arg('infile', type=str).infile
    if infile.endswith('.json'):
        with open(infile, 'r') as f:
            d = json.load(f, cls=common.Decoder)    
        from pprint import pprint
        pprint(d)
    elif infile.endswith('.root'):
        array = svj.open_root(infile, load_gen=True, load_jerjec=True)
        print(f'Found {len(array)} in {infile}')
        print(f'Metadata:\n{array.metadata}')
    else:
        print(f'Unknown file format: {infile}')


if __name__ == '__main__':
    scripter.run()
