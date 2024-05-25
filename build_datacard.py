import os, os.path as osp, sys, json, re, math
from time import strftime

import tqdm
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

import svj_ntuple_processing as svj
import common

import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc


THIS_DIR = osp.dirname(osp.abspath(__file__))
# MAIN_DIR = osp.dirname(THIS_DIR)
sys.path.append(osp.join(THIS_DIR, 'systematics'))

scripter = common.Scripter()
DST = osp.join(THIS_DIR, 'skims')

# Relative path to the BDT
# This specific BDT was choosen to be used during the L3 review 
bdt_model_file = '/uscms/home/bregnery/nobackup/SVJ_mass_bdt_studies/svj_uboost/models/svjbdt_Feb28_lowmass_iterative_qcdtt_100p38.json'
# make sure bdt features match the choosen file
bdt_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef', 
    ]


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


def basename(meta):
    """
    Generates a name based on signal metadata.
    """
    return (
        f'mz{meta["mz"]:.0f}_rinv{meta["rinv"]:.1f}_mdark{meta["mdark"]:.0f}'
        )


@scripter
def skim():
    """
    Produces a skim from TreeMaker Ntuples that is ready to be histogrammed
    """
    pbar = tqdm.tqdm(total=8)
    svj.BRANCHES_GENONLY.extend([
        'PDFweights', 'PSweights',
        'puSysUp', 'puSysDown',
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
    outdir = common.pull_arg('-o', '--outdir', type=str, default=strftime('skims_%Y%m%d')).outdir
    selection = common.pull_arg('selection', type=str).selection
    full_selection = selection
    common.logger.info(f'Selection: {selection}')
    keep = common.pull_arg('-k', '--keep', type=float, default=None).keep
    rootfile = common.pull_arg('rootfile', type=str).rootfile
    array = svj.open_root(rootfile, load_gen=True, load_jerjec=True)

    def apply_keep(array, sel, keep):
        if sel is None:
            return array
        array.array = array.array[sel]
        scale = 1./keep
        array.array["Weight"] = array.array["Weight"]*scale
        array.cutflow['raw'] = len(array)
        return array

    sel = None
    if keep is not None:
        common.logger.info(f'Keeping only fraction {keep} of total number of events for signal MC')
        n_before = len(array)
        sel = np.random.choice(len(array), int(keep * len(array)), replace=False)
        common.logger.info(f'Downsampling from {n_before} -> {len(sel)}')
        array = apply_keep(array, sel, keep)
        full_selection += f'_keep{keep:.2f}'

    common.logger.info(f'Will save skims in outdir {outdir}')
    common.logger.info(f'Found {len(array)} events in {rootfile}')
    common.logger.info(f'Metadata for {rootfile}: {array.metadata}')
    pbar.update()

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

    # Scale uncertainty
    # Compute normalizations before applying cuts
    scale_weight = array.array['ScaleWeights'].to_numpy()
    scale_weight = scale_weight[:,np.array([0,1,2,3,4,6,8])] # Throw away the mur/muf .5/2 and 2/.5 variations
    scale_norm_central = scale_weight[:,0].sum()
    scale_norm_up = np.max(scale_weight, axis=-1).sum()
    scale_norm_down = np.min(scale_weight, axis=-1).sum()
    scale_factor_up = scale_norm_central / scale_norm_up
    scale_factor_down = scale_norm_central / scale_norm_down
    svj.logger.info(
        'Scale unc:'
        f'\n    norm_central = {scale_norm_central:.5f}'
        f'\n    norm_up      = {scale_norm_up:.5f}'
        f'\n    norm_down    = {scale_norm_down:.5f}'
        f'\n    factor_up    = {scale_factor_up:.5f}'
        f'\n    factor_down  = {scale_factor_down:.5f}'
        )

    # ______________________________
    # Apply preselection and save needed vars

    common.logger.info('Running preselection now')
    central = svj.filter_preselection(array)
    # Adjust the load_mc value as needed... don't understand if the skims are alway on mc for example
    cols = svj.bdt_feature_columns(central, load_mc=True)
    bdt_cols = svj.bdt_feature_columns(central, load_mc=True)

    # Save scale weights
    cols.arrays['scaleweights'] = central.array['ScaleWeights'].to_numpy()
    cols.metadata['scale_norm_central'] = scale_norm_central
    cols.metadata['scale_norm_up'] = scale_norm_up
    cols.metadata['scale_norm_down'] = scale_norm_down
    cols.metadata['scale_factor_up'] = scale_factor_up
    cols.metadata['scale_factor_down'] = scale_factor_down

    # Save PDF normalization and weights
    cols.metadata['pdfw_norm_up'] = pdfw_norm_up
    cols.metadata['pdfw_norm_down'] = pdfw_norm_down
    cols.arrays['pdf_weights'] = central.array['PDFweights'].to_numpy()

    # Save PS weights
    ps_weights = central.array['PSweights'].to_numpy()
    cols.arrays['ps_isr_up'] = ps_weights[:,6]
    cols.arrays['ps_isr_down'] = ps_weights[:,8]
    cols.arrays['ps_fsr_up'] = ps_weights[:,7]
    cols.arrays['ps_fsr_down'] = ps_weights[:,9]

    # Save PU weights
    cols.arrays['pu_central'] = central.array['puWeight'].to_numpy()
    cols.arrays['pu_sys_up'] = central.array['puSysUp'].to_numpy()
    cols.arrays['pu_sys_down'] = central.array['puSysDown'].to_numpy()

    def apply_selection(cols):
        # Apply further selection now
        if selection == 'cutbased':
            common.logger.info('Applying cutbased selection')
            cols = cols.select(common.mask_cutbased(cols))
            cols.cutflow['cutbased'] = len(cols)
        # Apply the BDT
        elif selection.startswith('bdt='):
            common.logger.info('Applying bdt selection')

            # Split the selection string by '=' to extract the number following 'bdt='
            parts = selection.split('=')
            
            # Check if the second part of the split is a valid number
            if len(parts) == 2:
                try:
                    bdt_cut = float(parts[1])
                except ValueError:
                    # Handle the case where the number following 'bdt=' is not valid
                    print("Invalid number following 'bdt='.")
            else:
                # Handle the case where the number following 'bdt=' is not valid
                raise ValueError("Invalid number {} following 'bdt='.".format(parts[1]))

            # Grab the input features and weights
            X = []
            weight = []
  
            # Get the features for the bkg samples
            X = cols.to_numpy(bdt_features)
            # Load the model and get the predictions
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(bdt_model_file)
            with common.time_and_log(f'Calculating xgboost scores for {bdt_model_file}...'):
                score = xgb_model.predict_proba(X)[:,1]
            weight = cols.arrays['puweight']*cols.arrays['weight']
            print('weight length: ', len(weight), ' weight: ', weight)

            # Obtain the efficiencies for the desired BDT working point
            # bdt_cut is the user input bdt_cut
            bdt_Hist=np.histogram(score[score>bdt_cut],weights=weight[score>bdt_cut]*len(score)) 
            bdt_Hist_nom=np.histogram(score[score>0.0],weights=weight[score>0.0]*len(score))
            eff = sum(bdt_Hist[0])/sum(bdt_Hist_nom[0]) 

            # Apply the DDT
            mT = cols.to_numpy(['mt']).ravel() # make one d ... don't ask why it's not
            pT = cols.to_numpy(['pt']).ravel()
            rho = cols.to_numpy(['rho']).ravel()
            bdt_ddt_score = common.ddt(mT, pT, rho, score, weight, eff*100)

            # Now cut on the DDT above 0.0 (referring to above the given BDT cut value)
            cols = cols.select(bdt_ddt_score > 0.0) # mask for the selection
            cols.cutflow['ddt(bdt)'] = len(cols)

        else:
            raise common.InvalidSelectionException()
        return cols

    cols = apply_selection(cols)
    cols.metadata['selection'] = selection
    cols.metadata['basename'] = basename(array.metadata)
    cols.save(f'{outdir}/{basename(array.metadata)}_{full_selection}_central.npz')
    pbar.update()

    # ______________________________
    # JEC/JER

    for var_name, appl in [
        ('jer_up',   svj.apply_jer_up),
        ('jer_down', svj.apply_jer_down),
        ('jec_up',   svj.apply_jec_up),
        ('jec_down', svj.apply_jec_down),
        ]:
        variation = appl(array)
        variation = svj.filter_preselection(variation)
        cols = svj.bdt_feature_columns(variation, load_mc=True)
        cols = apply_selection(cols)
        cols.save(f'{outdir}/{basename(array.metadata)}_{full_selection}_{var_name}.npz')
        pbar.update()

    # ______________________________
    # JES

    from jes import apply_jes

    for var in ['up', 'down']:
        for match_type in ['both']: # other options are: 'full', 'partial'
            common.logger.info(f'{var=}, {match_type=}')
            variation = apply_jes(array, var, match_type)
            common.logger.info(f'Done, applying presel')
            variation = svj.filter_preselection(variation)
            common.logger.info(f'Done, to columns')
            cols = svj.bdt_feature_columns(variation, load_mc=True)
            cols.arrays['x_jes_1'] = variation.array['x_jes_15'][:,0].to_numpy()
            cols.arrays['x_jes_2'] = variation.array['x_jes_15'][:,1].to_numpy()
            cols.arrays['x_jes_3'] = ak.fill_none(ak.firsts(variation.array['x_jes_15'][:,2:]), -100.).to_numpy()
            cols.arrays['MET_precorr'] = variation.array['MET_precorr'].to_numpy()
            cols.arrays['METPhi_precorr'] = variation.array['METPhi_precorr'].to_numpy()
            cols = apply_selection(cols)
            common.logger.info(f'Saving')
            cols.save(f'{outdir}/{basename(array.metadata)}_{full_selection}_jes{var}_{match_type}.npz')
            pbar.update()

    pbar.close()


@scripter
def build_sig_histograms(args=None):
    if args is None:
        change_bin_width()
        # Read from sys.argv
        selection = common.pull_arg('selection', type=str).selection
        lumi = common.pull_arg('--lumi', type=float, default=137.2, help='Luminosity (in fb-1)').lumi
        lumi *= 1e3 # Convert to nb-1, same unit as xs
        common.logger.info(f'Selection: {selection}')
        skim_files = common.pull_arg('skimfiles', type=str, nargs='+').skimfiles
    else:
        # Use passed input
        selection, lumi, skim_files = args

    def get_by_tag(tag):
        return [s for s in skim_files if tag in s][0]

    mths = {}
    central = svj.Columns.load(get_by_tag('central'))

    mt = central.to_numpy(['mt']).ravel()
    w = central.to_numpy(['puweight']).ravel()
    w *= lumi * central.xs / central.cutflow['raw']
    common.logger.info(f'Event weight: {lumi}*{central.xs}/{central.cutflow["raw"]} = {lumi*central.xs/central.cutflow["raw"]}')

    # Scale
    scale_weight = central.to_numpy(['scaleweights'])[:, np.array([0,1,2,3,4,6,8])]
    weight_up = w * np.max(scale_weight, axis=-1) * central.metadata['scale_factor_up']
    weight_down = w * np.min(scale_weight, axis=-1) * central.metadata['scale_factor_down']
    mths['scale_up'] = common.MTHistogram(mt, weight_up)
    mths['scale_down'] = common.MTHistogram(mt, weight_down)

    # JEC/JER/JES
    def mth_jerjecjes(tag):
        col = svj.Columns.load(get_by_tag(tag))
        mt = col.to_numpy(['mt']).flatten()
        w = col.to_numpy(['puweight']).flatten()
        w *= lumi * col.xs / col.cutflow['raw']
        return common.MTHistogram(mt, w)
    mths['jer_up'] = mth_jerjecjes('jer_up')
    mths['jer_down'] = mth_jerjecjes('jer_down')
    mths['jec_up'] = mth_jerjecjes('jec_up')
    mths['jec_down'] = mth_jerjecjes('jec_down')
    mths['jes_up'] = mth_jerjecjes('jesup_both')
    mths['jes_down'] = mth_jerjecjes('jesdown_both')

    # PS
    ps_weights = w[:,None] * central.to_numpy(['ps_isr_up', 'ps_isr_down',
                                       'ps_fsr_up', 'ps_fsr_down'])
    mths['isr_up']   = common.MTHistogram(mt, ps_weights[:,0])
    mths['isr_down'] = common.MTHistogram(mt, ps_weights[:,1])
    mths['fsr_up']   = common.MTHistogram(mt, ps_weights[:,2])
    mths['fsr_down'] = common.MTHistogram(mt, ps_weights[:,3])

    # PU
    pu_weights = central.to_numpy(['puweight', 'pu_sys_up', 'pu_sys_down'])
    mths['pu_up'] = common.MTHistogram(mt, w / pu_weights[:,0] * pu_weights[:,1])
    mths['pu_down'] = common.MTHistogram(mt, w / pu_weights[:,0] * pu_weights[:,2])

    # PDF
    pdf_weights = central.to_numpy(['pdf_weights'])
    pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
    mu_pdf = np.mean(pdf_weights, axis=1)
    sigma_pdf = np.std(pdf_weights, axis=1)
    pdfw_up = (mu_pdf+sigma_pdf) / central.metadata['pdfw_norm_up']
    pdfw_down = (mu_pdf-sigma_pdf) / central.metadata['pdfw_norm_down']
    mths['pdf_up'] = common.MTHistogram(mt, w*pdfw_up)
    mths['pdf_down'] = common.MTHistogram(mt, w*pdfw_down)

    # MC stats
    mth_central = common.MTHistogram(mt, w)
    mth_central.metadata.update(central.metadata)
    mths['central'] = mth_central
    mc_stat_err = np.sqrt(np.histogram(mt, bins=mth_central.binning, weights=w**2)[0])

    for i in range(mth_central.nbins):
        mth = mth_central.copy()
        mth.vals[i] += mc_stat_err[i]
        mths[f'mcstat{i}_up'] = mth
        mth = mth_central.copy()
        mth.vals[i] -= mc_stat_err[i]
        mths[f'mcstat{i}_down'] = mth

    meta = central.metadata
    outfile = (
        f'mz{meta["mz"]:.0f}_rinv{meta["rinv"]:.1f}_mdark{meta["mdark"]:.0f}'
        f'_{selection}.json'
        )
    common.logger.info(f'Dumping histograms to {outfile}')
    with open(outfile, 'w') as f:
        json.dump(mths, f, cls=common.Encoder, indent=4)
    return outfile


@scripter
def build_bkg_histograms(args=None):
    if args is None:
        change_bin_width()
        # Read from sys.argv
        selection = common.pull_arg('selection', type=str).selection
        lumi = common.pull_arg('--lumi', type=float, default=137.2, help='Luminosity (in fb-1)').lumi
        lumi *= 1e3 # Convert to nb-1, same unit as xs
        common.logger.info(f'Selection: {selection}')
        skim_files = common.pull_arg('skimfiles', type=str, nargs='+').skimfiles
    else:
        # Use passed input
        selection, lumi, skim_files = args

    mths = {
        'qcd_individual' : [],
        'ttjets_individual' : [],
        'wjets_individual' : [],
        'zjets_individual' : [],
        'qcd' : common.MTHistogram.empty(),
        'ttjets' : common.MTHistogram.empty(),
        'wjets' : common.MTHistogram.empty(),
        'zjets' : common.MTHistogram.empty(),
        'bkg' : common.MTHistogram.empty(),
        }
    mths['bkg'].metadata['selection'] = selection
    mths['bkg'].metadata['lumi'] = lumi

    for skim_file in tqdm.tqdm(skim_files):
        process = osp.basename(skim_file)

        # Filter out a few things
        if 'QCD_Pt' in process:
            # Low pt QCD bins have very few events, which get absurd weights
            left_pt_bound = int(re.match(r'QCD_Pt_(\d+)', process).group(1))
            if left_pt_bound < 300.: continue
        elif 'WJetsToLNu_HT' in process:
            # Low HT WJets events have very few events, which get absurd weights
            left_ht_bound = int(re.match(r'WJetsToLNu_HT-(\d+)', process).group(1))
            if left_ht_bound < 400.: continue
        elif 'WJetsToLNu_TuneCP5' in process:
            # Inclusive WJets bin after the stitch filter is basically HT (0,70)
            # Also too few events, too crazy weights
            continue

        col = svj.Columns.load(skim_file)

        # Apply further selection: cutbased or bdt
        if len(col) > 0:
            if selection == 'cutbased':
                col = col.select(common.mask_cutbased(col))

            # Apply the BDT
            elif selection.startswith('bdt='):
                common.logger.info('Applying bdt selection')
     
                # Split the selection string by '=' to extract the number following 'bdt='
                parts = selection.split('=')
                
                # Check if the second part of the split is a valid number
                if len(parts) == 2:
                    try:
                        bdt_cut = float(parts[1])
                    except ValueError:
                        # Handle the case where the number following 'bdt=' is not valid
                        print("Invalid number following 'bdt='.")
                else:
                    # Handle the case where the number following 'bdt=' is not valid
                    raise ValueError("Invalid number {} following 'bdt='.".format(parts[1]))
     
                # Grab the input features and weights
                X = []
                weight = []
     
                # Get the features for the bkg samples
                X = col.to_numpy(bdt_features)
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(bdt_model_file)
                with common.time_and_log(f'Calculating xgboost scores for {bdt_model_file}...'):
                    score = xgb_model.predict_proba(X)[:,1]
                weight = col.arrays['puweight']*col.arrays['weight']
                print('weight length: ', len(weight), ' weight: ', weight)
     
                # Obtain the efficiencies for the desired BDT working point
                # bdt_cut is the user input bdt_cut
                bdt_Hist=np.histogram(score[score>bdt_cut],weights=weight[score>bdt_cut]*len(score)) 
                bdt_Hist_nom=np.histogram(score[score>0.0],weights=weight[score>0.0]*len(score))
                eff = sum(bdt_Hist[0])/sum(bdt_Hist_nom[0]) 
     
                # Apply the DDT
                mT = col.to_numpy(['mt']).ravel() # make one d ... don't ask why it's not
                pT = col.to_numpy(['pt']).ravel()
                rho = col.to_numpy(['rho']).ravel()
                bdt_ddt_score = common.ddt(mT, pT, rho, score, weight, eff*100)
     
                # Now cut on the DDT above 0.0 (referring to above the given BDT cut value)
                col = col.select(bdt_ddt_score > 0.0) # mask for the selection
            else:
                raise Exception(f'selection must be cutbased or bdt=X.XXX, found {selection}')
        
        if len(col) == 0:
            # Skip this background if it had 0 events passing the preselection
            common.logger.info(f'Skipping {skim_file} because no events passed the preselection')
            continue
    
        array = col.to_numpy(['mt', 'weight'])
        mth = common.MTHistogram(array[:,0], lumi*array[:,1])
        mth.metadata['process'] = process

        bkg = [b for b in ['QCD', 'TTJets', 'ZJets', 'WJets'] if b in process][0].lower()
        mths[bkg+'_individual'].append(mth) # Save individual histogram
        mths[bkg] += mth # Add up per background category (qcd/ttjet/...)
        mths['bkg'] += mth # Add up all

    outfile = f'bkghist_{strftime("%Y%m%d")}.json'
    common.logger.info(f'Dumping histograms to {outfile}')
    with open(outfile, 'w') as f:
        json.dump(mths, f, cls=common.Encoder, indent=4)
    return outfile


@scripter
def build_histograms():
    """
    Runs both build_sig_histograms and build_bkg_histograms.
    """
    change_bin_width()
    selection = common.pull_arg('selection', type=str).selection
    lumi = common.pull_arg('--lumi', type=float, default=137.2, help='Luminosity (in fb-1)').lumi
    lumi *= 1e3 # Convert to nb-1, same unit as xs
    common.logger.info(f'Selection: {selection}')
    skim_files = common.pull_arg('skimfiles', type=str, nargs='+').skimfiles

    # Divide passed skim_files into signal or background
    sig_outfile = None
    bkg_outfile = None
    sig_skim_files = []
    bkg_skim_files = []
    for skim_file in skim_files:
        if skim_file.endswith('.json'):
            if "bkghist" in skim_file:
                bkg_outfile = skim_file
            else:
                sig_outfile = skim_file
        else:
            for bkg_type in ['QCD', 'TTJets', 'WJets', 'ZJets']:
                if bkg_type in skim_file:
                    bkg_skim_files.append(skim_file)
                    break
            else:
                sig_skim_files.append(skim_file)

    if sig_outfile is None:
        common.logger.info(
            'Using the following skim files for signal:\n'
            + "\n".join(sig_skim_files)
        )
    else:
        common.logger.info('Reusing {} for signal'.format(sig_outfile))
    if bkg_outfile is None:
        common.logger.info(
            'Using the following skim files for background:\n'
            + "\n".join(bkg_skim_files)
        )
    else:
        common.logger.info('Reusing {} for background'.format(bkg_outfile))

    if sig_outfile is None:
        sig_outfile = build_sig_histograms((selection, lumi, sig_skim_files))
    if bkg_outfile is None:
        bkg_outfile = build_bkg_histograms((selection, lumi, bkg_skim_files))
    merged_outfile = sig_outfile.replace('.json', '_with_bkg.json')

    if common.MTHistogram.non_standard_binning:
        binw = int(common.MTHistogram.bins[1] - common.MTHistogram.bins[0])
        left = common.MTHistogram.bins[0]
        right = common.MTHistogram.bins[-1]
        merged_outfile = merged_outfile.replace(
            '.json',
            f'_binw{binw:02d}_range{left:.0f}-{right:.0f}.json'
            )
    merge((merged_outfile, [sig_outfile, bkg_outfile]))


# __________________________________________
# Plotting

def reorderLegend(ax,order,title):
    handles, labels = ax.get_legend_handles_labels()
    lhzip = list(zip(labels,handles))
    mapping = [labels.index(o) for o in order]
    labels, handles = zip(*[lhzip[i] for i in mapping])
    ax.legend(handles, labels, title=title)

class Plot:
    def __init__(self, selection):
        self.selection = selection
        self.fig, (self.top, self.bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))
        self.top.set_yscale('log')
        common.put_on_cmslabel(self.top)
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

def get_systs():
    return ['scale', 'jer', 'jec', 'jes', 'isr', 'fsr', 'pu', 'pdf', 'stat']

@scripter
def plot_systematics():
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    rebin = common.pull_arg('--rebin', type=int, default=1).rebin
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)

    n = mths['central'].vals.sum()
    common.logger.info(f'central integral: {n}')
    common.logger.info(f'central metadata:\n{mths["central"].metadata}')

    central = mths['central'].rebin(rebin).cut(mtmin,mtmax)
    meta = central.metadata

    model_str = osp.basename(json_file).replace(".json","")
    outdir = f'plots_{strftime("%Y%m%d")}_{model_str}'
    os.makedirs(outdir, exist_ok=True)

    systs = get_systs()
    if 'stat_up' not in mths.keys():
        stat_up = mths['central'].copy()
        stat_down = mths['central'].copy()
        i = 0
        while f'mcstat{i}_up' in mths.keys():
            stat_up.vals[i] = mths[f'mcstat{i}_up'].vals[i]
            stat_down.vals[i] = mths[f'mcstat{i}_down'].vals[i]
            i += 1

        mths['stat_up'] = stat_up
        mths['stat_down'] = stat_down

    for syst in systs:
        plot = Plot(meta['selection'])
        plot.plot_hist(central, label='Central')
        plot.plot_hist(mths[f'{syst}_up'].rebin(rebin).cut(mtmin,mtmax), central, f'{syst} up')
        plot.plot_hist(mths[f'{syst}_down'].rebin(rebin).cut(mtmin,mtmax), central, f'{syst} down')
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
                    where='post', label=sig.metadata['basename']
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
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)
    common.logger.info(f'central metadata:\n{mths["central"].metadata}')

    # loop over central and systematics
    variations = get_systs()
    variations.remove('stat')
    variations = [var+'_up' for var in variations]+[var+'_down' for var in variations]
    variations = ['central']+variations

    # find optimization target
    if len(target)>0:
        if target not in variations:
            raise ValueError("Unknown target {} (known: {})".format(target, ', '.join(variations)))
        # put target first
        variations = [target]+[v for v in variations if v!=target]

    mths_new = {}
    save = True
    if var is not None:
        target = var
        variations = [var]
        save = False
    for var in variations:
        hyield = mths[var].vals.sum()
        common.logger.info(f'{var} integral: {hyield}')

        hist = mths[var]
        meta = hist.metadata

        # normalize shape to unit area, then scale prediction by original yield
        hist = hist*(1./hyield)

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
        hsmooth = hsmooth*hyield

        mths_new[var] = hsmooth
        if var=='central':
            inames = ['down','up']
            for ind in [0,1]:
                hstat = hist.copy()
                # avoid errors going negative
                hstat.vals = np.clip(conf[ind],0,None)*hyield
                hstat.errs = np.zeros_like(hstat.vals)
                mths_new['stat_{}'.format(inames[ind])] = hstat

    if save:
        # copy any other contents from original input
        # omitting mcstat uncertainties, which are replaced by overall confidence interval
        for key in mths.keys():
            if key not in mths_new.keys() and 'mcstat' not in key:
                mths_new[key] = mths[key]
        outfile = osp.basename(json_file).replace(".json","_smooth.json")
        with open(outfile, 'w') as f:
            json.dump(mths_new, f, indent=4, cls=common.Encoder)

@scripter
def plot_smooth():
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    var = common.pull_arg('--variation', type=str, default='central', help="MT variation to plot (or 'all')").variation
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
        plot = Plot("")
        meta = mths[0]['central'].metadata
        plot.selection = meta['selection']
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

            if h_denom is None:
                h_denom = y
                plot.bot.set_ylabel('Ratio to {}'.format(name))
            else:
                plot.bot.plot(x,y/h_denom,color=line.get_color())

        plot.save(f'{outdir}/{var}.png',legend_order=legend_order)

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
