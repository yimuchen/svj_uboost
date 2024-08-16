# based on files in https://github.com/boostedsvj/svj_jobs_new

import os, os.path as osp, sys, json, re, math, traceback
from time import strftime

#########################################
# DEVELOPER NOTICE:
# This file gets sent to Condor jobs
# Therefore it should only depend on
# pip-installable packages
# (i.e. not any other files in this repo)
#########################################

import numpy as np
import awkward as ak

import svj_ntuple_processing as svj

###########
# JES stuff
###########

"""
This requires numba to run:

pip install numba
"""

try:
    import numba

    @numba.njit
    def gen_reco_matching(
        eta_reco, phi_reco, eta_gen, phi_gen
        ):
        N = len(eta_reco)
        M = len(eta_gen)

        match_dr = np.zeros(N*M)
        match_idxs = np.zeros((N*M, 2), dtype=np.int32)
        gen_index = -1 * np.ones(N, dtype=np.int32)

        for i_reco in range(N):
            for i_gen in range(M):
                drsq = calc_drsq(
                    eta_reco[i_reco], phi_reco[i_reco],
                    eta_gen[i_gen], phi_gen[i_gen]
                    )
                match_dr[i_reco*M + i_gen] = drsq
                match_idxs[i_reco*M + i_gen] = (i_reco, i_gen)

        order = match_dr.argsort()
        match_idxs = match_idxs[order]

        used_i_reco = set()
        used_i_gen = set()
        for i_reco, i_gen in match_idxs:
            i_reco = int(i_reco)
            i_gen = int(i_gen)
            if i_reco in used_i_reco: continue
            if i_gen in used_i_gen: continue
            used_i_gen.add(i_gen)
            used_i_reco.add(i_reco)
            gen_index[i_reco] = i_gen
            if len(used_i_reco) == N: break
            if len(used_i_gen) == M: break

        return gen_index

    @numba.njit
    def gen_truth_matching(
        eta_gen, phi_gen,
        eta_dq1, phi_dq1, eta_dq2, phi_dq2,
        eta_z, phi_z,
        drsq_comp = 1.5**2
        ):
        N = len(eta_gen)
        match_types = np.zeros(N, dtype=np.int32)

        for i in range(N):
            # Must match the Z
            if not calc_drsq(eta_gen[i], phi_gen[i], eta_z, phi_z) < drsq_comp:
                continue
            # Add 1 for every dark quark that's also matched;
            # 2 means both match (full), 1 means only one matches (partial)
            match_types[i] = (
                (calc_drsq(eta_gen[i], phi_gen[i], eta_dq1, phi_dq1) < drsq_comp)
                +
                (calc_drsq(eta_gen[i], phi_gen[i], eta_dq2, phi_dq2) < drsq_comp)
                )

        return match_types

    @numba.njit
    def calc_drsq(eta1, phi1, eta2, phi2):
        """
        Calculates the delta R squared between two 4-vectors
        """
        # First calculate dphi, ensure -pi < dphi <= pi
        twopi = 6.283185307179586
        dphi = (phi1 - phi2) % twopi # Map to 0..2pi range
        if dphi > 3.141592653589793: dphi -= twopi # Map pi..2pi --> -pi..0
        return (eta1-eta2)**2 + dphi**2

    def get_row_splits(akarray):
        counts = ak.count(akarray, axis=-1).to_numpy()
        row_splits = np.concatenate((np.zeros(1, dtype=int), np.cumsum(counts).astype(int)))
        return row_splits

    @numba.njit
    def calc_x_jes_rowsplits(
        pt_reco, eta_reco, phi_reco, rowsplits_reco,
        pt_gen, eta_gen, phi_gen, rowsplits_gen,
        eta_dq_1, phi_dq_1,
        eta_dq_2, phi_dq_2,
        eta_z, phi_z,
        do_match_type = 3, # 1: partial only; 2: full only; 3: both
        drsq_comp = 1.5**2 # min dR required for a match
        ):
        n_events = len(rowsplits_reco) - 1
        x_jes = np.zeros(len(eta_reco))
        for i_event in range(n_events):
            if eta_z[i_event] == -100.: continue
            if eta_dq_1[i_event] == -100. and eta_dq_2[i_event] == -100.: continue

            left_reco = rowsplits_reco[i_event]
            right_reco = rowsplits_reco[i_event+1]
            n_reco = right_reco - left_reco
            left_gen = rowsplits_gen[i_event]
            right_gen = rowsplits_gen[i_event+1]

            gen_index = gen_reco_matching(
                eta_reco[left_reco:right_reco], phi_reco[left_reco:right_reco],
                eta_gen[left_gen:right_gen], phi_gen[left_gen:right_gen],
                )

            match_type_gen = gen_truth_matching(
                eta_gen[left_gen:right_gen], phi_gen[left_gen:right_gen],
                eta_dq_1[i_event], phi_dq_1[i_event],
                eta_dq_2[i_event], phi_dq_2[i_event],
                eta_z[i_event], phi_z[i_event],
                drsq_comp = drsq_comp
                )

            match_type_reco = match_type_gen[gen_index]
            matching_gen_pt = pt_gen[left_gen:right_gen][gen_index]

            if do_match_type==3:
                # Select both partial and full matches
                selected_matches = (gen_index != -1) & (match_type_reco.astype(np.bool8))
            else:
                # Select only the matches we're interested in
                selected_matches = (gen_index != -1) & (match_type_reco == do_match_type)

            # Calculate x_jes for the selected matches
            for i in range(n_reco):
                if selected_matches[i]:
                    x_jes[left_reco+i] = matching_gen_pt[i] / pt_reco[left_reco+i] - 1.

        return x_jes

    def calc_x_jes(
        pt_reco, eta_reco, phi_reco,
        pt_gen, eta_gen, phi_gen,
        genparticles_eta, genparticles_phi, genparticles_pdgid, genparticles_status,
        do_match_type = 3, # 1: partial only; 2: full only; 3: both
        drsq_comp = 1.5**2
        ):
        rowsplits_reco = get_row_splits(pt_reco)
        rowsplits_gen = get_row_splits(pt_gen)

        # Get Z prime; fill in -100 if there is no Z prime
        select_z = genparticles_pdgid == 4900023
        eta_z = ak.fill_none(ak.firsts(genparticles_eta[select_z]), -100.)
        phi_z = ak.fill_none(ak.firsts(genparticles_phi[select_z]), -100.)

        # Get dark quarks; fill in -100 for missing dark quarks
        select_dq = ((np.abs(genparticles_pdgid)==4900101) & (genparticles_status==71))
        eta_dq = genparticles_eta[select_dq]
        phi_dq = genparticles_phi[select_dq]
        eta_dq_1 = ak.fill_none(ak.firsts(eta_dq), -100.)
        phi_dq_1 = ak.fill_none(ak.firsts(phi_dq), -100.)
        eta_dq_2 = ak.fill_none(ak.firsts(eta_dq[:,1:]), -100.)
        phi_dq_2 = ak.fill_none(ak.firsts(phi_dq[:,1:]), -100.)

        x_jes = calc_x_jes_rowsplits(
            ak.flatten(pt_reco).to_numpy(),
            ak.flatten(eta_reco).to_numpy(),
            ak.flatten(phi_reco).to_numpy(),
            rowsplits_reco,
            ak.flatten(pt_gen).to_numpy(),
            ak.flatten(eta_gen).to_numpy(),
            ak.flatten(phi_gen).to_numpy(),
            rowsplits_gen,
            eta_dq_1.to_numpy(),
            phi_dq_1.to_numpy(),
            eta_dq_2.to_numpy(),
            phi_dq_2.to_numpy(),
            eta_z.to_numpy(),
            phi_z.to_numpy(),
            do_match_type,
            drsq_comp
            )

        # Restructure x_jes to reco jets arrays
        return ak.unflatten(ak.Array(x_jes), ak.num(pt_reco))

except ImportError:
    def calc_x_jes(*args, **kwargs):
        raise Exception('Cannot run match_jes without numba; pip install numba first')

def apply_jes(arrays, var, match_type='both'):
    if not var in ['up', 'down', 'central']:
        raise Exception('var should be up, down, or central')
    if var == 'central': return

    try:
        match_type = dict(both=3, partial=1, full=2)[match_type]
    except KeyError:
        svj.logger.error('Possible choices for match_type are both, partial, or full')

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

##################
# General skimming
##################

def dst(path,stageout,suffs=[]):
    """
    Generates a destination .npz for a rootfile
    """
    suff = '_'.join(suffs)
    # Get the stump starting from the dir with year in it
    pathsplit = path.split('/')[-3:]
    # put suff in [1] because [2] gets discarded during hadd
    if len(suff)>0:
        pathsplit[1] += '_'+suff
    # remove extraneous
    pathsplit[2] = pathsplit[2].replace('_RA2AnalysisTree','')
    path = '/'.join(pathsplit)
    path = path.replace('.root', '.npz')
    if stageout[-1]!='/': stageout += '/'
    return stageout + path

def skim(rootfile, group_data):
    """
    Produces a skim from TreeMaker Ntuples that is ready to be histogrammed
    """
    import seutils # type: ignore
    seutils.set_preferred_implementation(group_data.storage_implementation)

    suffs = []
    keep = None
    if hasattr(group_data,"keep"):
        keep = group_data.keep
    if keep is not None:
        suffs.append(f'keep{keep:.2f}')
    outfile = dst(rootfile,group_data.stageout,suffs)
    if seutils.isfile(outfile):
        svj.logger.info('    File %s exists, skipping', outfile)
        return

    array = svj.open_root(rootfile,local=group_data.local_copy)

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
        svj.logger.info(f'    Keeping only fraction {keep} of total number of events for signal MC')
        n_before = len(array)
        sel = np.random.choice(len(array), int(keep * len(array)), replace=False)
        svj.logger.info(f'    Downsampling from {n_before} -> {len(sel)}')
        array = apply_keep(array, sel, keep)

    # ______________________________
    # Work before applying preselection

    if array.metadata["sample_type"]=="sig":
        # PDF weights
        svj.logger.info('    Calculating PDFweight norm factors')
        pdf_weights = array.array['PDFweights'].to_numpy()
        # set massive unphysical weights to physical max
        pdf_max = np.max(pdf_weights, where=pdf_weights<100, initial=1)
        pdf_weights = np.clip(pdf_weights,a_min=None,a_max=pdf_max)
        pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
        # mu and sigma _per event_
        mu = np.mean(pdf_weights, axis=1)
        sigma = np.std(pdf_weights, axis=1)
        # Normalization factors for the weights
        pdfw_norm_up   = np.mean(mu+sigma)
        pdfw_norm_down = np.mean(mu-sigma)
        svj.logger.info(
            '    PDF unc:'
            f'\n        norm_up      = {pdfw_norm_up:.5f}'
            f'\n        norm_down    = {pdfw_norm_down:.5f}'
            )

        # Scale uncertainty
        # Compute normalizations before applying cuts
        scale_weight = array.array['ScaleWeights'].to_numpy()
        good_scales = np.array([0,1,2,3,4,6,8])
        scale_weight = scale_weight[ak.num(scale_weight,axis=1)>np.max(good_scales)]
        scale_weight = scale_weight[:,good_scales] # Throw away the mur/muf .5/2 and 2/.5 variations
        scale_norm_central = scale_weight[:,0].sum()
        scale_norm_up = np.max(scale_weight, axis=-1).sum()
        scale_norm_down = np.min(scale_weight, axis=-1).sum()
        scale_factor_up = scale_norm_central / scale_norm_up
        scale_factor_down = scale_norm_central / scale_norm_down
        svj.logger.info(
            '    Scale unc:'
            f'\n        norm_central = {scale_norm_central:.5f}'
            f'\n        norm_up      = {scale_norm_up:.5f}'
            f'\n        norm_down    = {scale_norm_down:.5f}'
            f'\n        factor_up    = {scale_factor_up:.5f}'
            f'\n        factor_down  = {scale_factor_down:.5f}'
            )

    # ______________________________
    # Apply preselection and save needed vars

    svj.logger.info('Running preselection now')
    if array.metadata["sample_type"]=="bkg":
        array = svj.filter_stitch(array)
    central = svj.filter_preselection(array)
    # Adjust the load_mc value as needed...
    cols = svj.bdt_feature_columns(central, load_mc=array.metadata["sample_type"]!="data")

    if array.metadata["sample_type"]=="sig":
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

    cols.metadata['selection'] = 'preselection'
    # Check again, to avoid race conditions
    if seutils.isfile(outfile):
        svj.logger.info('    File %s exists now, not staging out', outfile)
    else:
        cols.save(outfile,force=True)

    # systematic variations
    if array.metadata["sample_type"]=="sig":
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
            cols.metadata['selection'] = var_name
            outfile = dst(rootfile,group_data.stageout,suffs+[var_name])
            cols.save(outfile,force=True)

        # ______________________________
        # JES

        for var in ['up', 'down']:
            for match_type in ['both']: # other options are: 'full', 'partial'
                variation = apply_jes(array, var, match_type)
                variation = svj.filter_preselection(variation)
                cols = svj.bdt_feature_columns(variation, load_mc=True)
                cols.arrays['x_jes_1'] = variation.array['x_jes_15'][:,0].to_numpy()
                cols.arrays['x_jes_2'] = variation.array['x_jes_15'][:,1].to_numpy()
                cols.arrays['x_jes_3'] = ak.fill_none(ak.firsts(variation.array['x_jes_15'][:,2:]), -100.).to_numpy()
                cols.arrays['MET_precorr'] = variation.array['MET_precorr'].to_numpy()
                cols.arrays['METPhi_precorr'] = variation.array['METPhi_precorr'].to_numpy()
                outfile = dst(rootfile,group_data.stageout,suffs+[f'jes{var}_{match_type}'])
                cols.save(outfile,force=True)

# loop over inputs
if __name__=="__main__":
    try:
        from jdlfactory_server import data, group_data # type: ignore
        rootfiles = data.rootfiles
    except:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--stageout', type=str, help='stageout directory', required=True)
        parser.add_argument('-k', '--keep', type=float, default=None)
        parser.add_argument('--impl', dest='storage_implementation', type=str, help='storage implementation', default='xrd', choices=['xrd', 'gfal'])
        parser.add_argument('rootfiles', type=str, nargs='+')
        group_data = parser.parse_args()
        rootfiles = group_data.rootfiles
        try:
            import XRootD
            group_data.local_copy = False
        except:
            group_data.local_copy = True

    failed_rootfiles = []

    for i, rootfile in enumerate(rootfiles):
        svj.logger.info('Processing rootfile %s/%s: %s', i, len(rootfiles)-1, rootfile)
        try:
            skim(rootfile,group_data)
        except Exception:
            failed_rootfiles.append(rootfile)
            svj.logger.error('Error processing %s; continuing. Error:\n%s', rootfile, traceback.format_exc())

    if failed_rootfiles:
        svj.logger.info(
            'Failures were encountered for the following rootfiles:\n%s',
            '\n'.join(failed_rootfiles)
            )
