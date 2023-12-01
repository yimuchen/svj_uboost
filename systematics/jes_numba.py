"""
This module contains numba JIT compiled code to calculate the JES factors.
It requires numba to run:

pip install numba
"""

import numpy as np
import awkward as ak


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
                selected_matches = match_type_reco.astype(np.bool8)
            else:
                # Select only the matches we're interested in
                selected_matches = match_type_reco == do_match_type

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


def test_calc_x_jes():
    # fmt: off
    pt_reco  = ak.Array([[2.05, 3.3,  5.5],            [2.1, 3.3],       [5.5, 3.3] ])
    eta_reco = ak.Array([[.61,  1.05, .11],            [.31, 1.05],      [.11, 1.05] ])
    phi_reco = ak.Array([[.64,  0.99, .09],            [.34, 0.99],      [.09, 0.99] ])
    pt_gen   = ak.Array([[2.1,  3.4,  5.5, 6.6, 7.7],  [3.3,  5.5, 2.2], [8.0] ])
    eta_gen  = ak.Array([[.31,  1.08, .11, .61, .71],  [1.05, .11, .31], [.11] ])
    phi_gen  = ak.Array([[.34,  1.07, .09, .69, .79],  [0.99, .09, .34], [.09], ])
    genparticles_eta = ak.Array([[1.13, 1.13, 1.00], [.31, .30, 1.01], []])
    genparticles_phi = ak.Array([[1.12, 1.14, 1.01], [.34, .32, 1.02], []])
    genparticles_pdgid = ak.Array([[4900023, 4900101, -4900101], [4900023, 4900101, -4900101], []])
    genparticles_status = ak.Array([[71, 71, 71], [71, 71, 71], []])

    x_jes = calc_x_jes(
        pt_reco, eta_reco, phi_reco,
        pt_gen, eta_gen, phi_gen,
        genparticles_eta, genparticles_phi,
        genparticles_pdgid, genparticles_status,
        drsq_comp=.15**2
        )
    exp = ak.Array([
        [0., 3.4/3.3-1., 0.], # only jet 1 corrected
        [2.2/2.1-1., 0.],     # only jet 0 corrected
        [0., 0.]              # no jets corrected
        ])
    print(f'x_jes=\n{x_jes}\nexp=\n{exp}')
    assert ak.all(np.abs(x_jes - exp) < 0.001)

    x_jes = calc_x_jes(
        pt_reco, eta_reco, phi_reco,
        pt_gen, eta_gen, phi_gen,
        genparticles_eta, genparticles_phi,
        genparticles_pdgid, genparticles_status,
        do_match_type=1, # Only correct jets with 1 quark matching
        drsq_comp=.15**2
        )
    exp = ak.Array([
        [0., 0., 0.],         # no jets corrected
        [2.2/2.1-1., 0.],     # only jet 0 corrected
        [0., 0.]              # no jets corrected
        ])
    print(f'x_jes=\n{x_jes}\nexp=\n{exp}')
    assert ak.all(np.abs(x_jes - exp) < 0.001)

    x_jes = calc_x_jes(
        pt_reco, eta_reco, phi_reco,
        pt_gen, eta_gen, phi_gen,
        genparticles_eta, genparticles_phi,
        genparticles_pdgid, genparticles_status,
        do_match_type=2, # Only correct jets with 2 quarks matching
        drsq_comp=.15**2
        )
    exp = ak.Array([
        [0., 3.4/3.3-1., 0.], # only jet 1 corrected
        [0., 0.],             # no jets corrected
        [0., 0.]              # no jets corrected
        ])
    print(f'x_jes=\n{x_jes}\nexp=\n{exp}')
    assert ak.all(np.abs(x_jes - exp) < 0.001)
    # fmt: on


def test_calc_dr():
    dr = calc_drsq(.1, .1, .2, .2)
    expected = .1**2 + .1**2
    np.testing.assert_almost_equal(dr, expected)

    dr = calc_drsq(.1, np.pi-.1, .2, 7*np.pi+.1)
    expected = .1**2 + .2**2
    np.testing.assert_almost_equal(dr, expected)

    dr = calc_drsq(.1, 10.*np.pi, .2, 12*np.pi+.1)
    expected = .1**2 + .1**2
    np.testing.assert_almost_equal(dr, expected)


def test_gen_reco_matching():
    eta_reco = np.array([.49,  .08, .29])
    phi_reco = np.array([.51,  .07, .32])
    eta_gen  = np.array([.31, 1.05, .11, .51, .74])
    phi_gen  = np.array([.34, 0.99, .09, .50, .69])
    gen_index = gen_reco_matching(eta_reco, phi_reco, eta_gen, phi_gen)
    exp = np.array([3, 2, 0])
    print(f'{gen_index=}, {exp=}')
    np.testing.assert_array_equal(gen_index, exp)


def test_gen_truth_matching():
    eta_gen  = np.array([.31, 1.05, .11, .22, .74])
    phi_gen  = np.array([.34, 0.99, .09, .22, .69])
    match_type = gen_truth_matching(
        eta_gen, phi_gen,
        eta_dq1=.30, phi_dq1=.30,
        eta_dq2=.40, phi_dq2=.40,
        eta_z=.30,   phi_z=.30,
        drsq_comp=0.15**2
        )
    exp = np.array([2, 0, 0, 1, 0])
    print(f'{match_type=}, {exp=}')
    np.testing.assert_array_equal(match_type, exp)


def test_get_row_splits():
    pt = ak.Array([[2.1, 3.3,  5.5, 6.6, 7.7],  [2.1, 3.3,  5.5], [8.0] ])
    rs = get_row_splits(pt)
    exp = np.array([0, 5, 8, 9])
    print(f'rs={rs}, exp={exp}')
    np.testing.assert_array_equal(rs, exp)

 
if __name__ == '__main__':
    test_calc_dr()
    test_get_row_splits()
    test_gen_reco_matching()
    test_gen_truth_matching()
    test_calc_x_jes()
