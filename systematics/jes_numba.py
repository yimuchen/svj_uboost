"""
This module contains numba JIT compiled code to calculate the JES factors.
It requires numba to run:

pip install numba
"""

import numpy as np
import awkward as ak


import sys
import os.path as osp
THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(THIS_DIR))
from skim import gen_reco_matching, gen_truth_matching, calc_drsq, get_row_splits, calc_x_jes_rowsplits, calc_x_jes

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
