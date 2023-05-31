import argparse, os, os.path as osp, math

import numpy as np

from common import logger, DATADIR, Columns


def split(directory, train_fraction=.9, seed=1001):
    """
    This script takes a path to a directory, and creates two directories that have the
    same structure, except that the npz files are split into train and test
    """
    directory = osp.abspath(directory)
    train_dir = osp.join(osp.dirname(directory), 'train_'+osp.basename(directory))
    test_dir = osp.join(osp.dirname(directory), 'test_'+osp.basename(directory))

    # Collect npz files 
    npzfiles = []
    for path, directories, files in os.walk(directory):
        npzfiles.extend(path+'/'+f for f in files if f.endswith('.npz'))

    # For every npz file, split it into train and test, and store it in
    # appropriate directories

    rng = np.random.default_rng(seed)

    for npz in npzfiles:
        dst_train = osp.join(train_dir, osp.relpath(npz, directory))
        dst_test = osp.join(test_dir, osp.relpath(npz, directory))

        cols = Columns.load(npz)

        # Pick (train_fraction*size) random indices for train
        sel_train = rng.choice(len(cols), math.ceil(train_fraction*len(cols)), replace=False)
        # Inverse selection for test (boolean mask instead of indices)
        sel_test = np.ones(len(cols), dtype=bool)
        sel_test[sel_train] = False

        cols_train = cols.copy()
        cols_train.arrays = {k:v[sel_train] for k, v in cols_train.arrays.items()}
        cols_train.save(dst_train)

        cols_test = cols.copy()
        cols_test.arrays = {k:v[sel_test] for k, v in cols_test.arrays.items()}
        cols_test.save(dst_test)
        
        logger.info(
            f'Splitting {osp.basename(npz)} from {len(cols)} entries to '
            f'train:{len(cols_train)} test:{len(cols_test)};'
            f' outfiles: {dst_train} and {dst_test}'
            )


def main():
    split(DATADIR+'/bkg')
    split(DATADIR+'/signal')


if __name__ == '__main__':
    main()
