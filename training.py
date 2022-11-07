import os, os.path as osp, glob, pickle, logging
from time import strftime

import numpy as np
import pandas as pd
from hep_ml import uboost

np.random.seed(1001)

from common import logger, DATADIR, Columns


training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi'
    ]
all_features = training_features + ['mt']


def main():
    # Add a logger to a file for easier monitoring
    file_handler = logging.FileHandler(strftime('log_train_%b%d.txt'))
    file_handler.setFormatter(logging.Formatter(
        fmt = '[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        ))
    logger.addHandler(file_handler)

    # Load training data
    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_bkg/Summer20UL18/QCD_*.npz')]
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*.npz')]

    X = []
    weight = []
    y = []

    # Get the features for the bkg samples
    for cols in qcd_cols:
        # Randomly select only 40% of the background events,
        # otherwise you encounter memory issues during training
        select = np.random.choice(len(cols), int(.4*len(cols)), replace=False)
        X.append(cols.to_numpy(all_features)[select])
        # Use the 'Weight' column from TreeMaker
        # I manually checked that it agrees with (xs * presel_eff / size)
        weight.append(cols.arrays['weight'][select])
        # Label is zero for background
        y.append(np.zeros(len(select)))

    # Set global signal weight equal to global background weight
    total_bkg_weight = sum(sum(w) for w in weight)
    total_signal_events = sum(len(cols) for cols in signal_cols)
    signal_weight = total_bkg_weight / total_signal_events

    # Get the features for the signal samples
    for cols in signal_cols:
        X.append(cols.to_numpy(all_features))
        weight.append(signal_weight*np.ones(len(cols)))
        y.append(np.ones(len(cols)))

    # Concatenate and turn features into pd.DataFrame
    X = pd.DataFrame(np.concatenate(X), columns=all_features)
    weight = np.concatenate(weight)
    y = np.concatenate(y)

    logger.info(f'Training shape: {X.shape}')

    base_tree = uboost.DecisionTreeClassifier(max_depth=4)
    model = uboost.uBoostClassifier(
        uniform_features=['mt'], uniform_label=0,
        base_estimator=base_tree,
        train_features=training_features,
        n_estimators=100,
        )
    
    logger.info('Begin training. This can take 5-10 hours...')
    model.fit(X, y, weight)
    
    outfile = strftime('model_uboost_%b%d.pkl')
    logger.info('Dumping trained model to %s', outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()