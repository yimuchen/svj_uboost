import os, os.path as osp, glob, pickle, logging, argparse, sys, re, pprint
from time import strftime

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1001)

from common import logger, DATADIR, Columns, time_and_log, columns_to_numpy, set_matplotlib_fontsizes, imgcat, add_key_value_to_json, filter_pt, mt_wind


training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    ]
all_features = training_features + ['rho']


def reweight(reference, samples, reweight_var, make_test_plot=False):
    # For the reference model, the new 'reweight' is equal to the old 'weight'
    reference.arrays['reweight'] = np.copy(reference.arrays['weight']) * np.copy(reference.arrays['puweight'])

    # Binning is hand-tuned for now
    reweight_bins = dict(
        girth = np.linspace(0., 1.5, 40),
        pt = np.linspace(0., 1000, 40),
        mt = np.linspace(0., 1000, 40),
        rho = np.linspace(-10., 0.5, 40)
        )[reweight_var]

    logger.info(
        f'Reweighting for variable {reweight_var};'
        f' histogram {reweight_bins[0]} to {reweight_bins[-1]} with {reweight_bins.shape[0]-1} bins;'
        f' last bin will be treated as overflow bin.'
        )
    reweight_bins[-1] = np.inf

    logger.info(f'Reference sample: {reference.metadata}')

    # Get the reference histogram, distribution of the reweight_var in the reference sample
    reference_hist, _ = np.histogram(reference.arrays[reweight_var], bins=reweight_bins)

    for sample in samples:
        if sample is reference or sample.metadata == reference.metadata: continue
        hist, _ = np.histogram(sample.arrays[reweight_var], bins=reweight_bins)
        reweight_hist = np.where(hist>0, reference_hist/hist, 0.)
        vals = sample.arrays[reweight_var]
        sample.arrays['reweight'] = np.copy(sample.arrays['weight']) * np.copy(sample.arrays['puweight'])
        for i, (left, right) in enumerate(zip(reweight_bins[:-1], reweight_bins[1:])):
            sample.arrays['reweight'][(left < vals) &  (vals <= right)] *= reweight_hist[i]

    if make_test_plot:
        logger.info(f'Making test plot for reweighting with variable {reweight_var}')

        # sample = [s for s in samples if s.metadata.get('bkg_type',None) == 'qcd' and s.metadata['ptbin'][0]==470][0]
        sample = [s for s in samples if osp.basename(s.metadata['src'])=='TTJets_TuneCP5_13TeV-madgraphMLM-pythia8.npz'][0]

        set_matplotlib_fontsizes()
        fig = plt.figure(figsize=(13,13))
        ax = fig.gca()

        ax.hist(
            reference.arrays[reweight_var], bins=reweight_bins,
            density=True, histtype='step',
            label=f'reference: {osp.basename(reference.metadata["src"])}'
            )
        ax.hist(
            sample.arrays[reweight_var], bins=reweight_bins,
            density=True, histtype='step',
            label=f'unreweighted: {osp.basename(sample.metadata["src"])}'
            )
        ax.hist(
            sample.arrays[reweight_var],
            bins=reweight_bins, density=True, histtype='step',
            weights=sample.arrays['reweight'],
            label=f'reweighted: {osp.basename(sample.metadata["src"])}',
            linestyle='dashed', color='red', linewidth=2
            )

        ax.set_title(f'Reweight: {reweight_var}', fontsize=30)
        ax.legend()
        ax.set_xlabel(reweight_var)
        ax.set_ylabel('a.u.')
        outfile = f'plots/reweighting_test_plot_{reweight_var}.png'
        plt.savefig(outfile, bbox_inches='tight')
        imgcat(outfile)



def print_weight_table(bkg_cols, signal_cols, weight_col='weight'):
    bkg_cols.sort(key=lambda s: (s.metadata['bkg_type'], s.metadata.get('ptbin',[0,0]), s.metadata.get('htbin',[0,0])))
    signal_cols.sort(key=lambda s: (s.metadata['mz'], s.metadata['rinv']))

    bkg_weights = []
    table = []

    for cols in bkg_cols:
        weight = cols.arrays[weight_col]
        weight_per_event = np.mean(weight) if len(weight) else -1.
        total_weight = np.sum(weight)
        bkg_weights.append(weight)
        table.append([
            osp.basename(cols.metadata['src']).replace('.npz',''),
            len(weight), weight_per_event, total_weight
            ])

    # Set signal weight equal to bkg weight

    total_bkg_weight = sum(np.sum(w) for w in bkg_weights)

    signal_weights = []
    for cols in signal_cols:
        signal_weights.append(np.ones(len(cols))/len(cols) if weight_col in {'manual', 'weight'} else cols.arrays[weight_col])

    total_signal_weight = sum(np.sum(w) for w in signal_weights)

    for weight, cols in zip(signal_weights, signal_cols):
        weight *= total_bkg_weight / total_signal_weight
        weight_per_event = np.mean(weight)
        total_weight = np.sum(weight)
        bkg_weights.append(weight)
        table.append([
            osp.basename(cols.metadata['src']).replace('.npz',''),
            len(weight), weight_per_event, total_weight
            ])

    total_weight = sum(r[-1] for r in table)    

    # Print the table
    width = max(len(r[0]) for r in table)
    print(f'{"Sample":{width}}  {"n_events":>10}  {"weight/evt":>14}  total_weight')
    for r in table:
        print(f'{r[0]:{width}}  {r[1]:10d}  {r[2]:14.9f}  {r[3]:6.3f} ({100.*r[3]/total_weight:.7f}%)')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['uboost', 'xgboost'])
    parser.add_argument('--reweight', type=str)
    parser.add_argument('--reweighttestplot', action='store_true')
    parser.add_argument('--downsample', type=float, default=.4)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--node', type=str, help='Run training on a different lpc node.')
    parser.add_argument('--tag', type=str, help='Add some output to the output model file')
    # Some test flags for alternative trainings
    parser.add_argument('--gradientboost', action='store_true')
    parser.add_argument('--use_eta', action='store_true')
    parser.add_argument('--ref', type=str, help='path to the npz file for the reference distribution for reweighting.')
    # adding signal models
    parser.add_argument('--mdark', type=str, default='10.')
    parser.add_argument('--rinv', type=str, default='0.3')
    args, leftover_args = parser.parse_known_args()

    global training_features
    if args.use_eta:
        training_features.append('eta')

    if args.node:
        # Just get the first integer from the args.node string, and parse it to a valid lpc address
        print(args.node)
        node_nr = re.search(r'\d+', args.node).group()
        # Delete the --node argument from the command line
        args = sys.argv[:]
        i = args.index('--node'); args.pop(i+1); args.pop(i)
        # Submit the exact same command on a different node
        cmd = f'ssh -o StrictHostKeyChecking=no cmslpc{node_nr}.fnal.gov "cd /uscms/home/klijnsma/svj/bdt/v3/svj_uboost; conda activate svj-bdt-light; nohup python ' + ' '.join(args) + '"'
        #cmd = f'ssh -o StrictHostKeyChecking=no hepcms{node_nr}.umd.edu "cd /uscms/home/klijnsma/svj/bdt/v3/svj_uboost; conda activate svj-bdt-light; nohup python ' + ' '.join(args) + '"'
        logger.info(f'Executing: {cmd}')
        os.system(cmd)
        return

    # Add a file logger for easier monitoring
    if not args.dry:
        if not osp.isdir('logs'): os.makedirs('logs')
        fmt = f'[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s {os.uname()[1]}] %(message)s'
        file_handler = logging.FileHandler(strftime('logs/log_train_%b%d.txt'))
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    logger.info(f'Running training script; args={args}')

    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*.npz')]
    bkg_cols = [
        Columns.load(f) for f in
        glob.glob(DATADIR+'/train_bkg/Summer20UL18/QCD_*.npz')
        + glob.glob(DATADIR+'/train_bkg/Summer20UL18/TTJets_*.npz')
        ]

    if args.mdark:
        signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*mdark'+args.mdark+'*.npz')]
    if args.rinv:
        signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*rinv'+args.rinv+'*.npz')]

    # Throw away the very low QCD bins (very low number of events)
    logger.info('Using QCD bins starting from pt>=300')
    # bkg_cols = list(filter(lambda cols: cols.metadata['bkg_type']!='qcd' or cols.metadata['ptbin'][0]>=300., bkg_cols))
    bkg_cols = filter_pt(bkg_cols, 300.)
    #bkg_cols = mt_wind(bkg_cols, 180, 650)
    #signal_cols = mt_wind(signal_cols, 180, 650)

    logger.info(f'Training features: {training_features}')

    if args.model == 'uboost':
        import pandas as pd
        from hep_ml import uboost

        print_weight_table(bkg_cols, signal_cols, 'weight')
        X, y, weight = columns_to_numpy(signal_cols, bkg_cols, all_features, downsample=.2)
        logger.info(f'Using {len(y)} events ({np.sum(y==1)} signal events, {np.sum(y==0)} bkg events)')
        X_df = pd.DataFrame(X, columns=all_features)

        if args.gradientboost:
            from hep_ml.gradientboosting import UGradientBoostingClassifier
            from hep_ml.losses import BinFlatnessLossFunction, KnnFlatnessLossFunction
            
            base_tree = uboost.DecisionTreeClassifier(max_depth=4)
            model = UGradientBoostingClassifier(
                #loss=BinFlatnessLossFunction(uniform_features=['mt'], uniform_label=0, n_bins=50),
                loss=BinFlatnessLossFunction(uniform_features=['rho'], uniform_label=0, n_bins=50),
                n_estimators=100,
                train_features=training_features,
                max_depth=4
                )
            outfile = strftime('models/uboost_%b%d_gradbin.pkl')
        else:
            base_tree = uboost.DecisionTreeClassifier(max_depth=4)
            model = uboost.uBoostClassifier( # "uBoostBDT" in the uBoost documentation
                #uniform_features=['mt'], uniform_label=0,
                uniform_features=['rho'], uniform_label=0,
                base_estimator=base_tree,
                train_features=training_features,
                n_estimators=100,
                n_threads=4,
                n_neighbors=30
                )
            outfile = strftime('models/uboost_%b%d_knn.pkl')
        
        if args.dry:
            logger.info('Dry mode: Quitting')
            return
        with time_and_log('Begin training uBoost. This can take >24h hours...'):
            model.fit(X_df, y, weight)
        if not osp.isdir('models'): os.makedirs('models')
        
        if args.tag: outfile = outfile.replace('.pkl', f'_{args.tag}.pkl')

        logger.info('Dumping trained model to %s', outfile)
        with open(outfile, 'wb') as f:
            pickle.dump(model, f)

    elif args.model == 'xgboost':
        # Parse the leftover arguments to see if there are any hyper parameters

        hyperpar_parser = argparse.ArgumentParser()
        hyperpar_parser.add_argument('--lr', dest='eta', type=float)
        hyperpar_parser.add_argument('--minchildweight', dest='min_child_weight', type=float)
        hyperpar_parser.add_argument('--maxdepth', dest='max_depth', type=int)
        hyperpar_parser.add_argument('--subsample', type=float)
        hyperpar_parser.add_argument('--nest', dest='n_estimators', type=int)
        hyperpar_args = hyperpar_parser.parse_args(leftover_args)

        parameters = dict( # Base parameters
            eta=.05,
            max_depth=4,
            n_estimators=850,
            )
        # Update with possible command line options
        parameters.update({k:v for k, v in hyperpar_args.__dict__.items() if v is not None})
        logger.warning(f'Using the following hyperparameters:\n{pprint.pformat(parameters)}')

        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, **parameters)

        if args.reweight:
            logger.info(f'Reweighting to {args.reweight}')
            # Add a 'reweight' column to all samples:
            cols = bkg_cols + signal_cols
            if args.ref:
                reference_col = Columns.load(osp.abspath(args.ref))
                reference = [col for col in cols if col.metadata == reference_col.metadata][0]
            else:
                # Use a default reference of mz=350, rinv=.3
                reference = [s for s in signal_cols if s.metadata['mz']==350 and s.metadata['rinv']==.3][0]
            logger.info(f'Using as a reference: {reference.metadata}')

            cols.remove(reference)
            reweight(reference, cols, args.reweight, make_test_plot=args.reweighttestplot)

            print('Weight table BEFORE reweighting:')
            print_weight_table(bkg_cols, signal_cols, 'weight')
            print('\nWeight table AFTER reweighting:')
            print_weight_table(bkg_cols, signal_cols, 'reweight')
            if args.reweighttestplot: return

            # Get samples using the new 'reweight' key (instead of the default 'weight')
            X, y, weight = columns_to_numpy(
                signal_cols, bkg_cols, training_features,
                weight_key='reweight', downsample=args.downsample
                )
            weight *= 100. # For training stability
            outfile = strftime(f'models/svjbdt_%b%d_reweight_{args.reweight}_allsignals_ttjets_refmz250.json')
        else:
            print_weight_table(bkg_cols, signal_cols, 'weight')
            X, y, weight = columns_to_numpy(
                signal_cols, bkg_cols, training_features,
                downsample=args.downsample
                )
            outfile = strftime('models/svjbdt_%b%d_allsignals_qcdttjets.json')

        logger.info(f'Using {len(y)} events ({np.sum(y==1)} signal events, {np.sum(y==0)} bkg events)')

        if args.use_eta: outfile = outfile.replace('.json', '_eta.json')
        if args.tag: outfile = outfile.replace('.json', f'_{args.tag}.json')

        if args.dry:
            logger.info('Dry mode: Quitting')
            return
        with time_and_log(f'Begin training, dst={outfile}. This can take a while...'):
            model.fit(X, y, sample_weight=weight)
        if not osp.isdir('models'): os.makedirs('models')
        model.save_model(outfile)
        logger.info(f'Dumped trained model to {outfile}')
        add_key_value_to_json(outfile, 'features', training_features)


if __name__ == '__main__':
    main()
