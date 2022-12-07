import os, os.path as osp, json, argparse


import matplotlib.pyplot as plt
import numpy as np

from common import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonfile', type=str)
    args = parser.parse_args()

    plotdir = 'plots_' + args.jsonfile.replace('.json','')
    if not osp.isdir(plotdir): os.makedirs(plotdir)

    with open(args.jsonfile, 'r') as f:
        d = json.load(f)

    d = d['histograms']

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    for bdtcut, histograms in d.items():
        for name, hist in histograms.items():
            ax.step(hist['binning'][:-1], hist['vals'], where='pre')
            is_bkg = 'mz' not in hist['metadata']
            outfile = osp.join(plotdir, f'bdt{bdtcut}_{"bkg" if is_bkg else "sig"}_{name}.png')
            ax.set_title(f'bdt{bdtcut}_{name}')
            logger.info(f'Saving to {outfile}')
            plt.savefig(outfile, bbox_inches='tight')
            ax.clear()



if __name__ == '__main__':
    main()