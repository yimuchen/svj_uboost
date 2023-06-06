import os, os.path as osp, itertools, math, pprint
from time import strftime
import multiprocessing as mp

from common import logger


def worker(tup):
    variations, lpc_node_nr = tup

    for learning_rate, min_child_weight, max_depth, subsample, n_estimators in variations:
        tag = (
            f'lr{learning_rate:.2f}_mcw{min_child_weight:1.1f}_maxd{max_depth}'
            f'_subs{subsample:1.1f}_nest{n_estimators}'
            )

        outfile = strftime(f'models/svjbdt_%b%d_reweight_rho_{tag}.json')
        if osp.isfile(outfile):
            logger.info(f'File {outfile} exists, skipping')
            continue

        cmd = (
            f'python training.py xgboost'
            f' --reweight rho --ref data/train_signal/madpt300_mz250_mdark10_rinv0.3.npz'
            f' --node {lpc_node_nr} --tag {tag}'
            #f' --tag {tag}'
            f' --lr {learning_rate}'
            f' --minchildweight {min_child_weight}'
            f' --maxdepth {max_depth}'
            f' --subsample {subsample}'
            f' --nest {n_estimators}'
            )
        logger.info(f'Submitting command: {cmd}')
        os.system(cmd)


def main():
    variations = list(itertools.product(
        [.01, .05, .3], # learning rate
        [.1, 1.], # min_child_weight
        [4, 6], # max_depth
        [.6, 1.], # subsample
        [400, 850, 1500], # n_estimators
        ))
    logger.info(f'{len(variations)=}')

    # Divide all variations into 10 pools
    n_threads = 10
    n = math.ceil(len(variations) / n_threads)
    lpc_node_nr = 130
    mp_args = []
    for i in range(0, len(variations), n):
        mp_args.append([variations[i:i+10], lpc_node_nr])
        lpc_node_nr += 1
        logger.info(f'Variations training on node lpc{mp_args[-1][1]}: {pprint.pformat(mp_args[-1][0])}')


    pool = mp.Pool(n_threads)
    pool.map(worker, mp_args)
    pool.join()


if __name__ == '__main__':
    main()
