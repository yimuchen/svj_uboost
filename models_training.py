import os, os.path as osp, itertools, math, pprint
from time import strftime
import multiprocessing as mp

from common import logger


def worker(tup):
    variations, lpc_node_nr = tup

    for mdark, rinv in variations:
        tag = (
            f'mdark{mdark:1.1f}_rinv{rinv:1.1f}'
            )

        outfile = strftime(f'models/svjbdt_%b%d_{tag}.json')
        if osp.isfile(outfile):
            logger.info(f'File {outfile} exists, skipping')
            continue

        cmd = (
            f'python training.py xgboost'
            f' --node {lpc_node_nr} --tag {tag}'
            f' --tag {tag}'
            f' --mdark {mdark}'
            f' --rinv {rinv}'
            )
        logger.info(f'Submitting command: {cmd}')
        os.system(cmd)


def main():
    variations = list(itertools.product(
        [1, 5, 10], # mdarks
        [.1, .3, 0.7], # rinv
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
