import os, os.path as osp
import seutils

from common import logger


DATADIR = osp.join(osp.dirname(osp.abspath(__file__)), 'data')


def copy(tup):
    src, dirname = tup
    logger.info('Copying %s -> %s', src, DATADIR)
    seutils.cp(src, DATADIR, recursive=True, parallel=4)
    wrong_named_dir = osp.join(DATADIR, osp.basename(src))
    correct_named_dir = osp.join(DATADIR, dirname)
    logger.info('Renaming %s -> %s', wrong_named_dir, correct_named_dir)
    os.rename(wrong_named_dir, correct_named_dir)


def main():
    if not osp.isdir(DATADIR): os.makedirs(DATADIR)

    #base_dir = '/home/snabili/data/svj_local_scripts/metdphicuts_forBDT/'
    #base_dir = '/home/snabili/data/svj_local_scripts/rinv_variation/'
    #base_dir = '/home/snabili/data/svj_local_scripts/metdphicuts_forBDT/'
    #base_dir = '/home/snabili/data/svj_local_scripts/rho_bdt_allfiles/'
    base_dir = '/home/snabili/data/svj_local_scripts/puweight/'
    #signal_dir = base_dir + 'sig_rinv0.3'
    signal_dir = base_dir + 'signal'
    bkg_dir = base_dir + 'bkg'
    #signal_dir_notruthcone = base_dir + '/signal_apr27_notruthcone'

    fn_args = [
        (signal_dir, 'signal'),
        #(signal_dir_notruthcone, 'signal_notruthcone'),
        (bkg_dir, 'bkg'),
        ]

    import multiprocessing as mp
    p = mp.Pool(3)
    p.map(copy, fn_args)
    p.close()
    p.join()



if __name__ == '__main__':
    main()
