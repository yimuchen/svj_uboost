# based on files in https://github.com/boostedsvj/svj_jobs_new

import os, os.path as osp, json, argparse, fnmatch, random, math
from time import strftime

import jdlfactory
import seutils

from skim import dst

datadir = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SusyRA2Analysis2015/Run2ProductionV20'
mcyears = ['Summer20UL16', 'Summer20UL17', 'Summer20UL18']
sigdir = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/signal_production_3Dscan'
sigyears = [f'20{year}/NTUPLE/Private3DUL{year}' for year in [16,17,18]]

samples = {
    'data': {
        'JetHT': {
            'dir': datadir,
            'years': [
                'Run2016B-UL2016_HIPM-ver2-v2', 'Run2016C-UL2016_HIPM-v2', 'Run2016D-UL2016_HIPM-v2', 'Run2016E-UL2016_HIPM-v2', 'Run2016F-UL2016_HIPM-v2',
                'Run2016F-UL2016-v2', 'Run2016G-UL2016-v2', 'Run2016H-UL2016-v2',
                'Run2017B-UL2017-v1', 'Run2017C-UL2017-v1', 'Run2017D-UL2017-v1', 'Run2017E-UL2017-v1', 'Run2017F-UL2017-v1',
                'Run2018A-UL2018-v1', 'Run2018B-UL2018-v1', 'Run2018C-UL2018-v1', 'Run2018D-UL2018-v2',
            ],
        },
        'HTMHT': {
            'dir': datadir,
            'years': [
                'Run2016B-UL2016_HIPM-ver2-v1', 'Run2016C-UL2016_HIPM-v1', 'Run2016D-UL2016_HIPM-v1', 'Run2016E-UL2016_HIPM-v1', 'Run2016F-UL2016_HIPM-v1',
                'Run2016F-UL2016-v1', 'Run2016G-UL2016-v1', 'Run2016H-UL2016-v1',
                'Run2017B-UL2017-v1', 'Run2017C-UL2017-v1', 'Run2017D-UL2017-v1', 'Run2017E-UL2017-v1', 'Run2017F-UL2017-v2',
            ],
        },
    },
    'bkg': {
        'QCD_Pt': {
            'dir': datadir,
            'years': mcyears,
        },
        'TTJets_': {
            'dir': datadir,
            'years': mcyears,
        },
        'WJetsToLNu_': {
            'dir': datadir,
            'years': mcyears,
        },
        'ZJetsToNuNu_': {
            'dir': datadir,
            'years': mcyears,
        },
    },
    'sig': {
        'SVJ_': {
            'dir': sigdir,
            'years': sigyears,
        },
    },
}

def get_list_of_all_rootfiles(cat):
    """
    Gets list of all rootfiles in a category.
    Stores result in a cache file, since the operation is somewhat slow (~5 min).
    """
    cache_file = f'cached_{cat}_rootfiles.json'
    if osp.isfile(cache_file):
        jdlfactory.logger.info('Returning cached list of rootfiles')
        with open(cache_file, 'r') as f:
            return json.load(f)

    seutils.MAX_RECURSION_DEPTH = 1000
    jdlfactory.logger.info(f'Rebuilding {cat} filelist...')
    rootfiles = []
    for sample,info in samples[cat].items():
        sampledir = info['dir']
        for year in info['years']:
            pat = f'{sampledir}/{year}/{sample}*/*.root'
            jdlfactory.logger.info('Querying for pattern %s...', pat)
            rootfiles_for_pat = seutils.ls_wildcard(pat)
            jdlfactory.logger.info('  {} rootfiles found'.format(len(rootfiles_for_pat)))
            rootfiles.extend(rootfiles_for_pat)

    jdlfactory.logger.info('Caching list of {} rootfiles to {}'.format(len(rootfiles), cache_file))
    with open(cache_file, 'w') as f:
        json.dump(rootfiles, f)
    return rootfiles

def get_list_of_existing_dsts(stageout, cache_file='cached_existing_npzs.json'):
    if osp.isfile(cache_file):
        jdlfactory.logger.info('Returning cached list of existing npz files')
        with open(cache_file, 'r') as f:
            return json.load(f)

    jdlfactory.logger.info('Building list of all existing .npz files. This can take ~10-15 min.')
    seutils.MAX_RECURSION_DEPTH = 1000
    existing = []
    for path, _, files in seutils.walk(stageout):
        jdlfactory.logger.info(path)
        existing.extend(fnmatch.filter(files, '*.npz'))

    jdlfactory.logger.info('Caching list of {} npz files to {}'.format(len(existing), cache_file))
    with open(cache_file, 'w') as f:
        json.dump(existing, f)

    return existing

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--go', action='store_true', help="submit to condor (otherwise run locally)")
    parser.add_argument('--missing', action='store_true', help="create jobs for missing outputs")
    parser.add_argument('--listmissing', action='store_true', help="just list missing outputs")
    parser.add_argument('--categories', type=str, default='all', nargs='*', choices=samples.keys(), help="categories to process")
    parser.add_argument('-k', '--keep', type=float, default=None, help="keep fraction of samples")
    parser.add_argument('--stageout', type=str, help='stageout directory', required=True)
    parser.add_argument('--branch', type=str, default=None, help='svj_ntuple_processing branch')
    parser.add_argument('--impl', type=str, help='storage implementation', default='xrd', choices=['xrd', 'gfal'])
    parser.add_argument('--singlejob', action='store_true', help='Single job for testing.')
    args = parser.parse_args()

    if args.branch is not None:
        args.branch = '@'+args.branch
    else:
        args.branch = ''

    if args.categories=='all':
        args.categories = list(samples.keys())

    if args.missing or args.listmissing:
        existing_dsts = get_list_of_existing_dsts(args.stageout)

    for cat in args.categories:
        group = jdlfactory.Group.from_file('skim.py')
        group.venv(py3=True)
        group.sh('pip install git+https://github.com/boostedsvj/seutils')
        group.sh('pip install --ignore-installed --no-cache "numpy<2"')
        group.sh('pip install --no-cache awkward')
        group.sh('pip install --no-cache numba')
        group.sh('pip install git+https://github.com/boostedsvj/svj_ntuple_processing'+args.branch)

        group.htcondor['on_exit_hold'] = '(ExitBySignal == true) || (ExitCode != 0)'

        group.group_data['keep'] = args.keep
        group.group_data['category'] = cat
        group.group_data['stageout'] = args.stageout
        group.group_data['storage_implementation'] = args.impl
        group.group_data['local_copy'] = True

        # 64249 rootfiles, approx 10s per file means approx 180 CPU hours needed
        # do 5h per job --> 180/5 = 36 jobs with 1785 files each
        # Little bit more liberal with no. of jobs:
        # ~80 jobs -> 800 rootfiles per job
        # ~320 jobs -> 200 rootfiles per job
        n_per_job = 100
        if cat=='sig':
            # signals already aggregated during ntuple production
            n_per_job = 1

        rootfiles = get_list_of_all_rootfiles(cat)

        if args.missing or args.listmissing:
            # suffs in dst
            suffs = []
            if args.keep: suffs.append(f'keep{keep:.2f}')
            needed_dsts = [dst(f,args.stageout,suffs) for f in rootfiles]
            missing_dsts = set(needed_dsts) - set(existing_dsts)

            rootfiles_for_missing_dsts = []
            for d in missing_dsts:
                rootfiles_for_missing_dsts.append(rootfiles[needed_dsts.index(d)])
            rootfiles = rootfiles_for_missing_dsts

        if args.listmissing:
            rootfiles.sort()
            jdlfactory.logger.info(
                'Missing %s .npz files for the following rootfiles:\n%s',
                len(missing_dsts),
                '  '+'\n  '.join(rootfiles)
                )
            continue

        if args.missing:
            random.shuffle(rootfiles) # To avoid job load imbalance
            if cat!='sig': n_per_job = 10
            jdlfactory.logger.info(
                'Missing %s .npz files; submitting %s jobs',
                len(missing_dsts),
                int(math.ceil(len(missing_dsts)/float(n_per_job)))
                )

        for i in range(0, len(rootfiles), n_per_job):
            group.add_job({'rootfiles' : rootfiles[i:i+n_per_job]})
            if args.singlejob: break

        group_name = strftime(f'{cat}feat_%Y%m%d_%H%M%S')
        if args.missing: group_name += '_missing'
        if args.keep: group_name += '_'+suffs[0]

        if args.go:
            group.prepare_for_jobs(group_name)
            os.system('cd {}; condor_submit submit.jdl'.format(group_name))
        else:
            group.run_locally(keep_temp_dir=False)

if __name__ == '__main__':
    main()
