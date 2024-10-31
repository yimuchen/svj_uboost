# originally from https://github.com/boostedsvj/svj_local_scripts

import glob, os, os.path as osp
import seutils
import svj_ntuple_processing as svj

def expand_wildcards(pats):
    seutils.MAX_RECURSION_DEPTH = 100000
    if not isinstance(pats,list): pats = [pats]
    expanded = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded

def process_directory(tup):
    directory, outfile = tup
    if directory[-1]!='/': directory += '/'
    npzfiles = expand_wildcards([directory+'*.npz'])
    svj.logger.info(f'Processing {directory} -> {outfile} ({len(npzfiles)} files)')
    cols = []
    for f in npzfiles:
        try:
            cols.append(svj.Columns.load(f, encoding='latin1'))
        except Exception as e:
            svj.logger.error(f'Failed for file {f}, error:\n{e}')
    concatenated = svj.concat_columns(cols)
    concatenated.save(outfile,force=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stageout', type=str, required=True)
    parser.add_argument('--skip-existing', default=False, action="store_true")
    parser.add_argument('-n', '--nthreads', default=10, type=int)
    parser.add_argument('directories', nargs='+', type=str)
    args = parser.parse_args()
    if args.stageout[-1]!='/':
        args.stageout += '/'

    directories = expand_wildcards(args.directories)

    fn_args = []
    for d in directories:
        outfile = args.stageout+'/'.join(d.split('/')[-2:])+'.npz'
        if args.skip_existing and seutils.isfile(outfile):
            svj.logger.info('    File %s exists, skipping', outfile)
            continue
        fn_args.append((d, outfile))

    import multiprocessing as mp
    p = mp.Pool(args.nthreads)
    p.map(process_directory, fn_args)
    p.close()
    p.join()

