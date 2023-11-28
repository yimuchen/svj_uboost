import os, os.path as osp, argparse, glob
from collections import OrderedDict

import numpy as np

from common import Columns, columns_to_numpy, DATADIR, filter_pt, set_matplotlib_fontsizes, imgcat


def format_val(s, ndec=2):
    """
    Turn something into a string
    """
    try:
        number_f = float(s)
        number_i = int(s)
        if number_f != number_i:
            return f'{s:.{ndec}f}'
        else:
            return f'{s:.0f}'
    except ValueError:
        return str(s)


def transpose_table(table):
    """
    Turns rows into columns. Does not check if all rows have the same length!
    Example: [[1,2,3], [4,5,6]] --> [[1,4], [2,5], [3,6]]
    """
    return [list(x) for x in zip(*table)]


def format_table(table, col_sep=' ', row_sep='\n', transpose=False, ndec=2):
    table = [ [format_val(c, ndec=ndec) for c in row ] for row in table ]
    if transpose: table = transpose_table(table)
    col_widths = [ max(map(len, column)) for column in zip(*table) ]
    return row_sep.join(
        col_sep.join(f'{col:{w}s}' for col, w in zip(row, col_widths)) for row in table
        )


def column_name(c):
    if 'colname' in c.metadata: return c.metadata['colname']
    elif 'mz' in c.metadata:
        return 'mz{mz:.0f}_rinv{rinv:.1f}'.format(**c.metadata)
    elif 'htbin' in c.metadata:
        return 'ht{:.0f}-{:.0f}'.format(*c.metadata['htbin'])
    elif 'ptbin' in c.metadata:
        return 'pt{:.0f}-{:.0f}'.format(*c.metadata['ptbin'])
    elif c.metadata.get('bkg_type',None) == 'wjets':
        return 'incl'
    elif c.metadata.get('bkg_type',None) == 'ttjets':
        if c.metadata.get('n_lepton_sample', None):
            name = f'{c.metadata["n_lepton_sample"]}lep'
            if c.metadata.get('genmet_sample', False):
                name += '_met150'
            return name
        else:
            return 'incl'
    else:
        raise Exception(f'Could not make a name for {c}')


def combined_column(cols):
    combined = Columns()
    total_xs = sum(c.xs for c in cols)
    combined.manual_xs = total_xs
    combined.metadata['colname'] = 'combined'

    sumxs = lambda x: sum(v*c.xs/total_xs for v, c in zip(x, cols))
    combined.cutflow = OrderedDict()
    for k in cols[0].cutflow.keys():
        combined.cutflow[k] = sumxs(c.cutflow[k]/c.cutflow['raw'] for c in cols)
    return combined


def make_column(c):
    s = [column_name(c)]
    s.append(f'{c.xs:.2e}')
    raw = c.cutflow['raw']
    for val in c.cutflow.values():
        s.append(f'{100*val/raw:.2f}%')
    #for i in range(1,len(c.cutflow)):   
        #s.append(100*c.cutflow.values()[i]/c.cutflow.values[i-1])

    n137 = c.xs * c.presel_eff * 137.2*1e3
    s.append(f'{n137:.2e}')
    return s


def header_column(c):
    return ['', 'xs'] + list(c.cutflow.keys()) + ['n137']


def make_table(cols, combined=True):
    table = [header_column(cols[0])]+[make_column(c) for c in cols]
    if combined: table.append(make_column(combined_column(cols)))
    return transpose_table(table)

#sig_DATADIR = '/home/snabili/data/svj_local_scripts/rho_bdt_allfiles'
#bkg_DATADIR = '/home/snabili/hadoop/BKG/Ultra_Legacy/HADD_BKGBDT/Summer20UL18'
sig_DATADIR = '/home/snabili/hadoop/HADD_puweight'
bkg_DATADIR = '/home/snabili/hadoop/HADD_puweight/bkg/Summer20UL18'
def collect_columns():
    #signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/signal_notruthcone/*.npz')]
    signal_cols = [Columns.load(f) for f in glob.glob(sig_DATADIR+'/signal_notruth/*mdark10_rinv0.3.npz')]
    signal_cols.sort(key=lambda s: (s.metadata['mz'], s.metadata['rinv']))

    bkg_cols = [Columns.load(f) for f in glob.glob(bkg_DATADIR+'/*.npz')]
    bkg_cols = filter_pt(bkg_cols, 170.)
    bkg_cols = [c for c in bkg_cols if not(c.metadata['bkg_type']=='wjets' and 'htbin' not in c.metadata)]

    def sort_key(c):
        return (c.metadata.get('ptbin', [-1,-1])[0], c.metadata.get('htbin', [-1,-1])[0])
    bkg_cols.sort(key=sort_key)

    bkg_cols_per_type = OrderedDict()
    bkg_types = list(sorted(set(b.metadata['bkg_type'] for b in bkg_cols)))

    for bkg_type in bkg_types:
        bkg_cols_per_type[bkg_type] = [c for c in bkg_cols if c.metadata['bkg_type']==bkg_type]

    return signal_cols, bkg_cols, bkg_cols_per_type


def print_cutflow_tables():
    signal_cols, bkg_cols, bkg_cols_per_type = collect_columns()

    for bkg_type, cols in bkg_cols_per_type.items():
        print('-'*80)
        print(bkg_type)
        print(format_table(make_table(cols)))

    print('-'*80)
    print('bkg_summary')

    combined_bkg_cols = []
    for bkg_type, cols in bkg_cols_per_type.items():
        combined_bkg_cols.append(combined_column(cols))
        combined_bkg_cols[-1].metadata['colname'] = bkg_type
    print(format_table(make_table(combined_bkg_cols)))

    print('-'*80)
    print('signal')
    print(format_table(make_table(signal_cols, combined=False)))


def print_cutflow_tables_rinv0p3_only():
    signal_cols, bkg_cols, bkg_cols_per_type = collect_columns()
    signal_cols = [s for s in signal_cols if s.metadata['rinv']==0.3]
    print('-'*80)
    print('signal')
    print(format_table(make_table(signal_cols, combined=False)))


def n137_plots():
    import matplotlib.pyplot as plt
    set_matplotlib_fontsizes()

    signal_cols, bkg_cols, bkg_cols_per_type = collect_columns()

    # group signals by rinv
    rinvs = list(sorted(set(s.metadata['rinv'] for s in signal_cols)))

    fig = plt.figure(figsize=(7,7))
    ax = fig.gca()

    for rinv in rinvs:

        sigs = [s for s in signal_cols if s.metadata['rinv']==rinv]
        sigs.sort(key=lambda s: s.metadata['mz'])
        mzs = np.array([s.metadata['mz'] for s in sigs])
        n137s = np.array([s.xs * s.presel_eff * 137.2*1e3 for s in sigs])
        ax.plot(mzs, n137s, '-o', label=f'$r_{{inv}}={rinv}$')

    mzs_2022 = np.array([250, 300, 350, 400, 450, 500, 550, 600])
    n137_2022 = np.array([91000, 82043, 67465, 55833, 48725, 42429, 36235, 38033])
    ax.plot(mzs_2022, n137_2022, '-o', label='$r_{inv}=0.3 (2022)$')

    ax.legend()
    ax.set_xlabel('$m_{Z\\prime}$ (GeV)')
    ax.set_ylabel('$N_{events}$ @ 137.2 $fb^{-1}$')
    plt.savefig('n137.png', bbox_inches='tight')
    imgcat('n137.png')



if __name__ == '__main__':
    n137_plots()
    # print_cutflow_tables_rinv0p3_only()
    print_cutflow_tables()
