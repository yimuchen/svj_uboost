import os, stat, sys
import argparse
from collections import OrderedDict, defaultdict
import svj_ntuple_processing as svj
import common
from hadd_skims import expand_wildcards

namesDict = OrderedDict([
    ('raw', ''),
    ('stitch', ''),
    ('n_ak15jets>=2', r'$\njet\geq2$'),
    ('ak15jets_id', 'Jet ID'),
    ('subl_eta<2.4', r'$\abs{\eta(\widejet_2})<2.4$'),
    ('triggers', r'Triggers'),
    ('ak8jet.pt>500', r'$\pt(\trigjet_1) > 500\GeV$'),
    ('subl_ecf>0', r'$\pt(\widejet_2^{\text{SD}}) > 100\GeV$'),
    ('rtx>1.1', r'$\RTx>1.1$'),
    ('muonpt<1500', r'$\pt(\Pgm)<1500\GeV$'),
    ('nmuons=0', r'$N_{\Pgm}=0$'),
    ('nelectrons=0', r'$N_{\Pe}=0$'),
    ('metfilter', 'MET filters'),
    ('n_ak4jets>=2', r'$\nnarrow\geq2$'),
    ('ecaldeadcells', 'Custom MET filter'),
    ('abs(metdphi)<1.5', r'$\Delta\phi(\MET,\widejet_2)<1.5$'),
    ('180<mt<650', r'$180<\MTx<650\GeV$'),
    ('preselection', r'Preselection'),
    ('rt_signalregion', r'$\RTx>1.18$'),
    ('cutbased', r'$\text{ecf}\ecfM>0.09$'),
])

def shortname(sig):
    keys = ['mz','mdark','rinv']
    return '_'.join([f'{k}{sig[k]}' for k in keys])

# implementation of recursive loop over any number of dimensions
# creates grid of all possible combinations of parameter values
def varyAll(pos,paramlist,sig,sigs):
    param = paramlist[pos][0]
    vals = paramlist[pos][1]
    for v in vals:
        stmp = sig[:]+[v]
        # check if last param
        if pos+1==len(paramlist):
            sigs.append(tuple(stmp))
        else:
            varyAll(pos+1,paramlist,stmp,sigs)

# expand into raw, abs, rel
# todo: error calculations...
def compute_cutflow(cutflow):
    nentries = None
    prev = 1
    for key,val in cutflow.items():
        if nentries is None: nentries = val
        cutflow[key] = {"raw": val, "abs": val/nentries*100, "rel": val/prev*100}
        prev = val
    return cutflow

def print_val(val, prcsn, minprec=0):
    sval = ''
    new_prcsn = prcsn
    has_enough = False
    while not has_enough:
        sval = f'{val:.{new_prcsn}f}'
        if minprec==0 or val==type(val)(0): break # no need for anything further in these cases
        tmp = sval[:]
        leading_chars = 0
        # remove 0.000 from 0.000x (stop at first non-zero char)
        for i in range(len(tmp)):
            if tmp[i]=='0' or tmp[i]=='.': leading_chars += 1
            else: break
        tmp = tmp[leading_chars:]
        has_enough = len(tmp)>=minprec
        if not has_enough: new_prcsn += 1
    return sval

if __name__=='__main__':
    # define options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir", type=str, default="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20240718_hadd", help="location of root files (PFN)")
    parser.add_argument("-o", "--outname", type=str, default="svj_cutflow.tex", help="output TeX file name")
    parser.add_argument("-t", "--type", type=str, default="abs", choices=['abs','rel','raw','rawrel'], help="type of cutflow (abs, rel, raw, rawrel)")
    parser.add_argument("-p", "--prec", type=int, default=1, help="numerical precision of output")
    parser.add_argument("-m", "--minprec", type=int, default=1, help="minimum number of digits to display")
    parser.add_argument("-e", "--error", default=False, action="store_true", help="display statistical errors")
    parser.add_argument("-u", "--summarize-error", dest="summarizeerror", default=False, action="store_true", help="summarize maximum value of statistical errors (goes w/ -e)")
    parser.add_argument("-z", "--skipzeros", default=False, action="store_true", help="ignore errors on zero-count bins")
    parser.add_argument("-k", "--skiplines", type=str, default=[], nargs='*', help="line to skip in cutflow")
    parser.add_argument("-l", "--lastline", type=str, default=None, help="line to stop at in cutflow")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="verbose output")
    parser.add_argument("-P", "--procs", type=str, default="all", help="category of procs to include")
    parser.add_argument("-E", "--efficiency", default=False, action="store_true", help="include line for overall efficiency")
    parser.add_argument("-D", "--align-decimal", dest="alignDecimal", default=False, action="store_true", help="align columns at decimal")
    parser.add_argument("--topcapt", dest="topcapt", default=False, action="store_true", help="use topcaption instead of caption")
    args = parser.parse_args()

    if args.dir[-1]!='/': args.dir += '/'
    args.efficiency = args.efficiency or 'raw' in args.type

    outDict = OrderedDict([])
    outDict["header1"] = r"\multicolumn{2}{c}{Selection}"

    outDict = OrderedDict(list(outDict.items())+list(namesDict.items()))
    if args.efficiency:
        outDict["efficiency"] = r"\multicolumn{2}{c}{Efficiency [\%]}"

    bkg_names = OrderedDict([
        ('qcd', 'QCD'),
        ('ttjets', r'\ttbar'),
        ('wjets', r'\wjets'),
        ('zjets', r'\zjets'),
    ])

    # signal groups
    benchmarks = OrderedDict([
        ("mz", 350),
        ("mdark", 10),
        ("rinv", 0.3),
    ])
    sig_vals = OrderedDict([
        ("mz", [250, 350, 450]),
        ("mdark", [1, 5, 10]),
        ("rinv", [0.1, 0.3, 0.6, 0.9]),
    ])
    sig_units = {"mz": r"\GeV", "mdark": r"\GeV", "rinv": ""}
    sig_names = OrderedDict()
    if args.procs=='all':
        sig_names = OrderedDict([
            (shortname(benchmarks), 'signal'),
        ])
    elif args.procs.startswith("sig"):
        bkg_names = OrderedDict()
        procsplit = args.procs.split('_')
        varied = procsplit[1]
        mZprime = int(procsplit[2])
        sig_captions = []
        for i,p in enumerate(sig_vals):
            if p==varied:
                varied_ind = i
                sig_captions.append("varying \\{} values".format(p))
                continue
            elif p=="mz": sig_vals[p] = [mZprime]
            else: sig_vals[p] = [benchmarks[p]]
            sig_captions.append("$\\{} = {}{}$".format(p,sig_vals[p][0],sig_units[p]))
        sig_captions.insert(-1,"and")
        sigs = []
        varyAll(0,list(sig_vals.items()),[],sigs)
        sig_names = OrderedDict([(shortname(point), '$\\{} = {}{}$'.format(varied,point[varied_ind],sig_units[varied])) for point in sigs])

    # todo: make this work for .json as well
    def add_all_cutflows(cats,dir,subdirs,wildcard):
        collected = {cat:[] for cat in cats}
        def get_col_key(col):
            meta = col.metadata
            if meta['sample_type']=='bkg': return meta['bkg_type']
            elif meta['sample_type']=='sig': return shortname(meta)
            else: return ''
        for subdir in subdirs:
            files = expand_wildcards(dir+subdir+'/'+wildcard)
            cols = [svj.Columns.load(file) for file in files]
            cols = common.filter_pt(cols, 170.)
            cols = [c for c in cols if not (c.metadata.get('bkg_type','')=='wjets' and 'htbin' not in c.metadata)]
            # sort cols into categories
            for col in cols:
                col_key = get_col_key(col)
                if col_key in collected: collected[col_key].append(col)
                elif args.verbose: print(f'Skipping column key {col_key}')
        cutflows = {cat:common.add_cutflows(*cols) for cat,cols in collected.items()}
        return cutflows

    bkg_cutflows = add_all_cutflows(list(bkg_names.keys()), args.dir, ["Summer20UL16","Summer20UL17","Summer20UL18"], '*.npz') if len(bkg_names)>0 else {}
    sig_cutflows = add_all_cutflows(list(sig_names.keys()), args.dir, ["Private3DUL16","Private3DUL17","Private3DUL18"], '*pythia8.npz') if len(sig_names)>0 else {}

    procs = OrderedDict(list(bkg_names.items())+list(sig_names.items()))
    cutflows = {}
    cutflows.update(bkg_cutflows)
    cutflows.update(sig_cutflows)

    multicol = 3 if args.type=='rawrel' and args.error else 2
    if not args.error: multicol = 1
    if args.alignDecimal: multicol *= 2
    max_err = 0
    for proc,procname in procs.items():
        outDict["header1"] += " & " + "\\multicolumn{"+str(multicol)+"}{r}{"+procname+"}"

        cutflow = cutflows[proc]
        # skipping a line essentially combines the efficiency of that line with the next line
        for skipline in args.skiplines:
            cutflow.pop(skipline,None)
        cutflow = compute_cutflow(cutflow)
        first = None
        last = None
        for key,val in cutflow.items():
            if (args.efficiency and key=='raw') or (not args.efficiency and key=='stitch'):
                started = True
            elif started:
                otmp = " & \\colspace"+print_val(val[args.type[:3]],args.prec,args.minprec) # +(" & "+splitline[cutflow_ind+2] if args.error else "")
                if args.alignDecimal: otmp = otmp.replace(".", "&.", 1)
                outDict[key] += otmp
                #max_err = max(max_err, splitline[cutflow_ind+2])
                if args.type=='rawrel': outDict[key] += " & "+print_val(val['rel'],args.prec,args.minprec)
                if first is None: first = float(val['raw'])
                last = float(val['raw'])
        if args.efficiency:
            print(procname,last/first*100)
            outDict["efficiency"] += " & \\multicolumn{"+str(multicol)+"}{"+("l" if args.alignDecimal else "r")+"}{"+("\\colspace" if args.alignDecimal else "")+"{:.2g}".format(last/first*100)+"}"

    wfile = open(args.outname,'w')

    # caption
    captions = {
        'raw': "Expected number of events for {:.1f}\\fbinv".format(sum([common.lumis[y] for y in ["2016","2017","2018"]])/1000.), # convert pbinv to fbinv
        'abs': "Absolute cumulative efficiencies in \%",
        'rel': "Relative efficiencies in \%",
    }
    captions['rawrel'] = captions['raw']+" (relative efficiency in \%)"
    staterr_caption = " Only statistical uncertainties are shown."
    maxerr_caption = " Statistical uncertainties, at most {}\%, are omitted.".format(max_err)
    eff_caption = " The line ``Efficiency [\%]'' is the absolute efficiency after the final selection."
    if args.procs=="bkg" or args.procs=="all":
        caption = "\\{}{{{} for each step of the event selection process for the major background processes{}.{}{}}}".format(
            "topcaption" if args.topcapt else "caption",
            captions[args.type],
            " and benchmark signal model ($\\mz = 350\\GeV$, $\\mdark = 10\\GeV$, $\\rinv = 0.3$)" if args.procs=="all" else "",
            staterr_caption if args.error else maxerr_caption if args.summarizeerror else "",
            eff_caption if args.efficiency else "",
        )
    else:
        caption = "\\{}{{{} for each step of the event selection process for signals with {}.{}{}}}".format(
            "topcaption" if args.topcapt else "caption",
            captions[args.type],
            ', '.join(sig_captions).replace("and,","and"),
            staterr_caption if args.error else maxerr_caption if args.summarizeerror else "",
            eff_caption if args.efficiency else "",
        )

    # preamble
    coltype = "S"
    if args.type=="rawrel":
        if args.error: coltype = "SP"
        else: coltype = "RP"
    elif args.type=="rel":
        if not args.error: coltype = "r"
    if args.alignDecimal: coltype = coltype.replace("r","A")
    wfile.write(
        '\n'.join([
            r'\begin{table}[htb]',
            r'\centering',
            caption,
            r'\cmsTable{'
            r'\begin{tabular}{M'+coltype*len(procs)+'}',
            r'\hline',
        ])
    )

    for key,val in outDict.items():
        if key=='stitch' and not 'raw' in args.type: continue
        wfile.write(val+" \\\\"+"\n")
        if key=="header1" or (args.efficiency and key==list(namesDict)[-1]): wfile.write("\\hline\n")

    # postamble
    wfile.write(
        '\n'.join([
            r'\hline',
            r'\end{tabular}',
            r'}',
            r'\label{tab:sel-eff-'+args.outname.replace('.tex',''),
            r'\end{table}',
        ])
    )

    wfile.close()
