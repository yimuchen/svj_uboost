import os, os.path as osp, logging, re, time, json, argparse, sys, math, shutil, subprocess
import matplotlib.pyplot as plt
from collections import OrderedDict
from contextlib import contextmanager
import svj_ntuple_processing as svj
from scipy.ndimage import gaussian_filter
import requests
import numpy as np
from datetime import datetime
import json
from cycler import cycler

np.random.seed(1001)


# Where training data will be stored
DATADIR = osp.join(osp.dirname(osp.abspath(__file__)), 'data')


def setup_logger(name: str = "bdt") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info("Logger %s is already defined", name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                + " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def debug(flag: bool = True) -> None:
    """Convenience switch to set the logging level to DEBUG.

    Args:
        flag (bool, optional): If true, set the logging level to DEBUG. Otherwise, set
            it to INFO. Defaults to True.
    """
    if flag:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


cms_style = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    #
    "mathtext.fontset": "custom",
    "mathtext.rm": "helvetica",
    "mathtext.bf": "helvetica:bold",
    "mathtext.sf": "helvetica",
    "mathtext.it": "helvetica:italic",
    "mathtext.tt": "helvetica",
    "mathtext.cal": "helvetica",
    #
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "axes.prop_cycle": cycler("color", ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2" , "#832db6" , "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd" ]),
    "savefig.transparent": False,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    'text.usetex' : True,
    }

def set_mpl_fontsize(small=16, medium=22, large=26):
    """Sets matplotlib font sizes to sensible defaults.

    Args:
        small (int, optional): Font size for text, axis titles, and ticks. Defaults to
            18.
        medium (int, optional): Font size for axis labels. Defaults to 22.
        large (int, optional): Font size for figure title. Defaults to 26.
    """
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title
    from matplotlib.pyplot import style as plt_style
    plt_style.use(cms_style)
    plt.rc('text', usetex=True)
    plt.rc(
        'text.latex',
        preamble=(
            r'\usepackage{helvet} '
            r'\usepackage{sansmath} '
            r'\sansmath '
            )
        )

def put_on_cmslabel(ax, text='Simulation Preliminary', year=2018):
    fontsize = 27
    ax.text(
        .0, 1.005,
        r'\textbf{CMS}\,\fontsize{21pt}{3em}\selectfont{}{\textit{'+text+'}}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=fontsize
        )
    # assume float means lumi, convert to fb-1
    if isinstance(year,float):
        year = "{:.1f} ".format(year/1000)
        year = year+r"$\mathrm{fb}^{-1}$"
    ax.text(
        1.0, 1.005,
        '{} (13 TeV)'.format(year),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=int(19./23. * fontsize)
        )

import matplotlib.pyplot as plt
set_mpl_fontsize()



def pull_arg(*args, **kwargs) -> argparse.Namespace:
    """Reads a specific argument out of sys.argv, and then deletes that argument from
    sys.argv.

    This useful to build very adaptive command line options to scripts. It does
    sacrifice auto documentation of the command line options though.

    Returns:
        argparse.Namespace: Namespace object for only the specific argument.
    """

    """
    Reads a specific argument out of sys.argv, and then
    deletes that argument from sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args


@contextmanager
def timeit(msg):
    """
    Prints duration of a block of code in the terminal.
    """
    try:
        logger.info(msg)
        sys.stdout.flush()
        t0 = time.time()
        yield None
    finally:
        t1 = time.time()
        logger.info(f"Done {msg[0].lower() + msg[1:]}, took {t1-t0:.2f} secs")


time_and_log = timeit # backwards compatibility


def imgcat(path) -> None:
    """
    Only useful if you're using iTerm with imgcat on the $PATH:
    Display the image in the terminal.
    """
    if shutil.which('imgcat'):
        os.system('imgcat ' + path)


def expand_wildcards(pats):
    import seutils
    import glob
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


class Scripter:
    """
    Command line utility.

    When an instance of this class is used as a contextwrapper on a function, that
    function will be considered a 'command'.

    When Scripter.run() is called, the script name is pulled from the command line, and
    the corresponding function is executed.

    Example:

        In file test.py:
        >>> scripter = Scripter()
        >>> @scripter
        >>> def my_func():
        >>>     print('Hello world!')
        >>> scripter.run()

        On the command line, the following would print 'Hello world!':
        $ python test.py my_func
    """

    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        """
        Stores a command line script with its name as the key.
        """
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg("script", choices=list(self.scripts.keys())).script
        logger.info(f"Running {script}")
        self.scripts[script]()


@contextmanager
def quick_ax(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Axes.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        yield ax
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass


@contextmanager
def quick_fig(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        yield fig
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass

@contextmanager
def quick_subplots(*args, **kwargs):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    outfile = kwargs.pop('outfile', 'tmp.png')
    try:
        fig, axes = plt.subplots(*args, **kwargs)
        yield fig, axes
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass


#__________________________________________________
# Automatic cross section getter

class Record(dict):
    @property
    def xs(self):
        return self['crosssection']['xs_13tev']

    @property
    def br(self):
        try:
            return self['branchingratio']['br_13tev']
        except KeyError:
            return 1.

    @property
    def kfactor(self):
        if 'kfactor' in self:
            for key, val in self['kfactor'].items():
                if key.startswith('kfactor_'):
                    return val
        return 1.

    @property
    def effxs(self):
        return self.xs*self.br*self.kfactor


def load_treemaker_crosssection_txt():
    """
    Downloads the cross section file from the TreeMaker repository and returns
    the contents. If the file has been previously downloaded it is not re-downloaded.
    """
    import requests
    cache = '/tmp/treemaker_xs.txt'
    if not osp.isfile('/tmp/treemaker_xs.txt'):
        url = 'https://raw.githubusercontent.com/TreeMaker/TreeMaker/Run2_UL/WeightProducer/python/MCSampleValues.py'
        text = requests.get(url).text
        with open(cache, 'w') as f:
            text = text.lower()
            f.write(text)
            return text
    else:
        with open(cache) as f:
            return f.read()


def get_record(key):
    """
    Looks for the sample key (e.g. "QCD_Pt_1400to1800") in the cross section
    file from TreeMaker
    """
    text = load_treemaker_crosssection_txt()
    match = re.search('"'+key+'"' + r' : ({[\w\W]*?})', text, re.MULTILINE)
    if not match: raise Exception(f'Could not find record for {key}')
    # Turn it into a dict of dicts
    record_txt = (
        match.group(1)
        .replace('xsvalues', 'dict')
        .replace('brvalues', 'dict')
        .replace('kfactorvalues', 'dict')
        )
    return Record(eval(record_txt))


def mt_wind(cols, mt_high, mt_low):
    mt_cut = (cols.arrays['mt']>mt_low) & (cols.arrays['mt']<mt_high)
    return mt_cut

def filter_pt(cols, min_pt):
    """
    Filters for a minimum pt (only valid for QCD).
    Does not filter out non-QCD backgrounds or signals.
    """
    filtered = []
    for c in cols:
        if c.metadata.get('ptbin', [1e6])[0] < min_pt:
            continue
        filtered.append(c)
    return filtered

def filter_ht(cols, min_ht, bkg_type=None):
    """
    Filters for a minimum pt (only valid for ttjets/wjets/zjets).
    Does not filter out QCD or signal.
    If bkg_type is None, it filters ttjets AND wjets AND zjets.
    """
    filtered = []
    for c in cols:
        if bkg_type and c.metadata.get('bkg_type', None) != bkg_type:
            filtered.append(c)
            continue
        if c.metadata.get('htbin',[1e6, 1e6])[0] < min_ht:
            continue
        filtered.append(c)
    return filtered


#__________________________________________________
# Histogram classes

# in units of pb-1 (xsec units: pb)
lumis = {
    "2016": 36330,
    "2017": 41530,
    "2018PRE": 21090,
    "2018POST": 38650,
}
lumis["2018"] = lumis["2018PRE"]+lumis["2018POST"] # 59740
lumis["RUN2"] = lumis["2016"]+lumis["2017"]+lumis["2018"] # 137600

# from madgraph (BR to dark included), MADPT>300, jet matching efficiency included, for gq = 0.25
# w/ Z-like k-factor 1.23
signal_xsecs = {
    200 : 9.143,
    250 : 6.910,
    300 : 5.279,
    350 : 4.077,
    400 : 3.073,
    450 : 2.448,
    500 : 1.924,
    550 : 1.578,
}

def get_event_weight(obj,lumi=None):
    if isinstance(obj,svj.Columns):
        if lumi is None:
            lumi = lumis[str(obj.metadata['year'])]

        if obj.metadata["sample_type"]=="sig":
            mz = obj.metadata["mz"]
            if mz in signal_xsecs:
                xsec = signal_xsecs[mz]
            else:
                # uses interpolation
                xsec = central.xs
            nevents = obj.cutflow['raw']
            event_weight = lumi*xsec/nevents
            logger.info(f'Event weight: {lumi}*{xsec}/{nevents} = {event_weight}')
            return event_weight
        elif obj.metadata["sample_type"]=="bkg":
            tree_weights = obj.to_numpy(['weight']).ravel()
            if len(tree_weights)>0: logger.info(f'Event weight: {lumi}*{tree_weights[0]} = {lumi*tree_weights[0]}')
            return lumi*tree_weights
        else: # data
            return 1.0

    elif isinstance(obj,Histogram):
        return obj.metadata.get('event_weight',1)

    else:
        raise RuntimeError(f'Unknown weight method for object of class {type(obj).__name__}')

def get_single_event_weight(weights):
    if isinstance(weights,float) or isinstance(weights,int): return weights
    elif len(weights)>0: return weights[0]
    else: return 1.0

def add_cutflows(*objs):
    # add cutflows if present, accounting for weights
    keys = [list(obj.cutflow.keys()) if hasattr(obj,'cutflow') else [] for obj in objs]
    if keys and all(keys) and all(k == keys[0] for k in keys):
        weights = [get_single_event_weight(get_event_weight(obj)) for obj in objs]
        return OrderedDict((k, sum(obj.cutflow[k]*weight for obj,weight in zip(objs,weights))) for k in keys[0])
    else:
        if keys and all(keys):
            keylist = '\n'.join(f'  {k}' for k in keys)
            logger.warning(f'Unable to add cutflows with different keys:\n{keylist}')
        return OrderedDict()

class Histogram:
    """
    Histogram container class.

    Keeps track of binning, values, errors, and metadata.
    Designed to be easily JSON-serializable.
    """
    @classmethod
    def from_dict(cls, dict):
        inst = cls.__new__(cls)
        inst.binning = np.array(dict['binning'])
        inst.vals = np.array(dict['vals'])
        inst.errs = np.array(dict['errs'])
        inst.metadata = dict['metadata'].copy()
        if 'cutflow_keys' and 'cutflow_vals' in dict:
            inst.cutflow = OrderedDict((k,v) for k,v in zip(dict['cutflow_keys'],dict['cutflow_vals']))
        return inst

    def __init__(self, binning, vals=None, errs=None):
        self.binning = binning
        self.vals = np.zeros(self.nbins) if vals is None else vals
        self.errs = np.sqrt(self.vals) if errs is None else errs
        self.metadata = {}
        self.cutflow = OrderedDict()

    @property
    def nbins(self):
        return len(self.binning)-1

    def json(self):
        # Convert anything that remotely looks like a float to python float.
        for k, v in self.metadata.items():
            try:
                self.metadata[k] = float(v)
            except (ValueError,TypeError) as e:
                pass
        return dict(
            type = 'Histogram',
            binning = list(self.binning),
            vals = list(self.vals),
            errs = list(self.errs),
            metadata = self.metadata.copy(),
            cutflow_keys = list(self.cutflow.keys()),
            cutflow_vals = [v.item() if type(v).__module__=='numpy' else v for v in self.cutflow.values()], # convert from np int64 to serializable type
            )

    def __repr__(self):
        d = np.column_stack((self.vals, self.errs))
        return (
            f'<H n={self.nbins} int={self.vals.sum():.3f}'
            f' binning={self.binning[0]:.1f}-{self.binning[-1]:.1f}'
            f' vals/errs=\n{d}'
            '>'
            )

    def copy(self):
        the_copy = Histogram(self.binning.copy(), self.vals.copy(), self.errs.copy())
        the_copy.metadata = self.metadata.copy()
        the_copy.cutflow = self.cutflow.copy()
        return the_copy

    def __add__(self, other):
        """Add another Histogram or a numpy array to this histogram. Returns new object."""
        ans = self.copy()

        if isinstance(other, Histogram):
            ans.vals = self.vals + other.vals
            ans.errs = np.sqrt(self.errs**2 + other.errs**2)
            ans.cutflow = add_cutflows(self, other)

        elif hasattr(other, 'shape') and self.vals.shape == other.shape:
            # Add a simple np histogram on top of it
            ans.vals += other
            ans.errs = np.sqrt(self.errs**2 + other)

        return ans

    def __radd__(self, other):
        if other == 0:
            return self.copy()
        raise NotImplementedError

    def __mul__(self, factor):
        """Multiply by a constant"""
        ans = self.copy()
        if isinstance(factor, (int, float)):
            ans.vals = factor*ans.vals
            ans.errs = factor*ans.errs
        else:
            raise NotImplemented
        return ans

    @property
    def norm(self):
        return self.vals.sum()

    def rebin(self, n=2):
        """
        Merge n bins together to make a coarser histogram.
        Mostly useful for plotting.
        """
        if n==1: return self.copy()
        n_bins_new = math.ceil(self.nbins / float(n))

        binning_new = self.binning[::n]
        if binning_new[-1] != self.binning[-1]:
            binning_new = np.append(binning_new, self.binning[-1])

        # Build a map from old binning to new binning
        map = np.repeat(np.arange(n_bins_new), n)[:self.nbins]

        values_new = np.zeros(n_bins_new)
        np.add.at(values_new, map, self.vals)

        errs_new = np.zeros(n_bins_new)
        np.add.at(errs_new, map, self.errs**2)
        errs_new = np.sqrt(errs_new)

        h = Histogram(binning_new, values_new, errs_new)
        h.metadata = self.metadata.copy()
        return h

    def cut(self, xmin=-np.inf, xmax=np.inf):
        """
        Throws away all bins with left boundary < xmin or right boundary > xmax.
        Mostly useful for plotting purposes.
        Returns a copy.
        """
        # safety checks
        if xmin>xmax:
            raise ValueError("xmin ({}) greater than xmax ({})".format(xmin,xmax))

        h = self.copy()
        imin = np.argmin(self.binning < xmin) if xmin > self.binning[0] else 0
        imax = np.argmax(self.binning > xmax) if xmax < self.binning[-1] else self.nbins+1
        h.binning = h.binning[imin:imax]
        h.vals = h.vals[imin:imax-1]
        h.errs = h.errs[imin:imax-1]
        return h

def _create_binning(binw, left, right):
    bins = left + binw * np.arange(math.ceil((right-left)/binw)+1)
    # Force casting to python floats, as numpy values causes issues with JSON serialization
    return [float(x) for x in bins]

class HistoBins(type):
    @property
    def bins(cls):
        if cls._bins is None:
            cls._bins = _create_binning(*cls.default_binning)
        return cls._bins
    @bins.setter
    def bins(cls, val):
        cls._bins = val
    @property
    def non_standard_binning(cls):
        return cls._non_standard_binning
    @non_standard_binning.setter
    def non_standard_binning(cls, val):
        cls._non_standard_binning = val

class VarArrHistogram(Histogram, metaclass=HistoBins):
    """
    Wrapper around histogram that initializes the value and weight arrays.
    It will also carry around the current and default binning information
    """
    default_binning = (10,0,100) # Width, min, max
    _bins = None # List of bin values
    _non_standard_binning = False
    name = '' # Name of variable to use

    @property
    def bins(self):
        return self.__class__.bins

    @bins.setter
    def bins(self, val):
        self.__class__.bins = val

    @property
    def non_standard_binning(self):
        return self.__class__.non_standard_binning

    @non_standard_binning.setter
    def non_standard_binning(self, val):
        self.__class__.non_standard_binning = val

    @classmethod
    def default_binw(cls)->float:
        return cls.default_binning[0]

    @classmethod
    def default_binmin(cls)->float:
        return cls.default_binning[1]

    @classmethod
    def default_binmax(cls)->float:
        return cls.default_binning[2]

    @classmethod
    def create_binning(cls, binw, left, right):
        return _create_binning(binw, left, right)

    @classmethod
    def empty(cls):
        return Histogram(cls.bins)

    def __init__(self, cols, weights=None):
        var_arr = self._create_var_array(cols)
        vals = np.histogram(var_arr, self.bins, weights=weights)[0].astype(float)
        weights2 = weights if weights is None else weights **2
        errs = np.sqrt(np.histogram(var_arr, self.bins, weights=weights2)[0].astype(float))
        super().__init__(self.bins, vals, errs)

    @classmethod
    def _create_var_array(cls,cols):
        """Method for creating the values array used to fill the histogram"""
        return cols.to_numpy([cls.name]).flatten()



# List of variables with defined binning
class MTHistogram(VarArrHistogram):
    name='mt'
    default_binning = (10, 180, 650)

class ECFN2B2Histogram(VarArrHistogram):
    name = 'ecfn2b2'
    default_binning = (0.01, 0, 0.5)

class ECFM2B1Histogram(VarArrHistogram):
    name = 'ecfm2b1'
    default_binning = (0.005,0,0.2)

class RTHistogram(VarArrHistogram):
    name = 'rt'
    default_binning = (0.03, 1.0, 2.5)

class METDPhiHistogram(VarArrHistogram):
    name = 'metdphi'
    default_binning = (0.08, 0, 3.2)

registered_varhists = { subcl.name: subcl for subcl in VarArrHistogram.__subclasses__() }



class Encoder(json.JSONEncoder):
    """
    Standard JSON encoder, but support for the Histogram class
    """
    def default(self, obj):
        if isinstance(obj, Histogram):
            return obj.json()
        return super().default(obj)


class Decoder(json.JSONDecoder):
    """
    Standard JSON decoder, but support for the Histogram class
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        try:
            is_histogram = d['type'] == 'Histogram'
        except (AttributeError, KeyError):
            is_histogram = False
        if is_histogram:
            return Histogram.from_dict(d)
        return d

#__________________________________________________
# Data pipeline

class Columns(svj.Columns):
    """
    Data structure that contains all the training data (features)
    and information about the sample.

    See: https://github.com/boostedsvj/svj_ntuple_processing/blob/main/svj_ntuple_processing/__init__.py#L357
    """
    @classmethod
    def load(cls, *args, **kwargs):
        inst = super().load(*args, **kwargs)
        # Transforming bytes keys to str keys
        old_cf = inst.cutflow
        inst.cutflow = OrderedDict()
        for key in old_cf.keys():
            if isinstance(key, bytes):
                inst.cutflow[key.decode()] = old_cf[key]
            else:
                inst.cutflow[key] = old_cf[key]
        return inst

    def __repr__(self):
        return (
            '<Column '
            + ' '.join([f'{k}={v}' for k, v in self.metadata.items() if k!='src'])
            + '>'
            )

    @property
    def key(self):
        return (
            osp.basename(self.metadata['src'])
            .replace('.npz', '')
            ).split('_TuneCP5_13TeV')[0].lower()

    @property
    def record(self):
        if not hasattr(self, '_record'):
            self._record = get_record(self.key)
        return self._record

    @property
    def presel_eff(self):
        if self.cutflow['raw']==0: return 0.
        return self.cutflow['preselection'] / self.cutflow['raw']

    @property
    def xs(self):
        if hasattr(self, 'manual_xs'):
            # Only for setting a manual cross section
            return self.manual_xs
        elif 'bkg_type' in self.metadata:
            return self.record.effxs
        else:
            if not hasattr(self, '_signal_xs_fit'):
                self._signal_xs_fit = np.poly1d(
                    requests
                    .get('https://raw.githubusercontent.com/boostedsvj/svj_madpt_crosssection/main/fit_madpt300.txt')
                    .json()
                    )
            return self._signal_xs_fit(self.metadata['mz'])

    @property
    def effxs(self):
        return self.xs * self.presel_eff

    @property
    def weight_per_event(self):
        return self.effxs / len(self)


def columns_to_numpy(
    signal_cols, bkg_cols, features,
    downsample=.4, weight_key='weight',
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    """
    X = []
    y = []
    bkg_weight = []
    signal_weight = []

    logger.info(f'Downsampling bkg, keeping fraction of {downsample}')
    # Get the features for the bkg samples
    for cols in bkg_cols:
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]
        this_weight = cols.arrays[weight_key][mtwind]
        if downsample < 1.:
            #select = np.random.choice(len(cols), int(downsample*len(cols)), replace=False)
            select = np.random.choice(len(this_weight), int(downsample*len(this_weight)), replace=False)
            this_X = this_X[select]
            this_weight = this_weight[select]
        X.append(this_X)
        bkg_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # Get the features for the signal samples
    for cols in signal_cols:
        sigmtwind = mt_wind(cols, mt_high, mt_low)
        X.append(cols.to_numpy(features)[sigmtwind])
        #print(features)
        len_sig_cols=len(cols.arrays[features[0]][sigmtwind])
        #print(cols.to_numpy(features)[sigmtwind])
        #print(len(cols.to_numpy(features)[sigmtwind]))
        #length_of_signalCol=len(cols.arrays(features)[mtwind])
        #print(length_of_signalCol, len(cols))
        y.append(np.ones(len_sig_cols))
        # All signal model parameter variations should get equal weight,
        # but some signal samples have more events.
        # Use 1/n_events as a weight per event.
        signal_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))

    bkg_weight = np.concatenate(bkg_weight)
    signal_weight = np.concatenate(signal_weight)
    # Set total signal weight equal to total bkg weight
    signal_weight *= np.sum(bkg_weight) / np.sum(signal_weight)
    weight = np.concatenate((bkg_weight, signal_weight))

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, weight


def add_key_value_to_json(json_file, key, value):
    with open(json_file, 'r') as f:
        json_str = f.read()
    json_str = json_str.rstrip()
    json_str = json_str[:-1] # Strip off the last }
    json_str += f',"{key}":{json.dumps(value)}}}'
    with open(json_file, 'w') as f:
        f.write(json_str)
    logger.info(f'Added "{key}":{json.dumps(value)} to {json_file}')


def add_manual_weight_column(signal_cols, bkg_cols):
    """
    Adds the manual weight calculation as a column
    """
    total_bkg_weight = 0
    for c in bkg_cols:
        c.arrays['manualweight'] = np.ones(len(c)) * c.weight_per_event
        total_bkg_weight += c.arrays['manualweight'].sum()

    # Set signal weights scaled relatively to one another (but not yet w.r.t. bkg)
    total_signal_weight = 0
    for c in signal_cols:
        c.arrays['manualweight'] = np.ones(len(c)) / len(c)
        total_signal_weight += c.arrays['manualweight'].sum()

    # Scale signal weights correctly w.r.t. bkg
    for c in signal_cols:
        c.arrays['manualweight'] *= total_bkg_weight / total_signal_weight


def read_training_features(model_file):
    """
    Reads the features used to train a model from a .json file.
    Only works for xgboost-trained models.
    """
    with open(model_file, 'rb') as f:
        model = json.load(f)
        return model['features']

def rhoddt_windowcuts(mt, pt, rho):
    '''
    Basically a tool to constantly check the kinematics during the DDT processes
    '''
    cuts = (mt>100) & (mt<1000) & (pt>110) & (pt<1500)
    return cuts

def weighted_percentile(x, w, percent):
    """
    Returning the percentile while properly handling event weights.
    """
    # Sorting the input entries according to the values
    sorted_x = x[np.argsort(x)]
    sorted_w = w[np.argsort(x)]
    # Generating the cumulative sum
    sorted_c = np.cumsum(sorted_w)
    # Scaling the cumulative sum so that the actual data points are centered at half the weight
    sorted_c = ((sorted_c - (sorted_w / 2)) / sorted_c[-1]) * 100

    # Special case handling, extreme percent values
    if percent > sorted_c[-1]:
        return sorted_x[-1]
    if percent < sorted_c[0]:
        return sorted_x[0]

    # Getting the points where the cumulative sum pass above or below the percentile of interest
    lower_val = np.max(sorted_x[sorted_c <= percent])
    upper_val = np.min(sorted_x[sorted_c >= percent])

    lower_sum = np.max(sorted_c[sorted_c <= percent])
    upper_sum = np.min(sorted_c[sorted_c >= percent])

    # Getting the interpolation weight if the percentile does not land on a single number
    int_w = 0.5 if upper_sum == lower_sum else (percent - lower_sum)/(upper_sum - lower_sum)

    return upper_val *int_w + lower_val * (1-int_w)

def varmap(mt, pt, rho, var, weight, percent, cut_val):
    '''
    2D map for DDT, now defined in (mt, pt) space instead of (rho, pt).
    Decorrelates the tagger variable with respect to mt using a 2D (mt, pt) map.
    '''

    # Apply the rho-ddt window cuts to the data (still useful for pt/mt range)
    cuts = rhoddt_windowcuts(mt, pt, rho)
    mt_pt = mt/pt

    # Create a 2D histogram of mt and pt, weighted by event weights
    C, MT_PT_edges, PT_edges = np.histogram2d(mt_pt[cuts], pt[cuts], bins=73, weights=weight[cuts])

    # Initialize the variable map
    w, h = 74, 74
    VAR_map = [[0 for x in range(h)] for y in range(w)]

    # Get data arrays for events passing the cuts
    VAR = var[cuts]
    WEIGHT = weight[cuts]
    mt_pt_bin_idx = np.digitize(mt_pt[cuts], MT_PT_edges) - 1
    pt_bin_idx = np.digitize(pt[cuts], PT_edges) - 1

    # Loop over mt and pt bins
    for i in range(len(MT_PT_edges)-1):
        for j in range(len(PT_edges)-1):
            CUT = (mt_pt_bin_idx == i) & (pt_bin_idx == j)

            # If there is data in this bin, calculate the percentile of the variable
            if len(VAR[CUT]) > 0:
                VAR_map[i][j] = weighted_percentile(VAR[CUT], WEIGHT[CUT], 100 - percent)

    # Apply smoothing (you can replace this with adaptive smoothing if you like)
    VAR_map_smooth = gaussian_filter(VAR_map, sigma=1.0)

    # Return smoothed map and the new mt and pt bin edges
    return VAR_map_smooth, MT_PT_edges, PT_edges

    # Return the smoothed variable map, along with the rho and pt edges
    return VAR_map_smooth, MT_PT_edges, PT_edges

# Class that converts numpy arrays into list so they can be easily stored in json files
class NumpyArrayEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)

def create_DDT_map_dict(mt, pt, rho, var, weight, percents, cut_vals, ddt_name):
    '''
    Creates a dictionary of DDT 2D maps for a range of cut_vals and background efficiencies (percents).
    Each DDT map is a 2D array of tagger thresholds binned in (mt, pt) space,
    smoothed and stored along with the corresponding bin edges.

    Inputs:
    - mt, pt
    - var: the tagger variable (e.g. BDT score)
    - weight: event weights
    - percents: list of background efficiencies (e.g. [10, 20, 30])
    - cut_vals: corresponding working point names (e.g. [0.1, 0.2, 0.3])
    - ddt_name: output filename for the JSON dictionary
    '''

    # Apply kinematic window cuts (still uses rho for now)
    cuts = rhoddt_windowcuts(mt, pt, rho)

    var_dict = {}

    for cut_val, percent in zip(cut_vals, percents):
        print(f"Creating DDT 2D map for cut value {cut_val}, efficiency {percent}%")

        # Build the DDT map in (mt, pt) space
        var_map_smooth, MT_PT_edges, PT_edges = varmap(mt, pt, rho, var, weight, percent, cut_val)

        # Store the map and bin edges
        var_dict[str(cut_val)] = (var_map_smooth.tolist(), MT_PT_edges.tolist(), PT_edges.tolist())

    # Save to file
    if ddt_name is None:
        ddt_name = 'ddt_' + str(var) + '_' + datetime.now().strftime('%Y%m%d') + '.json'
    with open(ddt_name, 'w') as f:
        json.dump(var_dict, f, cls=NumpyArrayEncoder)

def calculate_varDDT(mt, pt, rho, var, weight, cut_val, ddt_name):
    '''
    Applies a DDT transformation to 'var' using a DDT map in (mt, pt) space.
    
    Inputs:
    - mt, pt
    - var: tagger variable (e.g. BDT score)
    - weight: event weights (not used here, but passed for compatibility)
    - cut_val: key in the DDT map dict to use (e.g. 0.1, 0.2)
    - ddt_name: path to JSON file containing the DDT map

    Returns:
    - varDDT: array of DDT-transformed tagger values
    '''

    if not osp.exists(ddt_name):
        raise FileNotFoundError(f"The file {ddt_name} does not exist.")

    with open(ddt_name, 'r') as f:
        var_dict = json.load(f)

    if str(cut_val) not in var_dict:
        raise KeyError(f"The key {cut_val} does not exist in the dictionary.")

    # Get the smoothed map and bin edges
    var_map_smooth, MT_PT_edges, PT_edges = var_dict[str(cut_val)]
    var_map_smooth = np.array(var_map_smooth)
    MT_PT_edges = np.array(MT_PT_edges)
    PT_edges = np.array(PT_edges)

    # Apply DDT window cuts (you can update this function later if you drop rho)
    cuts = rhoddt_windowcuts(mt, pt, rho)

    # Bin index lookup with digitize
    pt_bin = np.clip(np.digitize(pt, PT_edges) - 1, 0, len(PT_edges) - 1)
    mt_pt_bin = np.clip(np.digitize(mt/pt, MT_PT_edges) - 1, 0, len(MT_PT_edges) - 1)

    # Apply DDT: var - DDT map value at the appropriate bin
    varDDT = var - var_map_smooth[mt_pt_bin, pt_bin]

    return varDDT

def apply_hemveto(cols):
    cols = cols.select(svj.veto_HEM(cols.arrays['ak4_subl_eta'],cols.arrays['ak4_subl_phi'],cols.arrays['ak4_subl_pt']))
    cols.cutflow['hem_veto'] = len(cols)
    return cols

def apply_rt_signalregion(cols):
    cols = cols.select(cols.arrays['rt'] > 1.18)
    cols.cutflow['rt_signalregion'] = len(cols)
    return cols

def check_if_model_exists(model_file, xrootd_url) :
    # Check if the file exists locally
    if not os.path.exists(model_file):
        print(f"File {model_file} not found. Downloading from {xrootd_url}...")
        try:
            os.makedirs(os.path.dirname(model_file), exist_ok=True)  # Ensure directory exists
            subprocess.run(["xrdcp", xrootd_url+model_file, model_file], check=True)
            print(f"Downloaded {model_file} successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error downloading {model_file}: {e}")
            return None

def cutbased_ddt(cols, lumi, ddt_map_file, xrootd_url, cut_val = 0.12):
    check_if_model_exists(ddt_map_file, xrootd_url)

    # Get features necessary to apply the DDT
    mT = cols.to_numpy(['mt']).ravel() # make one d ... don't ask why it's not
    pT = cols.to_numpy(['pt']).ravel()
    rho = cols.to_numpy(['rho']).ravel()
    ecfm2b1 = cols.to_numpy(['ecfm2b1']).ravel()
    weight = get_event_weight(cols, lumi)

    ddt_val = calculate_varDDT(mT, pT, rho, ecfm2b1, weight, cut_val, ddt_map_file)
    return ddt_val

def apply_cutbased(cols):
    cols = apply_rt_signalregion(cols)
    cols = cols.select(cols.arrays['ecfm2b1'] > 0.09)
    cols.cutflow['cutbased'] = len(cols)
    return cols

def apply_cutbased_ddt(cols, lumi, ddt_map_file = 'models/cutbased_ddt_map_ANv6.json', xrootd_url = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/cutbased_ddt/') :
    cols = apply_rt_signalregion(cols)
    ddt_val = cutbased_ddt(cols, lumi, ddt_map_file, xrootd_url)

    # Now cut on the DDT above 0.0 (referring to above the ecfm2b1 cut value)
    cols = cols.select(ddt_val > 0.0) # mask for the selection
    cols.cutflow['cutbased_ddt'] = len(cols)
    return cols

def apply_cutbasedCR(cols):
    cols = apply_rt_signalregion(cols)
    cols = cols.select(cols.arrays['ecfm2b1'] < 0.032)
    cols.cutflow['cutbasedCR'] = len(cols)
    return cols

def apply_cutbasedCRloose(cols):
    cols = cols.select(cols.arrays['ecfm2b1'] < 0.032)
    cols.cutflow['cutbasedCRloose'] = len(cols)
    return cols

def apply_anticutbased_ddt(cols, lumi, ddt_map_file = 'models/cutbased_ddt_map_ANv6.json', xrootd_url = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/cutbased_ddt/') :

    cols = apply_rt_signalregion(cols)
    ddt_val = cutbased_ddt(cols, lumi, ddt_map_file, xrootd_url)

    # Now cut on the DDT BELOW 0.0 (referring to above the ecfm2b1 cut value)
    cols = cols.select(ddt_val < 0.0) # mask for the selection
    cols.cutflow['anticutbased_ddt'] = len(cols)
    return cols

def apply_anticutbased(cols):
    cols = apply_rt_signalregion(cols)
    cols = cols.select(cols.arrays['ecfm2b1'] < 0.09)
    cols.cutflow['anticutbased'] = len(cols)
    return cols

def apply_antiloosecutbased_ddt(cols, lumi, ddt_map_file = 'models/cutbased_ddt_map_ANv6.json', xrootd_url = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/cutbased_ddt/') :

    ddt_val = cutbased_ddt(cols, lumi, ddt_map_file, xrootd_url)

    # Now cut on the DDT BELOW 0.0 (referring to above the ecfm2b1 cut value)
    cols = cols.select(ddt_val < 0.0) # mask for the selection
    cols.cutflow['anticutbased_ddt'] = len(cols)
    return cols

def apply_antiloosecutbased(cols):
    cols = cols.select(cols.arrays['ecfm2b1'] < 0.09)
    cols.cutflow['antiloosecutbased'] = len(cols)
    return cols

# Relative path to the BDT
# This specific BDT was chosen to be used during the L3 review
bdt_model_file = 'models/svjbdt_obj_rev_version.json'

def split_bdt(sel):
    parts = sel.split('=')
    if len(parts)==2:
        try:
            bdt_cut = float(parts[1])
        except ValueError:
            # Handle the case where the number following 'bdt=' is not valid
            print(f"Invalid number {parts[1]} following 'bdt='.")
    else:
        raise InvalidSelectionException(sel=selection)
    return parts[1]

def calc_bdt_scores(X, model_file=bdt_model_file):
    import xgboost as xgb

    # Load the model and get the predictions
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_file)
    with time_and_log(f'Calculating xgboost scores for {bdt_model_file}...'):
        score = xgb_model.predict_proba(X)[:,1]
    return score

def apply_bdtbased(cols,wp,lumi,anti=False,model_file = bdt_model_file,ddt_map_file = 'models/ddt_AN_v5.json',
                   xrootd_url='root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BDT_based/'):

    check_if_model_exists(ddt_map_file, xrootd_url)
    check_if_model_exists(model_file, xrootd_url)
    cols = apply_rt_signalregion(cols)

    # make sure bdt features match the choosen file
    bdt_features = read_training_features(model_file)

    # Grab the weights and scores
    X = []
    score = {}
    weight = []

    # Get the features for the bkg samples
    X = cols.to_numpy(bdt_features)
    # Calculate bdt scores and event weights
    score = calc_bdt_scores(X)
    weight = get_event_weight(cols, lumi)

    # Apply the DDT
    mT = cols.to_numpy(['mt']).ravel() # make one d ... don't ask why it's not
    pT = cols.to_numpy(['pt']).ravel()
    rho = cols.to_numpy(['rho']).ravel()
    bdt_ddt_score = calculate_varDDT(mT, pT, rho, score, weight, wp, ddt_map_file)

    # Now cut on the DDT above 0.0 (referring to above the given BDT cut value)
    # or < 0.0 for anti-tag CR
    if anti:
        cols = cols.select(bdt_ddt_score < 0.0) # mask for the selection
        cols.cutflow['ddt(antibdt)'] = len(cols)
    else:
        cols = cols.select(bdt_ddt_score > 0.0) # mask for the selection
        cols.cutflow['ddt(bdt)'] = len(cols)
    return cols

class InvalidSelectionException(Exception):
    def __init__(self, msg='Unknown selection {}; choices are "preselection", "cutbased", or "bdt=X.XXX".', sel="", *args, **kwargs):
        msg = msg.format(sel)
        super().__init__(msg, *args, **kwargs)
