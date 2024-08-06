# About this repository

This is a repository that can train a uBoost or xgboost BDT for the SVJ boosted analysis.
It uses as much training data as it can, by using the precalculated TreeMaker Weight column.


## Setup

```
conda create -n bdtenv python=3.10
conda activate bdtenv  # Needed every time

conda install xgboost

pip install pandas
pip install requests
pip install numpy
pip install matplotlib
pip install tqdm
pip install numba

pip install git+https://github.com/boostedsvj/jdlfactory
pip install git+https://github.com/boostedsvj/seutils
pip install git+https://github.com/boostedsvj/svj_ntuple_processing
pip install hep_ml

git clone git@github.com:boostedsvj/svj_uboost
```

Alternatively, an editable `svj_ntuple_processing` can be installed for simultaneous developments:
```
git clone git@github.com:boostedsvj/svj_ntuple_processing
pip install -e svj_ntuple_processing/
```

Optional additional packages to read files over xrootd directly instead of making local copies (may not work on all machines):
```
pip install xrootd
pip install fsspec-xrootd
```

## Skims

A minimal setup just to run the skims (avoiding the need for a heavy conda environment):
```
git clone git@github.com:boostedsvj/svj_uboost
cd svj_uboost
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/boostedsvj/jdlfactory
pip install git+https://github.com/boostedsvj/seutils
pip install git+https://github.com/boostedsvj/svj_ntuple_processing
```

The skim code can be tested interactively:
```
python3 skim.py --stageout root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_test [filename]
```

To submit all skim jobs:
```
python3 submit_skim.py --stageout root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_[date] --go
```
(The argument `--keep X` can be included to select a random subset of signal events for statistical studies, where `X` is a float between 0 and 1.)

Then hadd the skims to get one file per sample:
```
python3 hadd_skims.py --stageout root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_[date]_hadd "root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_[date]/*/*"
```

## How to run a training

First download the training data (~4.7 Gb), and split it up into a training and test sample:

```bash
python download.py
python split_train_test.py
```

This should give you the following directory structure:

```bash
$ ls data/
bkg  signal  test_bkg  test_signal  train_bkg  train_signal

$ ls data/train_bkg/Summer20UL18/
QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8.npz
QCD_Pt_120to170_TuneCP5_13TeV_pythia8.npz
... <more>
```

Then launch the training script:

```bash
python training.py xgboost \
    --reweight mt --ref data/train_signal/madpt300_mz350_mdark10_rinv0.3.npz \
    --lr .05 \
    --minchildweight .1 \
    --maxdepth 6 \
    --subsample 1. \
    --nest 400
```

Training with xgboost on the full background should take about 45 min.
The script `hyperparameteroptimization.py` runs this command for various settings in parallel.

### Evaluate

```bash
python evaluate.py
```

The paths to the model are currently hard-coded! Things are still too fluid for a good abstraction.

### Overfitting check: Kolmogorov-Smirnov test

```bash
python overfitting.py models/svjbdt_Nov29_reweight_mt_lr0.05_mcw0.1_maxd6_subs1.0_nest400.json
```

![overfit plot](example_plots/overfit.png)

With p-values close to 1.0, there is no reason to assume any overfitting.

## Cutflow table

```bash
python makeCutflowSVJ.py -o svj.tex -d skims_20240718_hadd -t rawrel -p 0 -k raw preselection 'n_ak4jets>=2' -l '180<mt<650' --compile
```

Creates the cutflow tables in LaTeX format, along with a compiled pdf (`--compile` argument, requires that LaTeX is installed).

## Building Data Cards

One of the key files in this repo is the `build_datacard.py` this is the magic file that makes everything come together. Here, datacards that can be fed into combine are built.

Taking the skims as input (with the consistent preselection applied to all samples), output MT histograms are produced for a specified final selection (either cutbased or bdt=X for a working point X; DDT is applied).
The final selection also includes the HEM veto for the 2018POST era.
For signals, systematic variations are evaluated, and a wider mT range is used to facilitate smoothing of the shapes (using local regression).

```bash
# For the cut based: sig, bkg, data
python3 build_datacard.py build_all_histograms --mtmin 130 --mtmax 700 --binw 10 cutbased "root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20240718_hadd/Private3D*/*pythia8.npz"
python3 build_datacard.py build_all_histograms --binw 10 cutbased "root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20240718_hadd/Summer*/*.npz"
python3 build_datacard.py build_all_histograms --binw 10 cutbased "root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20240718_hadd/Run*/*.npz"
```

After creating the histograms, merge across all data-taking years:
```bash
for CAT in bkg data sig; do
	python3 build_datacard.py merge_histograms cutbased hists_20240718 --cat $CAT
done
```

Smoothing is applied to the merged signal histograms:
```bash
python build_datacard.py smooth_shapes --optimize 1000 --target central --mtmin 180 --mtmax 650 merged/signal.json
```
The output histograms from this step are truncated to the final mT range. (`--target central` means that the optimization of the smoothing span via generalized cross-validation uses the central histogram, and then that optimized span value is applied to the systematic variations.)

These merged, smoothed json files are the inputs to the limit setting. The signal, background, and (optionally) data are supplied separately.
The resulting merged file should use the signal name, the selection type, bin widths, and ranges: `signal_name_cutbased_or_bdt_smooth_with_bkg_binwXY_rangeXYZ-XYZ.json`.

## Extras

An additional function for checking the histogram json files is `ls`. However, this is not the most easy to read it provides a quick way to check for mistakes during file creation.

```bash
python3 build_datacard.py ls signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
# or alternatively, just look at it
head -n 100 signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
```

Then all the up, down, and nominal values can be plotted for the systematics:

```bash
python build_datacard.py plot_systematics signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
```

Similar plots can be made to compare the results of smoothing (e.g. for systematics, between different `keep` percentages, etc.) using the `plot_smooth` function.

A table of systematic uncertainty yield effects can be made as follows:
```bash
python build_datacard.py systematics_table signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
```
Currently, this function only handles one signal model at a time.
It will be expanded to summarize across all signal models once the full scans are available.

And that's it for this part. To use these histograms for fits and limit setting, see the [svj_limits](https://github.com/boostedsvj/svj_limits) repo.
