# About this repository

This is a lightweight repository that can train a uBoost BDT.
It uses as much training data as it can, by using the precalculated TreeMaker Weight column.


## Setup

```
python3 -m venv env
source env/bin/activate  # Needed every time

pip install pandas
pip install requests
pip install numpy
pip install matplotlib

pip install https://github.com/boostedsvj/svj_ntuple_processing/archive/main.zip
pip install hep_ml
```

Note: `xgboost` needs a full conda environment; if we decide we still want to run trainings using `xgboost` this setup code will change.


## How to run a uBoost training

First download the training data (~2.7 Gb), and split it up into a training and test sample:

```
python download.py
python split_train_test.py
```

This should give you the following directory structure:

```
$ ls data/
bkg  signal  test_bkg  test_signal  train_bkg  train_signal

$ ls data/train_bkg/Summer20UL18/
QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8.npz
QCD_Pt_120to170_TuneCP5_13TeV_pythia8.npz
... <more>
```

Then launch the training script:

```
python training.py
```

It will take quite a few hours for the training to converge (about 12h on my last attempt).
