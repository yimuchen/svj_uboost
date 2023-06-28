import os.path as osp
import common
import svj_ntuple_processing as svj
import xgboost
import numpy as np
import matplotlib.pyplot as plt


scripter = common.Scripter()

# (mur, muf)
mur_muf = [
    (1., 1.),
    (1., 2.),
    (1., .5),
    (2., 1.),
    (2., 2.),
    (2., .5),
    (.5, 1.),
    (.5, 2.),
    (.5, .5),
    ]

mur_muf_titles = [ rf'$\mu_{{R}}={mur:.1f}$ $\mu_{{F}}={muf:.1f}$' for mur, muf in mur_muf ]


@scripter
def plot():
    infile = common.pull_arg('infile', type=str).infile
    model_file = common.pull_arg('modelfile', type=str).modelfile

    cols = svj.Columns.load(infile)
    features = common.read_training_features(model_file)
    X = cols.to_numpy(features)

    model = xgboost.XGBClassifier()
    model.load_model(model_file)

    score = model.predict_proba(X)[:,1]

    sel = (score > .3)
    common.logger.info(f'bdt>.3: selecting {sel.sum()} / {len(sel)} events')

    mt = cols.to_numpy(['mt']).ravel()
    scale_weight = cols.to_numpy(['scaleweights'])

    mt = mt[sel]
    scale_weight = scale_weight[sel]
    bins = np.linspace(180, 600, 40)



    fig, (top, bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))

    central, _ = np.histogram(mt, bins)
    l = top.step(bins[:-1], central, where='post', label=mur_muf_titles[0])[0]
    bot.step(bins[:-1], central/central, where='post', label=mur_muf_titles[0], color=l.get_color())

    max_up = np.ones_like(central)
    max_down = np.ones_like(central)

    for i_var in range(1, len(mur_muf)):
        h, _ = np.histogram(mt, bins, weights=scale_weight[:,i_var])
        l = top.step(bins[:-1], h, where='post', label=mur_muf_titles[i_var])[0]
        bot.step(bins[:-1], h / central, where='post', label=mur_muf_titles[i_var], color=l.get_color())
        max_up = np.maximum(max_up, h/central)
        max_down = np.minimum(max_down, h/central)

    top.set_title(osp.basename(infile).replace('.npz',''))
    bot.set_xlabel('MT (GeV)')
    top.set_ylabel(r'$N_{events}$')
    top.legend()

    err_up = max_up.mean()
    err_down = max_down.mean()

    bot.plot([bins[0], bins[-1]], [err_up, err_up], c='black', linestyle='--')
    bot.plot([bins[0], bins[-1]], [err_down, err_down], c='black', linestyle='--')

    plt.savefig('tmp.png', bbox_inches="tight")
    common.imgcat('tmp.png')

    print(f'{err_up-1=:.5f}')
    print(f'{1-err_down=:.5f}')

    print(f'avg err={.5*(err_up-1) + .5*(1-err_down)}')

if __name__ == '__main__': scripter.run()