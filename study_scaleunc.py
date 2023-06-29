import os.path as osp
import common
import svj_ntuple_processing as svj
import xgboost
import numpy as np
import matplotlib.pyplot as plt


scripter = common.Scripter()

# (mur, muf)
mur_muf = [
    (1., 1.), # 0
    (1., 2.), # 1
    (1., .5), # 2
    (2., 1.), # 3
    (2., 2.), # 4
    (2., .5), # 5 <-
    (.5, 1.), # 6
    (.5, 2.), # 7 <-
    (.5, .5), # 8
    ]

mur_muf_titles = [ rf'$\mu_{{R}}={mur:.1f}$ $\mu_{{F}}={muf:.1f}$' for mur, muf in mur_muf ]


@scripter
def plot():
    infile = common.pull_arg('infile', type=str).infile
    model_file = common.pull_arg('modelfile', type=str).modelfile
    mtaxis = list(common.pull_arg('mtaxis', type=float, nargs='*').mtaxis)
    unnormalized = common.pull_arg('--unnormalized', action='store_true').unnormalized

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

    # Drop column 5 and 7: the .5/2 and 2/.5 variations
    scale_weight = scale_weight[:, np.array([0,1,2,3,4,6,8])]
    del mur_muf[7]; del mur_muf[5]
    del mur_muf_titles[7]; del mur_muf_titles[5]

    assert scale_weight.shape == (len(mt), 7)


    bins = np.linspace(180, 600, 40)
    if len(mtaxis) == 3:
        bins = np.linspace(mtaxis[0], mtaxis[1], int(mtaxis[2]))
    else:
        raise Exception('Specify mt axis as left right nbins')


    fig, (top, bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))

    central, _ = np.histogram(mt, bins)
    l = top.step(bins[:-1], central, where='post', label=mur_muf_titles[0])[0]
    bot.step(bins[:-1], central/central, where='post', label=mur_muf_titles[0], color=l.get_color())

    weight_up = np.max(scale_weight, axis=-1)
    weight_down = np.min(scale_weight, axis=-1)

    up, _ = np.histogram(mt, bins, weights=weight_up)
    if not unnormalized: up *= cols.metadata['factor_up']
    l = top.step(bins[:-1], up, where='post', label='Scale up')[0]
    bot.step(bins[:-1], up / central, where='post', color=l.get_color())

    down, _ = np.histogram(mt, bins, weights=weight_down)
    if not unnormalized: down *= cols.metadata['factor_down']
    l = top.step(bins[:-1], down, where='post', label='Scale down')[0]
    bot.step(bins[:-1], down / central, where='post', color=l.get_color())

    top.set_title(osp.basename(infile).replace('.npz',''))
    bot.set_xlabel('MT (GeV)')
    top.set_ylabel(r'$N_{events}$')
    top.legend()
    top.set_yscale('log')

    err_up = (up / central).mean()
    err_down = (down / central).mean()

    bot.plot([bins[0], bins[-1]], [err_up, err_up], c='black', linestyle='--')
    bot.plot([bins[0], bins[-1]], [err_down, err_down], c='black', linestyle='--')

    plt.savefig('tmp.png', bbox_inches="tight")
    common.imgcat('tmp.png')

    print(f'{err_up-1=:.5f}')
    print(f'{1-err_down=:.5f}')
    print(f'avg err={.5*(err_up-1) + .5*(1-err_down)}')




if __name__ == '__main__': scripter.run()