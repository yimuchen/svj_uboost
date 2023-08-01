import os.path as osp
import common
import svj_ntuple_processing as svj
import xgboost
import numpy as np
import matplotlib.pyplot as plt


scripter = common.Scripter()


@scripter
def plot():
    infiles = common.pull_arg('infiles', type=str, nargs=5).infiles
    model_file = common.pull_arg('modelfile', type=str).modelfile
    # mtaxis = list(common.pull_arg('mtaxis', type=float, nargs='*').mtaxis)
    # unnormalized = common.pull_arg('--unnormalized', action='store_true').unnormalized

    jer_up = [f for f in infiles if 'jer_up' in f][0]
    jer_down = [f for f in infiles if 'jer_down' in f][0]
    jec_up = [f for f in infiles if 'jec_up' in f][0]
    jec_down = [f for f in infiles if 'jec_down' in f][0]

    for f in [jer_up, jer_down, jec_up, jec_down]: infiles.remove(f)
    central = infiles[0]

    svj.logger.info(f'jer_up:   {jer_up}')
    svj.logger.info(f'jer_down: {jer_down}')
    svj.logger.info(f'jec_up:   {jec_up}')
    svj.logger.info(f'jec_down: {jec_down}')
    svj.logger.info(f'central : {central}')

    jer_up = svj.Columns.load(jer_up)
    jer_up.metadata['systvar'] = 'jer_up'
    jer_down = svj.Columns.load(jer_down)
    jer_down.metadata['systvar'] = 'jer_down'
    jec_up = svj.Columns.load(jec_up)
    jec_up.metadata['systvar'] = 'jec_up'
    jec_down = svj.Columns.load(jec_down)
    jec_down.metadata['systvar'] = 'jec_down'
    central = svj.Columns.load(central)
    central.metadata['systvar'] = 'central'

    features = common.read_training_features(model_file)

    model = xgboost.XGBClassifier()
    model.load_model(model_file)


    bins = np.linspace(180, 600, 40) # mT axis


    for collection in [
        [central, jer_up, jer_down],
        [central, jec_up, jec_down],
        ]:

        fig, (top, bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))

        for cols in collection:
            systvar = cols.metadata['systvar']
            common.logger.info(f'Processing {systvar}')
            X = cols.to_numpy(features)
            score = model.predict_proba(X)[:,1]

            sel = (score > .3)
            common.logger.info(f'bdt>.3: selecting {sel.sum()} / {len(sel)} events')
            
            mt = cols.to_numpy(['mt']).ravel()

            hist, _ = np.histogram(mt, bins)
            if systvar == 'central':
                central_hist = hist

            common.logger.info(f'{systvar}: {hist}')

            l = top.step(bins[:-1], hist, where='post', label=systvar)[0]
            bot.step(bins[:-1], hist/central_hist, where='post', label=systvar, color=l.get_color())

        # top.set_title(osp.basename(infile).replace('.npz',''))
        bot.set_xlabel('MT (GeV)')
        top.set_ylabel(r'$N_{events}$')
        top.legend()
        top.set_yscale('log')

        plt.savefig('tmp.png', bbox_inches="tight")
        common.imgcat('tmp.png')



if __name__ == '__main__': scripter.run()