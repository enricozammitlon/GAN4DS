# This is the main program for the training part
# It is the start of the program and also connects all the other modules

from Preprocessor import Preprocessor
from NeuralNetworkLayout import NeuralNetworkLayout
from NeuralNetworkTraining import NeuralNetworkTraining
from Postprocessor import Postprocessor
from tensorboard.plugins.hparams import api as hp
from os import makedirs
import tensorflow as tf
from os.path import exists
import yaml
from itertools import product, permutations, combinations
from pathlib import Path
import shutil
import subprocess
import pickle

# Spin up tensorboard to be able to monitor output on localhost:6006
# Use  -L 16006:localhost:6006 with ssh to be able to see it in your browser
process = subprocess.Popen(['tensorboard','--bind_all' ,'--logdir', 'out/'])
# Load in the configuration variables like epochs, epoch check + overrides
stream = open("./layouts/config.yaml", "r+")
data = yaml.load(stream, Loader=yaml.FullLoader)

variables_of_interest = data['variables_of_interest']
energies_inputted = data['energies']
if(isinstance(energies_inputted, dict)):
    if(energies_inputted.get('range')):
        energies = list(map(str, range(energies_inputted.get(
            'range')[0], energies_inputted.get('range')[1])))
    if(energies_inputted.get('exact')):
        energies = list(map(str, energies_inputted.get('exact')))
epochs = data['epochs']
epoch_check = data['epochCheck']
batch_size = data['batchSize']
noise_size = data['noiseSize']
overrides = data['overrides']
verbose = data['verbose']
g_rate = data['g_rate']
d_rate = data['d_rate']
g_beta1 = data['g_beta1']
d_beta1 = data['d_beta1']
# For now the metric in the config doesnt do anything but later
# It would be nice to have which metrics to show from there
METRIC_ACCURACY = 'accuracy'
# Needed for hyperparameters combinations
hp_methods = {'d_nodes': hp.Discrete,
              'g_nodes': hp.Discrete, 'dropout': hp.RealInterval}
# The following setup the lists and collections
# For the hyperparams to then be used directly in tensorboard
# And also be used to permute different hyperparams
all_hyper_params = []
hyperparams_limits = []
for override in overrides:
    current_override = overrides[override]
    if(override == 'dropout'):
        all_hyper_params.append(hp.HParam(override, hp_methods[override](
            current_override[0], current_override[1])))
        current_lim = []
        for param in (all_hyper_params[-1].domain.min_value, all_hyper_params[-1].domain.max_value):
            current_lim.append(param)
        hyperparams_limits.append(current_lim)
    else:
        all_hyper_params.append(
            hp.HParam(override, hp_methods[override](current_override)))
        current_lim = []
        for param in all_hyper_params[-1].domain.values:
            current_lim.append(param)
        hyperparams_limits.append(current_lim)
# Load in the initial training data
if not exists('out/'):
    makedirs('out/')
pre = Preprocessor(idir='in/pickles',
                   variables=variables_of_interest, en=energies)
# Save plots of what training data looks like
pre.visualiseData(pre.training_data, save=True)
# This is actually not needed
all_metrics = [hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
# This will appear under the hyperparams tab in tensorboard
for num, i in enumerate(variables_of_interest):
    METRIC_FIRST_MOMENT = 'First Moment '+i
    METRIC_SECOND_MOMENT = 'Second Moment '+i
    all_metrics.append(hp.Metric(METRIC_FIRST_MOMENT,
                                 display_name=METRIC_FIRST_MOMENT))
    all_metrics.append(hp.Metric(METRIC_SECOND_MOMENT,
                                 display_name=METRIC_SECOND_MOMENT))

logs_hyperparam_dir = pre.dir_output+'/logs/hyperparams'
file_writer = tf.summary.create_file_writer(logs_hyperparam_dir)
with file_writer.as_default():
    hp.hparams_config(
        hparams=all_hyper_params,
        metrics=all_metrics,
    )
file_writer.flush()
makedirs(pre.dir_output+'/sessions')
pre.dir_output += '/sessions'
session_num = 0

for current_config in product(*hyperparams_limits):
    current_version = int(pre.getLatestVersion(pre.dir_output))+1
    dir_output = pre.dir_output+'/session_'+str(current_version)+"_"
    # Setup some useful folders
    if not exists(dir_output):
        makedirs(dir_output)
        makedirs(dir_output+'/figures/')
        makedirs(dir_output+'/figures/all/')
        makedirs(dir_output+'/figures/final_product/')
    hparams = {all_hyper_params[i]: current_config[i]
               for i, override in enumerate(overrides)}
    # A session is defined as one configuration of hyperparams
    run_name = "session-%d" % session_num
    print('--- Starting session: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    current_overrides = {h.name: hparams[h] for h in hparams}
    conditions = []
    # Load in any hyperparams particular to this variable being learnt
    for num, var in enumerate(variables_of_interest):
        if exists("./layouts/"+var+"/subconfig.yaml"):
            stream = open("./layouts/"+var+"/subconfig.yaml", "r+")
            data = yaml.load(stream, Loader=yaml.FullLoader)
            g_rate = data['g_rate']
            d_rate = data['d_rate']
            g_beta1 = data['g_beta1']
            d_beta1 = data['d_beta1']
            verbose = data['verbose']
            d_nodes = data['d_nodes']
            g_nodes = data['g_nodes']
            dropout = data['dropout']
            current_overrides = {'g_nodes': g_nodes,
                                 'd_nodes': d_nodes, 'dropout': dropout}
        pre.training_data = pre.getData([var])
        # Setup the layouts
        n = NeuralNetworkLayout(layoutDir="layouts/"+var,
                                gtrate=g_rate, dtrate=d_rate, gbeta1=g_beta1, dbeta1=d_beta1, dimensions=num+1, noise=noise_size, echeck=epoch_check, fileDir=dir_output, overrides=current_overrides)
        n.compileGAN(verbose)
        # Kind of legacy function for saving hyperparams but still useful to have
        n.saveHyperParameters()
        dirpath = Path(dir_output+'/figures/', 'all')
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        makedirs(dir_output+'/figures/all')
        # Setup training
        t = NeuralNetworkTraining(n, tds=pre.training_data, conditions=conditions, bs=batch_size,
                                  epochs=epochs, epochCheck=n.epochCheck, fileDir=dir_output, filewriter=file_writer)
        # Train and once complete returns the best generated data for all energies
        # for that variable, to be appended as a condition into the next variable
        latest_generated_data = t.initiateTraining()
        currentCond = []
        for en in latest_generated_data:
            currentCond.append(latest_generated_data[en][var])
        conditions.append(currentCond)
        # Useful to create plots such as acc/loss, moments etc. Also saves the movie reel
        post = Postprocessor(t.all_epochs, t.epochCheck, t.d_acc,
                             t.d_loss, t.d_x, t.d_x2, var, dir_output, len([var]))
        post.createAnimation()

        # Dump all logs about metrics and latest generated data for analysis later
        dirpath = Path(dir_output+'/data_logs')
        if not (dirpath.exists()):
            makedirs(dirpath)
        subdata=[t.d_x,t.d_x2,t.d_acc,t.d_loss]
        all_stuff={'data':latest_generated_data,'metrics':subdata}
        pickle.dump( all_stuff, open(dir_output+'/data_logs/data_log_'+var, "wb" ) )

        file_writer2 = tf.summary.create_file_writer(
            logs_hyperparam_dir+'/'+str(session_num))
        with file_writer2.as_default():
            hp.hparams(hparams)
            tf.summary.scalar(METRIC_ACCURACY, t.d_acc[-1], step=1)
            METRIC_FIRST_MOMENT = 'First Moment '+var
            METRIC_SECOND_MOMENT = 'Second Moment '+var
            tf.summary.scalar(METRIC_FIRST_MOMENT, t.d_x[-1], step=1)
            tf.summary.scalar(METRIC_SECOND_MOMENT, t.d_x2[-1], step=1)
        file_writer2.flush()
    session_num += 1
print("JOB DONE")
