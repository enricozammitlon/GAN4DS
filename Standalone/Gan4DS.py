from Preprocessor import Preprocessor
from NeuralNetworkLayout import NeuralNetworkLayout
from NeuralNetworkTraining import NeuralNetworkTraining
from Postprocessor import Postprocessor
from tensorboard.plugins.hparams import api as hp
from os import makedirs
import tensorflow as tf
from os.path import exists
import yaml
from itertools import product,permutations

stream = open("./layouts/config.yaml","r+")
data = yaml.load(stream)

variables_of_interest=data['variables_of_interest']
energies_inputted=data['energies']
if(isinstance(energies_inputted,dict)):
  if(energies_inputted.get('range')):
    energies=list(map(str, range(energies_inputted.get('range')[0],energies_inputted.get('range')[1])))
  if(energies_inputted.get('exact')):
    energies=list(map(str, energies_inputted.get('exact')))
epochs=data['epochs']
epoch_check=data['epochCheck']
batch_size=data['batchSize']
noise_size=data['noiseSize']
overrides=data['overrides']
verbose=data['verbose']
g_rate=data['g_rate']
d_rate=data['d_rate']
g_beta1=data['g_beta1']
d_beta1=data['d_beta1']
METRIC_ACCURACY = 'accuracy'
hp_methods={'d_nodes':hp.Discrete,'g_nodes':hp.Discrete,'dropout':hp.RealInterval}

all_hyper_params = []
hyperparams_limits = []
for override in overrides:
  current_override= overrides[override]
  if(override=='dropout'):
    all_hyper_params.append(hp.HParam(override,hp_methods[override](current_override[0],current_override[1])))
    current_lim=[]
    for param in (all_hyper_params[-1].domain.min_value,all_hyper_params[-1].domain.max_value):
      current_lim.append(param)
    hyperparams_limits.append(current_lim)
  else:
    all_hyper_params.append(hp.HParam(override,hp_methods[override](current_override)))
    current_lim = []
    for param in all_hyper_params[-1].domain.values:
      current_lim.append(param)
    hyperparams_limits.append(current_lim)

pre=Preprocessor(idir='in/pickles',variables=variables_of_interest,en=energies)
pre.visualiseData(pre.training_data,save=True)

logs_hyperparam_dir=pre.dir_output+'/logs/hyperparams'
file_writer = tf.summary.create_file_writer(logs_hyperparam_dir)
with file_writer.as_default():
  hp.hparams_config(
    hparams=all_hyper_params,
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
makedirs(pre.dir_output+'/sessions')
pre.dir_output+='/sessions'
session_num = 0
for current_config in product(*hyperparams_limits):
      current_version=int(pre.getLatestVersion(pre.dir_output))+1
      dir_output=pre.dir_output+'/session_'+str(current_version)+"_"
      if not exists(dir_output):
        makedirs(dir_output)
        makedirs(dir_output+'/figures/')
        makedirs(dir_output+'/figures/all/')
        makedirs(dir_output+'/figures/final_product/')
      hparams = { all_hyper_params[i]:current_config[i] for i,override in enumerate(overrides)}

      run_name = "session-%d" % session_num
      print('--- Starting session: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      current_overrides=hparams
      n=NeuralNetworkLayout(
      gtrate=g_rate,dtrate=d_rate,gbeta1=g_beta1,dbeta1=d_beta1,dimensions=len(variables_of_interest),noise=noise_size,echeck=epoch_check,fileDir=dir_output,overrides=current_overrides)

      n.compileGAN(verbose)
      n.saveHyperParameters()

      t=NeuralNetworkTraining(n,tds=pre.training_data,bs=batch_size,epochs=epoch_check,epochCheck=n.epochCheck,fileDir=dir_output,filewriter=file_writer)
      t.initiateTraining()

      post=Postprocessor(t.all_epochs,t.d_acc,t.d_loss,dir_output,len(variables_of_interest))
      post.createAnimation()

      file_writer2 = tf.summary.create_file_writer(logs_hyperparam_dir+'/'+str(session_num))
      with file_writer2.as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_ACCURACY, t.d_acc[-1], step=1)
      session_num += 1
print("--JOB DONE--")