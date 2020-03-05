from Preprocessor import Preprocessor
from NeuralNetworkLayout import NeuralNetworkLayout
from NeuralNetworkTraining import NeuralNetworkTraining
from Postprocessor import Postprocessor
#add in where to modify the unseen energy
from tensorboard.plugins.hparams import api as hp
from os import makedirs
import tensorflow as tf
from os.path import exists

variables_of_interest=['s1']
energies=['1','10','20','40','60']
unsen_energies=[]

HP_NODES = hp.HParam('num_nodes', hp.Discrete([40, 50]))
HP_LAYERS = hp.HParam('num_layers', hp.Discrete([2, 4]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))

METRIC_ACCURACY = 'accuracy'

pre=Preprocessor(idir='in/pickles',variables=variables_of_interest,en=energies)
pre.visualiseData(pre.training_data,save=True)

logs_hyperparam_dir=pre.dir_output+'/logs/hyperparams'
file_writer = tf.summary.create_file_writer(logs_hyperparam_dir)
with file_writer.as_default():
  hp.hparams_config(
    hparams=[HP_NODES, HP_LAYERS, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
makedirs(pre.dir_output+'/sessions')
pre.dir_output+='/sessions'
session_num = 0
for nodes in HP_NODES.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for layers in HP_LAYERS.domain.values:
      current_version=int(pre.getLatestVersion(pre.dir_output))+1
      dir_output=pre.dir_output+'/session_'+str(current_version)+"_"
      if not exists(dir_output):
        makedirs(dir_output)
        makedirs(dir_output+'/figures/')
        makedirs(dir_output+'/figures/all/')
        makedirs(dir_output+'/figures/final_product/')

      hparams = {
          HP_NODES: nodes,
          HP_DROPOUT: dropout_rate,
          HP_LAYERS: layers,
      }
      run_name = "session-%d" % session_num
      print('--- Starting session: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})

      n=NeuralNetworkLayout(
      gtrate=1e-4,dtrate=1e-4,glayers=hparams[HP_LAYERS],dlayers=hparams[HP_LAYERS],gnodes=hparams[HP_NODES],dnodes=hparams[HP_NODES],
      gdo=hparams[HP_DROPOUT],ddo=hparams[HP_DROPOUT],gbeta1=0.9,dbeta1=0.9,dimensions=len(variables_of_interest),noise=100,echeck=20,fileDir=dir_output)

      n.compileGAN(verbose=False)
      n.saveHyperParameters()

      trainable_data={k:pre.training_data[k] for k in pre.training_data if k not in unsen_energies}
      unseen_data={k:pre.training_data[k] for k in pre.training_data if k in unsen_energies}


      t=NeuralNetworkTraining(n,tds=trainable_data,bs=100,epochs=100,epochCheck=n.epochCheck,fileDir=dir_output,filewriter=file_writer)
      t.initiateTraining()

      post=Postprocessor(t.all_epochs,t.d_acc,t.d_loss,dir_output,len(variables_of_interest))
      post.createAnimation()

      file_writer2 = tf.summary.create_file_writer(logs_hyperparam_dir+'/'+str(session_num))
      with file_writer2.as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_ACCURACY, t.d_acc[-1], step=1)

      session_num += 1
