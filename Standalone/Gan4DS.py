from Preprocessor import Preprocessor
from NeuralNetworkLayout import NeuralNetworkLayout
from NeuralNetworkTraining import NeuralNetworkTraining

#add in where to modify the unseen energy

variables_of_interest=['s1']
energies=['50','100','150','200']

p=Preprocessor(variables=variables_of_interest,en=energies)
p.visualiseData(p.training_data,save=True)

n=NeuralNetworkLayout(
    gtrate=1e-4,dtrate=1e-4,glayers=4,dlayers=4,gnodes=50,dnodes=50,
    gdo=0.1,ddo=0.1,gbeta1=0.9,dbeta1=0.9,dimensions=len(variables_of_interest),noise=100,fileDir=p.dir_output)

n.saveHyperParameters()
n.compileGAN(verbose=False)

trainable_data={k:p.training_data[k] for k in p.training_data if k not in {'200'}}
unseen_data={k:p.training_data[k] for k in p.training_data if k in {'200'}}

t=NeuralNetworkTraining(n,tds=trainable_data,unseends=unseen_data,bs=100,fileDir=p.dir_output)
t.normaliseData()
t.initiateTraining()