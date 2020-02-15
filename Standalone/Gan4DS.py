from Preprocessor import Preprocessor
from NeuralNetworkLayout import NeuralNetworkLayout
from NeuralNetworkTraining import NeuralNetworkTraining
from Postprocessor import Postprocessor
#add in where to modify the unseen energy

variables_of_interest=['s1']
energies=['50','75','100','150','200']
unsen_energies=[]

pre=Preprocessor(variables=variables_of_interest,en=energies)
pre.visualiseData(pre.training_data,save=True)

n=NeuralNetworkLayout(
    gtrate=1e-4,dtrate=1e-4,glayers=4,dlayers=4,gnodes=50,dnodes=50,
    gdo=0.1,ddo=0.1,gbeta1=0.9,dbeta1=0.9,dimensions=len(variables_of_interest),noise=100,fileDir=pre.dir_output)

n.compileGAN(verbose=False)
n.saveHyperParameters()

trainable_data={k:pre.training_data[k] for k in pre.training_data if k not in unsen_energies}
unseen_data={k:pre.training_data[k] for k in pre.training_data if k in unsen_energies}

t=NeuralNetworkTraining(n,tds=trainable_data,bs=100,epochs=50,epochCheck=5,fileDir=pre.dir_output)
t.initiateTraining()

post=Postprocessor(t.all_epochs,t.d_acc,t.d_loss,pre.dir_output,len(variables_of_interest))
post.createAnimation()