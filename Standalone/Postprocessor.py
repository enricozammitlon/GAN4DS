import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

from os import listdir,makedirs
from os.path import isfile, join,exists

import matplotlib
import matplotlib.pyplot as plt

class Postprocessor : 

    def __init__(self,epochs,dacc,dloss,dirOut,dimensionality):  
        self.epochs=epochs
        self.d_acc=dacc
        self.d_loss=dloss
        self.dir_out=dirOut
        self.dimensionality=dimensionality

        self.createAnimation()
        self.visualiseAccLoss()



    def createAnimation(self):
        basedir=self.dir_out+'/figures/all/'
        basename='reel_f_'
        images=[]
        files = [f for f in listdir(basedir) if isfile(join(basedir, f))]

        for filename in range(0,len(files)-1):
            images.append(imageio.imread(basedir+basename+str(filename+1)+'.png'))
        imageio.mimsave(self.dir_out+'/figures/final_product/'+str(self.dimensionality)+"D_cGAN_complete_reel.mp4",images)

    def visualiseAccLoss(self,save=True):
        del self.epochs[0]
        del self.d_loss[0]
        del self.d_acc[0]
        plt.figure()
        plt.plot(self.epochs,self.d_loss, color = 'blue', label = "Loss")
        plt.plot(self.epochs,self.d_acc, color = 'orange', label = "Accuracy")
        plt.xlabel("Epoch number", size=11, labelpad=5)
        plt.ylabel("Accuracy/Loss", size=11, labelpad=5, rotation="vertical")
        plt.title(f"Disciminator performance for "+str(self.dimensionality)+"D cGAN")
        plt.legend(loc="center right", fontsize=11)
        if(save):
            plt.savefig(self.dir_out+'/figures/final_product/'+"performance_graph_"+str(self.dimensionality)+"D_cGAN.png")

#tensorboard --logdir out/run_6_/model/logs/train --host 0.0.0.0 --port 6006