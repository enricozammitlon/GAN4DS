# This script generates plots such as acc/loss, first and second moments ratios
# Also creates the movie reel of training
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

from os import listdir,makedirs
from os.path import isfile, join,exists

import matplotlib
import matplotlib.pyplot as plt

class Postprocessor :

    def __init__(self,epochs,epochCheck,dacc,dloss,d_x,d_x2,var_of_interest,dirOut,dimensionality):
        self.epochs=epochs
        self.epochCheck=epochCheck
        self.d_acc=dacc
        self.d_loss=dloss
        self.d_x=d_x
        self.d_x2=d_x2
        self.dir_out=dirOut
        self.var=var_of_interest
        self.dimensionality=dimensionality
        e=[epoch for epoch in range(0,len(epochs)) if epoch == 0 or (epoch+1) % self.epochCheck == 0]
        del self.epochs[0]
        del self.d_loss[0]
        del self.d_acc[0]
        del e[0]
        self.createAnimation()
        self.visualiseAccLoss()
        del self.d_x[0]
        del self.d_x2[0]
        self.visualiseMoments1(self.var,e,d_x)
        self.visualiseMoments2(self.var,e,d_x2)



    def createAnimation(self):
        basedir=self.dir_out+'/figures/all/'
        basename='reel_f_'
        images=[]
        files = [f for f in listdir(basedir) if isfile(join(basedir, f))]

        for filename in range(0,len(files)-1):
            images.append(imageio.imread(basedir+basename+str(filename+1)+'.png'))
        imageio.mimsave(self.dir_out+'/figures/final_product/'+self.var+'_'+str(self.dimensionality)+"D_cGAN_complete_reel.mp4",images)

    def visualiseMoments1(self,var,e,d_x,save=True):
        plt.figure()
        plt.plot(e,d_x, color = 'blue', label = "First Moment Ratio")
        plt.xlabel("Epoch number", size=11, labelpad=5)
        plt.ylabel("First Moment Ratio", size=11, labelpad=5, rotation="vertical")
        plt.title(f"Disciminator moment 1 Ratio for "+var+" in"+str(self.dimensionality)+"D cGAN")
        plt.legend(fontsize=11)
        if(save):
            plt.savefig(self.dir_out+'/figures/final_product/'+var+"_moment_1_graph_"+str(self.dimensionality)+"D_cGAN.png")
            plt.close()
    def visualiseMoments2(self,var,e,d_x2,save=True):
        plt.figure()
        plt.plot(e,d_x2, color = 'blue', label = "Second Moment Ratio")
        plt.xlabel("Epoch number", size=11, labelpad=5)
        plt.ylabel("Second Moment Ratio", size=11, labelpad=5, rotation="vertical")
        plt.title(f"Disciminator moment 2 Ratio for "+var+"in"+str(self.dimensionality)+"D cGAN")
        plt.legend(fontsize=11)
        if(save):
            plt.savefig(self.dir_out+'/figures/final_product/'+var+"_moment_2_graph_"+str(self.dimensionality)+"D_cGAN.png")
            plt.close()
    def visualiseAccLoss(self,save=True):
        plt.figure()
        plt.plot(self.epochs,self.d_loss, color = 'blue', label = "Loss")
        plt.plot(self.epochs,self.d_acc, color = 'orange', label = "Accuracy")
        plt.xlabel("Epoch number", size=11, labelpad=5)
        plt.ylabel("Accuracy/Loss", size=11, labelpad=5, rotation="vertical")
        plt.title(f"Disciminator performance for "+str(self.dimensionality)+"D cGAN")
        plt.legend(fontsize=11)
        if(save):
            plt.savefig(self.dir_out+'/figures/final_product/'+self.var+"_performance_graph_"+str(self.dimensionality)+"D_cGAN.png")
            plt.close()
#tensorboard --logdir out/run_6_/model/logs/train --host 0.0.0.0 --port 6006