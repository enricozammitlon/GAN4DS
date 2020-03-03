import uproot
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import listdir,makedirs
from os.path import isdir, join,exists
import pickle

class Preprocessor:

    def __init__(self,idir='in',odir='out/',variables=['s1'],en=['50'],v=False):
        self.dir_input=idir
        self.dir_output=odir
        self.variables_of_interest=variables
        self.energies=en
        self.current_version=int(self.getLatestVersion(self.dir_output))+1
        self.dir_output+='/run_'+str(self.current_version)+"_"
        makedirs(self.dir_output)
        self.training_data=self.getData(v)

    def getLatestVersion(self,folder):
        lastFile = [f for f in listdir(folder) if isdir(join(folder, f))]
        if(len(lastFile)<1):
            return "0"
        else:
            lastFile=lastFile[0]
            begin = lastFile.find('_')+1
            end = lastFile.find('_',begin)
            version = lastFile[begin:end]
        return version

    #Convert root to numpy arrays of variables of interest
    def getData(self,verbose=False):
        allTrees={}
        for energy in self.energies:
            allTrees[energy]=pickle.load( open( self.dir_input+"/outRun_"+energy+".p", "rb" ) )
        return allTrees

    #Remember to add in units on plots
    def obtainUnits(self):
        pass

    def visualiseData(self,ds,save=True,source='training'):
        if not exists(self.dir_output+'/figures/'):
            makedirs(self.dir_output+'/figures/')

        cmap = matplotlib.cm.get_cmap('tab10')
        norm = matplotlib.colors.Normalize(vmin=float(self.energies[0]), vmax=float(self.energies[-1]))

        for variable in self.variables_of_interest:
            plt.figure()
            for energy in self.energies:
                plt.hist(ds[energy][variable], density = True, bins = 205, color=cmap(norm(float(energy))), alpha = 0.8,label=r"($E_{nr}="+energy+r"~{\rm KeV}$)")
            plt.xlabel(variable.capitalize(), size=11, labelpad=5)
            plt.ylabel(r"$\rho\left(x\right)$", size=11, labelpad=5, rotation="horizontal")
            plt.legend(loc="upper right", fontsize=11)
            plt.title(source.capitalize()+" data for "+variable.capitalize())
            if(save):
                plt.savefig(self.dir_output+'/figures/'+source+"_data_"+variable+".png")
