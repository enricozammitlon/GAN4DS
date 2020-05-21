# This script reads in training data from the saved pickles
# and also plots out what it looks like. In the future the plotting needs to be fixed
# cause right now it plots out all on one plot for each variable and it looks messy
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
        self.training_data=self.getData(variables)

    def getLatestVersion(self,folder):
        lastFile = [int(f.split('_')[1]) for f in listdir(folder) if isdir(join(folder, f))]
        lastFile.sort()
        if(len(lastFile)<1):
            return "0"
        else:
            version = lastFile[-1]
        return version

    #Convert root to numpy arrays of variables of interest
    def getData(self,vars_of_interest):
        allTrees={}
        for energy in self.energies:
            allTrees[energy]=pickle.load( open( self.dir_input+"/outRun_"+energy+".p", "rb" ) )
        result={}
        for key,value in allTrees.items():
          filt_value = {k2:v2 for k2,v2 in value.items() if k2 in vars_of_interest}
          if(filt_value):
            result[key]=filt_value
        return result

    def getConditions(self,num):
        conditions=[]
        for var in range(num):
            currentCond=[]
            for en in self.training_data:
                currentCond.append(self.training_data[en][self.variables_of_interest[var]])
            conditions.append(currentCond)
        return conditions

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
                plt.close()
