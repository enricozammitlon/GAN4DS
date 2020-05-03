#from __future__ import division
#import tensorflow as tf
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv
from os.path import exists
from os import makedirs
import matplotlib as mpl
from collections import defaultdict
from mlxtend.evaluate import permutation_test
from scipy.stats import mannwhitneyu,ttest_ind,ks_2samp,gmean,moment,wasserstein_distance,energy_distance,chisquare
# http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def getData(energies, variables_of_interest):
    allTrees = []
    for energy in energies:
        allTrees.append( pickle.load(
            open("in/pickles/outRun_"+energy+".p", "rb")))
    return allTrees

def get_distributed_energies(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
        distribution={'midpoint':[],'count':[]}
        for row in reader:
            distribution['midpoint'].append(row[0])
            distribution['count'].append(row[1])

    data = distribution['count']
    hist, bins = np.histogram(distribution['midpoint'],bins=200,weights=data)

    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(1000)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = np.asarray(bin_midpoints[value_bins]).astype(int)
    plt.subplot(121)
    plt.hist(distribution['midpoint'],bins=50,weights=data)
    plt.title("Distribution")
    plt.subplot(122)
    plt.hist(random_from_cdf, 50)
    plt.title("Random Sample from Distribution")
    plt.savefig(path.replace('.dat','.png'))
    plt.close()
    return random_from_cdf

def get_train_data(energies, variables_of_interest, training_ds, batch_size, conditions):
    batches = []
    for en in energies:
        sub_batches = []
        for var in variables_of_interest:
            random_num = np.random.randint(
                0, len(training_ds[en][var]), batch_size)
            sub_batches.append(
                training_ds[en][var][random_num].reshape(batch_size))
        single_batch = np.array(np.matrix(sub_batches).T)
        batches.append(single_batch)
    batches = np.concatenate(batches)

    hyperparams = np.concatenate([np.full(fill_value=float(
        a), shape=(batch_size, 1)) for a in energies])
    all_hyperparams = [hyperparams]

    for param in conditions:
        currentParam = []
        for en in range(len(energies)):
            random_num = np.random.randint(
                0, len(param[en]), batch_size)
            currentParam.append(
                param[en][random_num].reshape(batch_size, 1))
        currentParam = np.concatenate(currentParam)
        all_hyperparams += [currentParam]
    return batches, all_hyperparams

def analyse_differences(training_ds,conditions,energies_inputted,variables_of_interest):
    for num,var in enumerate(variables_of_interest):
        all_p_values=[]
        distance=[[],[]]
        for i in range(len(energies_inputted)):
            #print('Observed whitney (U,P): %.6f %.6f' % mannwhitneyu(x, y))
            test_variance=lambda x,y : np.abs(moment(x,moment=2)/moment(y,moment=2))
            test_man = lambda x,y : mannwhitneyu(x,y)[0]
            test_amean= lambda x,y : np.abs(np.mean(x)-np.mean(y))
            test_gmean = lambda x,y : np.abs(gmean(x)-gmean(y))
            test_ks = lambda x,y : ks_2samp(x,y)[0]
            test_chi = lambda x,y : chisquare(np.histogram(x,bins=50)[0],np.histogram(y,bins=50)[0])[0]
            x=conditions[num][i]
            y=training_ds[i][var]
            distance[0].append([wasserstein_distance(x,y),i])
            distance[1].append([energy_distance(x,y),i])
            x=normalize(x)
            y=normalize(y)
            p_value = permutation_test(x, y,
                                method='approximate',
                                num_rounds=1000,
                                func= test_gmean,
                                seed=0)
            all_p_values.append([p_value,i])

        #all_p_values_mean=[[0.001, 0], [0.0, 1], [0.0, 2], [0.0, 3], [0.0, 4], [0.0, 5], [0.0, 6], [0.0, 7], [0.0, 8], [0.0, 9], [0.0, 10], [0.0, 11], [0.0, 12], [0.0, 13], [0.0, 14], [0.0, 15], [0.0, 16], [0.0, 17], [0.0, 18], [0.0, 19], [0.0, 20], [0.0, 21], [0.0, 22], [0.0, 23], [0.0, 24], [0.0, 25], [0.0, 26], [0.0, 27], [0.0, 28], [0.0, 29], [0.0, 30], [0.0, 31], [0.0, 32], [0.0, 33], [0.0, 34], [0.0, 35], [0.0, 36], [0.0, 37], [0.0, 38], [0.0, 39], [0.0, 40], [0.0, 41], [0.0, 42], [0.0, 43], [0.0, 44], [0.0, 45], [0.0, 46], [0.0, 47], [0.0, 48], [0.0, 49], [0.0, 50], [0.0, 51], [0.0, 52], [0.0, 53], [0.0, 54], [0.0, 55], [0.0, 56], [0.0, 57], [0.0, 58], [0.0, 59], [0.0, 60], [0.0, 61], [0.0, 62], [0.0, 63], [0.0, 64], [0.0, 65], [0.0, 66], [0.0, 67], [0.0, 68], [0.0, 69], [0.0, 70], [0.0, 71], [0.0, 72], [0.0, 73], [0.0, 74], [0.0, 75], [0.022, 76], [0.0, 77], [0.571, 78], [0.444, 79], [0.198, 80], [0.0, 81], [0.0, 82], [0.0, 83], [0.0, 84], [0.009, 85], [0.116, 86], [0.0, 87], [0.0, 88], [0.0, 89], [0.066, 90], [0.009, 91], [0.0, 92], [0.0, 93], [0.101, 94], [0.0, 95], [0.0, 96], [0.0, 97], [0.0, 98], [0.036, 99], [0.19, 100], [0.004, 101], [0.207, 102], [0.651, 103], [0.416, 104], [0.0, 105], [0.9, 106], [0.0, 107], [0.0, 108], [0.232, 109], [0.0, 110], [0.766, 111], [0.0, 112], [0.138, 113], [0.928, 114], [0.554, 115], [0.0, 116], [0.125, 117], [0.0, 118], [0.347, 119], [0.01, 120], [0.0, 121], [0.0, 122], [0.41, 123], [0.046, 124], [0.53, 125], [0.066, 126], [0.0, 127], [0.04, 128], [0.0, 129], [0.004, 130], [0.0, 131], [0.0, 132], [0.022, 133], [0.091, 134], [0.005, 135], [0.298, 136], [0.307, 137], [0.0, 138], [0.055, 139], [0.001, 140], [0.518, 141], [0.425, 142], [0.0, 143], [0.447, 144], [0.0, 145], [0.0, 146], [0.179, 147], [0.0, 148], [0.0, 149], [0.0, 150], [0.364, 151], [0.0, 152], [0.0, 153], [0.0, 154], [0.075, 155], [0.0, 156], [0.0, 157], [0.0, 158], [0.0, 159], [0.0, 160], [0.172, 161], [0.0, 162], [0.0, 163], [0.053, 164], [0.0, 165], [0.0, 166], [0.029, 167], [0.0, 168], [0.0, 169], [0.161, 170], [0.006, 171], [0.0, 172], [0.975, 173], [0.692, 174], [0.0, 175], [0.503, 176], [0.071, 177], [0.938, 178], [0.003, 179], [0.006, 180], [0.0, 181], [0.0, 182], [0.118, 183], [0.105, 184], [0.0, 185], [0.0, 186], [0.0, 187], [0.0, 188], [0.066, 189], [0.27, 190], [0.038, 191], [0.793, 192], [0.931, 193], [0.107, 194], [0.0, 195], [0.0, 196], [0.149, 197], [0.603, 198], [0.0, 199], [0.0, 200], [0.204, 201], [0.0, 202], [0.068, 203], [0.0, 204], [0.0, 205], [0.297, 206], [0.769, 207], [0.675, 208], [0.035, 209], [0.0, 210], [0.006, 211], [0.0, 212], [0.0, 213], [0.0, 214], [0.0, 215], [0.0, 216], [0.0, 217], [0.0, 218], [0.0, 219], [0.0, 220], [0.0, 221], [0.0, 222], [0.0, 223], [0.0, 224], [0.0, 225], [0.0, 226], [0.0, 227], [0.0, 228], [0.0, 229]]
        x=[el[1] for el in all_p_values]
        y=[el[0] for el in all_p_values]
        print(len(np.where(np.array(y)>0.5)))
        print("Percentage in which p > 0.5 for %s : %.5f"%(var,len(np.where(np.array(y)>0.5))/len(y)))
        plt.scatter(x,y)
        plt.xlabel("Nuclear Recoil Energy", size=11, labelpad=5)
        plt.ylabel("P value", size=11, labelpad=5)
        plt.title("Permutation test for \n"+var+" on two-tailed geometric mean differences.")
        plt.savefig('./final_result/analysis/pval_gmean_test_'+var+'.png')
        plt.close()

        x=[el[1] for el in distance[0]]
        y=[el[0] for el in distance[0]]
        h,xr,yr,im=plt.hist2d(x,y,bins=[50,50])
        plt.colorbar(im)
        plt.xlabel("Nuclear Recoil Energy", size=11, labelpad=5)
        plt.ylabel("Wasserstein Distance", size=11, labelpad=5)
        plt.title("Wasserstein distance for \n"+var+" for varying recoil energies")
        plt.savefig('./final_result/analysis/wasserstein_distance_'+var+'.png')
        plt.close()

        x=[el[1] for el in distance[1]]
        y=[el[0] for el in distance[1]]
        h,xr,yr,im=plt.hist2d(x,y,bins=[50,50])
        plt.colorbar(im)
        plt.xlabel("Nuclear Recoil Energy", size=11, labelpad=5)
        plt.ylabel("Energy Distance", size=11, labelpad=5)
        plt.title("Energy distance for \n"+var+" for varying recoil energies")
        plt.savefig('./final_result/analysis/energy_distance_'+var+'.png')
        plt.close()

def get_noise(energies, training_ds, batch_size, noise_size, conditions):
    hyperparams = np.concatenate([np.full(fill_value=float(
        a), shape=(batch_size, 1)) for a in energies])
    all_hyperparams = [hyperparams]
    noise = np.random.normal(
        size=(len(training_ds)*batch_size, noise_size))
    for param in conditions:
        currentParam = []
        for en in range(len(energies)):
            random_num = np.random.randint(
                0, len(param[en]), batch_size)
            currentParam.append(
                param[en][random_num].reshape(batch_size, 1))
        currentParam = np.concatenate(currentParam)
        all_hyperparams += [currentParam]
    return noise, all_hyperparams

def normaliseData(training_ds, energies, variables_of_interest):
    normalisation = []
    for en in range(len(training_ds)):
        for var in variables_of_interest:
            temp_maximum = np.max(training_ds[energies[en]][var])
            training_ds[energies[en]][var] /= temp_maximum
            normalisation.append(temp_maximum)
    return normalisation

def draw_physical_graphs(gan_data,gan_norms,mass):
    conditions=gan_data.copy()
    allNorms=gan_norms.copy()
    selected_energies=get_distributed_energies('./sampling_distributions/Ar_c1dat_m'+mass+'.dat')

    massString=mass.replace('-','.')
    if not exists('./final_result/discrimination_plots/'+massString):
        makedirs('./final_result/discrimination_plots/'+massString)
    for num in range(len(conditions)):
        conditions[num]=[conditions[num][s]*allNorms[num][s] for s in selected_energies]

    training_ds = getData([str(e) for e in selected_energies], ['s1','s2','f200like'])

    y=[]
    x=[]
    for i in range(len(selected_energies)):
        y.append(training_ds[i]['f200like'])
        x.append(training_ds[i]['s1'])
    y = np.concatenate(y)
    x = np.concatenate(x)

    h1_training,x_edges_1,y_edges_1,im= plt.hist2d(x, y,range= [[0, 600], [0.2, 1]],bins=[100,100],norm=mpl.colors.LogNorm(),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.ylabel(r"$f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel("$S_1$(NPE)", size=11, labelpad=5)
    plt.title(r"G4DS with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/discrimination_plots/'+massString+'/g4_f200_vs_s1.png')
    plt.close()

    plt.figure()
    y = np.array(conditions[-1])
    x = np.array(conditions[0])
    y = np.concatenate(y)
    x = np.concatenate(x)

    h1_gan,x_edges_1,y_edges_1,im= plt.hist2d(x,y,range= [[0, 600], [0.2, 1]],bins=[100,100],norm=mpl.colors.LogNorm(),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.ylabel(r"$f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.title(r"GAN with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/discrimination_plots/'+massString+'/gan_f200_vs_s1.png')
    plt.close()

    plt.figure()
    y=[]
    x=[]
    for i in range(len(selected_energies)):
        x.append(training_ds[i]['s1'])
        y.append(np.log(np.divide(np.array(training_ds[i]['s2']),np.array(training_ds[i]['s1']))))

    y = np.concatenate(y)
    x = np.concatenate(x)

    h2_training,x_edges,y_edges,im= plt.hist2d(x, y,range= [[0, 600], [0, 5]],bins=[100,100],norm=mpl.colors.LogNorm(vmin=1),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(r"G4DS with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/discrimination_plots/'+massString+'/g4_s1_over_s2_vs_s1.png')
    plt.close()

    x = np.array(conditions[0])
    y = np.log(np.divide(np.array(conditions[1]),np.array(conditions[0])))
    y = np.concatenate(y)
    x = np.concatenate(x)

    h2_gan,x_edges_2,y_edges_2,im = plt.hist2d(x, y,range= [[0, 600], [0, 5]],bins=[100,100],norm=mpl.colors.LogNorm(vmin=1),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(r"GAN with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/discrimination_plots/'+massString+'/gan_s1_over_s2_vs_s1.png')
    plt.close()

    diff = (h2_gan - h2_training)
    custom_cmap=plt.get_cmap("seismic")
    h2_diff= plt.pcolormesh(x_edges_2, y_edges_2,diff.T,norm=mpl.colors.Normalize(vmin=-1000,vmax=1000),cmap=custom_cmap)
    cbar= plt.colorbar(h2_diff)
    plt.xlabel(r"Difference in $S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"Difference in log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(r"GAN4DS-G4DS with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/discrimination_plots/'+massString+'/difference_s1_over_s2_vs_s1.png')
    plt.close()

    diff = (h1_gan - h1_training)
    custom_cmap=plt.get_cmap("seismic")
    h1_diff= plt.pcolormesh(x_edges_1, y_edges_1,diff.T,norm=mpl.colors.Normalize(vmin=-1000,vmax=1000),cmap=custom_cmap)
    cbar= plt.colorbar(h1_diff)
    plt.ylabel(r"Difference in $f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel(r"Difference in $S_1$(NPE)", size=11, labelpad=5)
    plt.title(r"GAN4DS-G4 with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/discrimination_plots/'+massString+'/difference_f200_vs_s1.png')
    plt.close()
    plt.figure()

def draw_gan_output(training_ds,conditions,variables_of_interest):
    for num,var in enumerate(variables_of_interest):
        for i in range(len(conditions[num])):
            plt.hist(conditions[num][i],density=True,label="Gan4DS")
            plt.hist(training_ds[i][var],density=True,label="G4DS")
            varString=''
            if(var=='s1'):
                varString=r'$S_1$'
            elif(var=='s2'):
                varString=r'$S_2$'
            else:
                varString=r'$f_{200}$'
            plt.xlabel(varString, size=11, labelpad=5)
            plt.ylabel(r"$\rho(x)$", size=11, labelpad=2)
            plt.legend()
            plt.title(varString+" for "+str(i)+" keV recoil energy")
            plt.savefig('./final_result/gan_output/'+var+'/'+var+'_'+str(i)+'.png')
            plt.close()

currentRun = "run_1_"
currentSession = "session_1_"

stream = open("/workspaces/mphys-project/G4_RUNS/serial_architecture/working_3D_cgan_s1_s2_f200/sessions/session_1_/layouts/config.yaml", "r+")
data = yaml.load(stream, Loader=yaml.FullLoader)

variables_of_interest = data['variables_of_interest']
energies_inputted = data['energies']
if(isinstance(energies_inputted, dict)):
    if(energies_inputted.get('range')):
        energies = list(map(str, range(energies_inputted.get(
            'range')[0], energies_inputted.get('range')[1])))
    if(energies_inputted.get('exact')):
        energies = list(map(str, energies_inputted.get('exact')))
batch_size = data['batchSize']
noise_size = data['noiseSize']
epochs = data['epochs']
epoch_check = data['epochCheck']
conditions = []
allNorms = []


all_stuff = pickle.load(
            open("./final_result/testing_data.p", "rb"))
conditions=all_stuff['data'].copy()
allNorms=all_stuff['normalisation'].copy()

energies_inputted=np.arange(5,235,1)
energies_inputted = [str(e) for e in energies_inputted]

for num,cond in enumerate(conditions):
    conditions[num]=[conditions[num][s]*allNorms[num][s] for s in range(len(energies_inputted))]
training_ds = getData(energies_inputted, ['s1','s2','f200like'])

#draw_gan_output(training_ds,conditions,variables_of_interest)
analyse_differences(training_ds,conditions,energies_inputted,['s1','s2','f200like'])

'''
masses=["1-5","2","2-5","3","4"]
#masses=['1-5']
for mass in masses:
    print("Current log-mass: %s"%(mass.replace('-','.')))
    conditions=all_stuff['data'].copy()
    allNorms=all_stuff['normalisation'].copy()
    draw_physical_graphs(conditions,allNorms,mass)
'''