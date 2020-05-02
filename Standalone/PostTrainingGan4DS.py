#from __future__ import division
#import tensorflow as tf
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib as mpl
from collections import defaultdict

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
'''
for var in variables_of_interest:
    print('Current variable: '+var)
    model = tf.keras.models.load_model("../G4_RUNS/serial_architecture/working_3D_cgan_s1_s2_f200/sessions/session_1_/model/"+var+"_weights.model")
    training_ds = getData(energies, [var])
    current_energies = list(training_ds.keys())
    current_variables_of_interest = list(training_ds[energies[0]].keys())
    normalisation = normaliseData(training_ds, energies,
                                  current_variables_of_interest)

    generated_ds = {}

    noise, noise_hyperparams = get_noise(
        current_energies, training_ds, batch_size, noise_size, conditions)
    temp_generated = model.predict([noise]+noise_hyperparams)

    gen_class_length = int(temp_generated.shape[0]/(len(current_energies)))

    for en in range(1, len(current_energies)+1):
        generated_ds[current_energies[en-1]] = {}
        for num, var in enumerate(current_variables_of_interest):
            current_var = np.asarray(temp_generated)[:, num]
            gen_energies_var = current_var[(
                en-1)*gen_class_length:en*gen_class_length]
            generated_ds[current_energies[en-1]][var] = gen_energies_var

    multiples = int(len(
        training_ds[current_energies[0]][current_variables_of_interest[0]])/batch_size)
    for i in range(1, multiples):
        noise, noise_hyperparams = get_noise(
            current_energies, training_ds, batch_size, noise_size, conditions)
        temp_generated = model.predict([noise]+noise_hyperparams)

        gen_class_length = int(temp_generated.shape[0]/(len(current_energies)))

        for en in range(1, len(current_energies)+1):
            for num, var in enumerate(current_variables_of_interest):
                current_var = np.asarray(temp_generated)[:, num]
                gen_energies_var = current_var[(
                    en-1)*gen_class_length:en*gen_class_length]
                generated_ds[current_energies[en-1]][var] = np.concatenate(
                    [generated_ds[current_energies[en-1]][var], gen_energies_var])
    currentCond = []
    for en in generated_ds:
        currentCond.append(generated_ds[en][var])
    conditions.append(currentCond)
    allNorms.append(normalisation)

all_stuff={'data':conditions,'normalisation':allNorms}
pickle.dump( all_stuff, open("./final_result/testing_data.p", "wb" ) )
'''
masses=["1-5","2","2-5","3","4"]
for mass in masses:
    print("Current log-mass: %s"%(mass.replace('-','.')))
    all_stuff = pickle.load(
            open("./final_result/testing_data.p", "rb"))
    conditions=all_stuff['data']
    allNorms=all_stuff['normalisation']

    selected_energies=get_distributed_energies('sampling_distributions/Ar_c1dat_m'+mass+'.dat')

    #selected_energies = [str(e) for e in selected_energies]
    massString=mass.replace('-','.')
    for num,cond in enumerate(conditions):
        conditions[num]=[conditions[num][s]*allNorms[num][s] for s in selected_energies]


    # order is [s1,s2,f200]
    # This graph is f200 vs s1
    training_ds = getData([str(e) for e in selected_energies], ['s1','s2','f200like'])

    y=[]
    x=[]
    for i in range(len(selected_energies)):
        y.append(training_ds[i]['f200like'])
        x.append(training_ds[i]['s1'])
    y = np.concatenate(y)
    x = np.concatenate(x)
    h1_training,x_edges_1,y_edges_1,im= plt.hist2d(x, y,range= [[0, 600], [0.2, 1]],bins=[100,100],norm=mpl.colors.LogNorm(vmax=1000),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.ylabel(r"$f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel("$S_1$(NPE)", size=11, labelpad=5)
    plt.title(r"G4DS with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_g4_f200_vs_s1.png')
    plt.close()

    plt.figure()
    y = np.array(conditions[-1])
    x = np.array(conditions[0])
    h1_gan,x_edges_1,y_edges_1,im= plt.hist2d(np.concatenate(x), np.concatenate(y),range= [[0, 600], [0.2, 1]],bins=[100,100],norm=mpl.colors.LogNorm(vmax=1000),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.ylabel(r"$f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.title(r"GAN with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_gan_f200_vs_s1.png')
    plt.close()

    plt.figure()
    y=[]
    x=[]
    for i in range(len(selected_energies)):
        x.append(training_ds[i]['s1'])
        y.append(np.log(np.divide(np.array(training_ds[i]['s2']),np.array(training_ds[i]['s1']))))

    y = np.concatenate(y)
    x = np.concatenate(x)

    h2_training,x_edges,y_edges,im= plt.hist2d(x, y,range= [[0, 600], [0, 5]],bins=[100,100],norm=mpl.colors.LogNorm(vmax=1000),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(r"G4DS with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_g4_s1_over_s2_vs_s1.png')
    plt.close()

    x = np.array(conditions[0])
    y = np.log(np.divide(np.array(conditions[1]),np.array(conditions[0])))
    y = np.concatenate(y)
    x = np.concatenate(x)

    h2_gan,x_edges_2,y_edges_2,im = plt.hist2d(x, y,range= [[0, 600], [0, 5]],bins=[100,100],norm=mpl.colors.LogNorm(vmax=1000),cmap=plt.get_cmap("jet"))
    plt.colorbar(im)
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(r"GAN with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_gan_s1_over_s2_vs_s1.png')
    plt.close()

    diff = (h2_gan - h2_training)
    custom_cmap=plt.get_cmap("seismic")
    #custom_cmap.set_bad('white')
    #diff = np.ma.masked_equal(diff, 0)
    h2_diff= plt.pcolormesh(x_edges_2, y_edges_2,diff.T,norm=mpl.colors.Normalize(vmin=-1000,vmax=1000),cmap=custom_cmap)
    cbar= plt.colorbar(h2_diff)
    #cbar.ax.set_yticklabels([r'$-10^3$',r'$-10^2$',r'$10^2$',r'$10^3$'])
    plt.xlabel(r"Difference in $S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"Difference in log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(r"GAN4DS-G4DS with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_difference_s1_over_s2_vs_s1.png')
    plt.close()
    '''
    ratio = np.divide(h2_gan,h2_training , out=np.zeros_like(h2_gan), where=h2_training!=0)
    #ratio = np.ma.masked_equal(ratio, 1)
    #linthresh=0.03, linscale=0.03 ,vmin=0, vmax=1000, base=10
    h2_ratio= plt.pcolormesh(x_edges_2, y_edges_2,ratio.T,norm=mpl.colors.LogNorm(),cmap=plt.get_cmap("jet"))
    #ticks=[1000,2, 1, 10e-1]
    cbar= plt.colorbar(h2_ratio)
    #cbar.ax.set_yticklabels(['1000','2','1','10e-1'])
    plt.xlabel(r"Ratio in $S_1$(NPE)", size=11, labelpad=5)
    plt.ylabel(r"Ratio in log$(S_2/S_1)$", size=11, labelpad=5)
    plt.title(rf"GAN4DS/G4 with log$(m)=%s$ for log$(S_2/S_1)$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_ratio_s1_over_s2_vs_s1.png')
    plt.close()
    '''
    diff = (h1_gan - h1_training)
    custom_cmap=plt.get_cmap("seismic")
    #custom_cmap.set_bad('white')
    #diff = np.ma.masked_equal(diff, 0)
    h1_diff= plt.pcolormesh(x_edges_1, y_edges_1,diff.T,norm=mpl.colors.Normalize(vmin=-1000,vmax=1000),cmap=custom_cmap)
    cbar= plt.colorbar(h1_diff)
    #cbar.ax.set_yticklabels([r'$-10^3$',r'$-10^2$',r'$10^2$',r'$10^3$'])
    plt.ylabel(r"Difference in $f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel(r"Difference in $S_1$(NPE)", size=11, labelpad=5)
    plt.title(r"GAN4DS-G4 with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_difference_f200_vs_s1.png')
    plt.close()
    plt.figure()

    '''
    custom_cmap=plt.get_cmap("seismic")
    custom_cmap.set_bad('white')
    ratio = np.divide(h1_gan,h1_training , out=np.zeros_like(h1_gan), where=h1_training!=0)
    ratio = np.ma.masked_equal(ratio, 0)
    #linthresh=0.03, linscale=0.03 ,vmin=0, vmax=1000, base=10
    h1_ratio= plt.pcolormesh(x_edges_1, y_edges_1,ratio.T,norm=mpl.colors.LogNorm(),cmap=plt.get_cmap("seismic"))
    #ticks=[1000,2, 1, 10e-1]
    cbar= plt.colorbar(h2_ratio)
    #cbar.ax.set_yticklabels(['1000','2','1','10e-1'])
    plt.ylabel(r"$f_{200}$", size=11, labelpad=5, rotation="vertical")
    plt.xlabel(r"$S_1$(NPE)", size=11, labelpad=5)
    plt.title(rf"GAN4DS/G4 with log$(m)=%s$ for $f_{200}$ vs $S_1$"%(massString))
    plt.savefig('./final_result/'+massString+'_ratio_f200_vs_s1.png')
    plt.close()
    plt.figure()
    '''