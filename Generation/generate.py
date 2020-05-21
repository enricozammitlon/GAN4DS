import tensorflow as tf
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def getData(energies, variables_of_interest):
    allTrees = {}
    for energy in energies:
        allTrees[energy] = pickle.load(
            open("../Standalone/in/pickles/outRun_"+energy+".p", "rb"))
    result = {}
    for key, value in allTrees.items():
        filt_value = {k2: v2 for k2,
                      v2 in value.items() if k2 in variables_of_interest}
        if(filt_value):
            result[key] = filt_value
    return result


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

stream = open("../G4_RUNS/serial_architecture/working_3D_cgan_s1_s2_f200"+"/sessions/" +
              currentSession+"/layouts/"+"config.yaml", "r+")
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
startTime = datetime.now()
for var in variables_of_interest:
    model = tf.keras.models.load_model("../G4_RUNS/serial_architecture/working_3D_cgan_s1_s2_f200"+"/sessions/" +
                                       currentSession+"/model/"+var+"_weights.model")
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
np.savetxt("generated_s1.csv",conditions[0],delimiter=',')
np.savetxt("generated_s2.csv",conditions[1],delimiter=',')
np.savetxt("generated_f200.csv",conditions[2],delimiter=',')
pickle.dump(conditions,open('generated_data.p','wb'))
print(datetime.now() - startTime)
