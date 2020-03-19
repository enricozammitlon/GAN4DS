from tensorflow.keras.layers     import BatchNormalization, Dense, Dropout, Input, LeakyReLU, Concatenate
from tensorflow.keras.models     import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
from os import makedirs
from os.path import exists
import yaml
from distutils.dir_util import copy_tree

class NeuralNetworkLayout(object):
    def __init__(self,gtrate,dtrate,gbeta1,dbeta1,dimensions,noise,echeck,overrides,layoutDir='layouts',fileDir='out'):
        self.dir_output=fileDir
        self.gan_training_rate=gtrate
        self.d_training_rate=dtrate
        self.gan_beta1=gbeta1
        self.d_beta1=dbeta1
        self.dimensionality=dimensions
        self.noise=noise
        self.epochCheck=echeck
        self.logdir=self.dir_output+'/model/logs/'
        self.layout_dir=layoutDir
        self.overrides=overrides
        self.d=None
        self.g=None
        self.gan=None

    def compileComponent(self,verbose=False,component_type=None):
        stream = open(self.layout_dir+"/discriminator_layout.yaml","r+")
        data = yaml.load(stream)
        g1_instructions= data['join'][0]
        g2_instructions= data['join'][1]
        gc_instructions = data['layers']
        if(component_type=='discriminator'):
            g_in_1 = Input((self.dimensionality,))
            g_in_2 = Input((1,))
        elif(component_type=='generator'):
            g_in_1 = Input((self.noise,))
            g_in_2 = Input((1,))

        g1 = g_in_1
        for layer_info in g1_instructions['layers']:
            if(self.overrides.get('activation')):
                activation = self.overrides.get('activation')
            else:
                activation = layer_info.get('activation')
            if(component_type=='discriminator' and self.overrides.get('d_nodes')):
                units = self.overrides.get('d_nodes')
            elif(component_type=='generator' and self.overrides.get('g_nodes')):
                units = self.overrides.get('g_nodes')
            else:
                units =  layer_info.get('nodes')
            if(self.overrides.get('dropout_amount')):
                rate = self.overrides.get('dropout_amount')
            else:
                rate = layer_info.get('dropout_amount')
            if(self.overrides.get('leaky_amount')):
                alpha = self.overrides.get('leaky_amount')
            else:
                alpha = layer_info.get('leaky_amount')
            if(layer_info['layer_type']=='dense'):
                g1=Dense(units=units,activation=activation)(g1)
            elif(layer_info['layer_type']=='dropout'):
                g1=Dropout(rate=rate)(g1)
            elif(layer_info['layer_type']=='selu'):
                g1=LeakyReLU(alpha=alpha)(g1)
            elif(layer_info['layer_type']=='batchnorm'):
                g1=BatchNormalization()(g1)

        g2 = g_in_2
        for layer_info in g2_instructions['layers']:
            if(self.overrides.get('activation')):
                activation = self.overrides.get('activation')
            else:
                activation = layer_info.get('activation')
            if(component_type=='discriminator' and self.overrides.get('d_nodes')):
                units = self.overrides.get('d_nodes')
            elif(component_type=='generator' and self.overrides.get('g_nodes')):
                units = self.overrides.get('g_nodes')
            else:
                units =  layer_info.get('nodes')
            if(self.overrides.get('dropout_amount')):
                rate = self.overrides.get('dropout_amount')
            else:
                rate = layer_info.get('dropout_amount')
            if(self.overrides.get('leaky_amount')):
                alpha = self.overrides.get('leaky_amount')
            else:
                alpha = layer_info.get('leaky_amount')
            if(layer_info['layer_type']=='dense'):
                g2=Dense(units=units,activation=activation)(g2)
            elif(layer_info['layer_type']=='dropout'):
                g2=Dropout(rate=rate)(g2)
            elif(layer_info['layer_type']=='selu'):
                g2=LeakyReLU(alpha=alpha)(g2)
            elif(layer_info['layer_type']=='batchnorm'):
                g2=BatchNormalization()(g2)

        gc = Concatenate()([g1, g2])

        for layer_info in gc_instructions:
            if(self.overrides.get('activation')):
                activation = self.overrides.get('activation')
            else:
                activation = layer_info.get('activation')
            if(component_type=='discriminator' and self.overrides.get('d_nodes')):
                units = self.overrides.get('d_nodes')
            elif(component_type=='generator' and self.overrides.get('g_nodes')):
                units = self.overrides.get('g_nodes')
            else:
                units =  layer_info.get('nodes')
            if(self.overrides.get('dropout_amount')):
                rate = self.overrides.get('dropout_amount')
            else:
                rate = layer_info.get('dropout_amount')
            if(self.overrides.get('leaky_amount')):
                alpha = self.overrides.get('leaky_amount')
            else:
                alpha = layer_info.get('leaky_amount')
            if(layer_info['layer_type']=='dense'):
                gc=Dense(units=units,activation=activation)(gc)
            elif(layer_info['layer_type']=='dropout'):
                gc=Dropout(rate=rate)(gc)
            elif(layer_info['layer_type']=='selu'):
                gc=LeakyReLU(alpha=alpha)(gc)
            elif(layer_info['layer_type']=='batchnorm'):
                gc=BatchNormalization()(gc)

        if(component_type=='generator'):
            gc = Dense(self.dimensionality, activation="linear")(gc)
            gc = Model(name="Generator", inputs=[g_in_1, g_in_2], outputs=[gc])
        elif(component_type=='discriminator'):
            gc = Dense(2, activation="softmax")(gc)
            gc = Model(name="Discriminator", inputs=[g_in_1, g_in_2], outputs=[gc])
            gc.compile(loss="categorical_crossentropy", optimizer=Adam(self.d_training_rate, beta_1=self.d_beta1), metrics=["accuracy"])

        if(verbose):
            gc.summary()
        return gc

    def compileGAN(self,verbose=False):
        self.d=self.compileComponent(verbose,component_type='discriminator')
        self.g=self.compileComponent(verbose,component_type='generator')

        hyper_in = Input((1,))
        noise_in = Input((self.noise,))

        gan_out = self.d([self.g([noise_in, hyper_in]), hyper_in])
        gan = Model([noise_in, hyper_in], gan_out, name="GAN")
        self.d.trainable = False
        gan.compile(loss="categorical_crossentropy", optimizer=Adam(self.gan_training_rate, beta_1=self.gan_beta1), metrics=["accuracy"])
        if(verbose):
            gan.summary()
        self.gan=gan

    def visualiseData(self,save=True):
        pass

    def saveHyperParameters(self):
        if not exists(self.dir_output+'/layouts/'):
            makedirs(self.dir_output+'/layouts/')
        copy_tree('./layouts/', self.dir_output+'/layouts/')

        if not exists(self.dir_output+'/model/'):
            makedirs(self.dir_output+'/model/')
            makedirs(self.dir_output+'/model/logs/')

        self.tensorboard = TensorBoard(
            log_dir=self.logdir,
            histogram_freq=self.epochCheck,
            batch_size=self.noise,
            write_graph=True,
            write_grads=True
            )

        self.tensorboard.set_model(self.gan)

        with open(self.dir_output+'/model/hyperparameters.gan4ds',"w") as f:
            allParams=[a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
            for param in allParams:
                f.writelines(param+"\t"+str(getattr(self,param))+"\n")