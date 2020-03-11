from tensorflow.keras.layers     import BatchNormalization, Dense, Dropout, Input, LeakyReLU, Concatenate
from tensorflow.keras.models     import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
from os import makedirs
from os.path import exists

class NeuralNetworkLayout(object):
    def __init__(self,gtrate,dtrate,gnodes,dnodes,gdo,ddo,gbeta1,dbeta1,dimensions,noise,echeck,fileDir='out'):
        self.dir_output=fileDir
        self.gan_training_rate=gtrate
        self.d_training_rate=dtrate
        self.gan_nodes=gnodes
        self.d_nodes=dnodes
        self.gan_dropout=gdo
        self.d_dropout=ddo
        self.gan_beta1=gbeta1
        self.d_beta1=dbeta1
        self.dimensionality=dimensions
        self.noise=noise
        self.epochCheck=echeck
        self.logdir=self.dir_output+'/model/logs/'
        self.d=None
        self.g=None
        self.gan=None


    def compileGenerator(self,verbose=False):
        #Input of noise to generator
        noise_in = Input((self.noise,))
        g1 = Dense(self.gan_nodes, activation="relu")(noise_in)
        g1 = Dropout(self.gan_dropout)(g1)
        g1 = Dense(self.gan_nodes, activation="relu")(g1)

        #Input of condition(eg. energy)
        hyper_in = Input((1,))
        g2 = Dense(self.gan_nodes, activation="relu")(hyper_in)
        g2 = Dropout(self.gan_dropout)(g2)
        g2 = Dense(self.gan_nodes, activation="relu")(g1)

        gc = Concatenate()([g1, g2])
        gc = BatchNormalization()(gc)

        gc = Dropout(self.gan_dropout)(gc)
        gc = Dense(self.gan_nodes, activation="relu")(gc)
        gc = Dropout(self.gan_dropout)(gc)
        gc = Dense(self.gan_nodes, activation="relu")(gc)

        gc = Dense(self.dimensionality, activation="linear")(gc)

        gc = Model(name="Generator", inputs=[noise_in, hyper_in], outputs=[gc])

        if(verbose):
            gc.summary()

        return gc

    def compileDiscriminator(self,verbose=False):
        #Input from generator
        d1_in = Input((self.dimensionality,))
        d1 = Dense(self.d_nodes, activation="relu")(d1_in)
        d1 = Dropout(self.d_dropout)(d1)

        #Input of condition(eg. energy)
        hyper_in = Input((1,))
        d2 = Dense(self.d_nodes, activation="relu")(hyper_in)
        d2 = Dropout(self.d_dropout)(d2)

        dc = Concatenate()([d1, d2])

        dc = Dense(self.d_nodes , activation="relu")(dc)
        dc = Dense(self.d_nodes , activation="relu")(dc)
        dc = Dropout(self.d_dropout)(dc)
        dc = LeakyReLU(0.2)(dc)
        dc = Dense(self.d_nodes , activation="relu")(dc)
        dc = Dropout(self.self.d_dropout)(dc)
        dc = LeakyReLU(0.2)(dc)
        dc = Dropout(self.self.d_dropout)(dc)

        dc = Dense(2, activation="softmax")(dc)

        dc = Model(name="Discriminator", inputs=[d1_in, hyper_in], outputs=[dc])
        dc.compile(loss="categorical_crossentropy", optimizer=Adam(self.d_training_rate, beta_1=self.d_beta1), metrics=["accuracy"])

        if(verbose):
            dc.summary()

        return dc

    def compileGAN(self,verbose=False):
        self.d=self.compileDiscriminator(verbose)
        self.g=self.compileGenerator(verbose)

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