# Probably the most important script is where training occurs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from progress.bar import IncrementalBar
import io
from statistics import mean,variance
from random import randint

class NeuralNetworkTraining:

    def __init__(self,layout,tds,conditions,fileDir,epochs,epochCheck,filewriter,plotRes=200,bs=1000,ns=1000,save=True):
        self.plot_resolution=plotRes
        self.dir_output=fileDir
        self.all_epochs=[]
        self.epoch=0
        self.epochs=epochs
        self.epochCheck=epochCheck
        self.batch_size=bs
        self.minloss=1e6
        self.save=save
        self.layout=layout
        self.dimensionality=layout.dimensionality
        self.gan=layout.gan
        self.noise_size=self.layout.noise
        self.conditions=conditions
        self.training_ds=tds
        self.generated_ds=None
        self.energies=list(tds.keys())
        self.variables_of_interest=list(tds[self.energies[0]].keys())
        self.d_loss=[]
        self.d_acc=[]
        self.d_x=[]
        self.d_x2 = []
        self.reel_images=[]
        self.last=False
        self.file_writer=filewriter
        self.normalisation={}
        #self.normaliseData()
        self.final_produced={}
        self.mockNormaliseData()
    # One can use this to not normalize
    # Training data before being fed to the GAN
    def mockNormaliseData(self):
        for en in range(len(self.training_ds)):
            self.normalisation[self.energies[en]]={}
            for var in self.variables_of_interest:
                self.normalisation[self.energies[en]][var]=1

    def normaliseData(self):
        for en in range(len(self.training_ds)):
            self.normalisation[self.energies[en]]={}
            for var in self.variables_of_interest:
                temp_maximum=np.max(self.training_ds[self.energies[en]][var])
                self.training_ds[self.energies[en]][var]/=temp_maximum
                self.normalisation[self.energies[en]][var]=temp_maximum

    #Returns true if this generator is better than the last one checked
    def metric(self,genHist,trainHist):

        #Check difference
        sum_diff=0
        for var in range(len(genHist)):#Each row is a variable, each column is an energy
            for en in range(len(genHist[var])):
                current_diff=trainHist[var][en][0]-genHist[var][en][0]
                for diff in current_diff:
                    sum_diff+=abs(diff)

        if sum_diff < self.minloss:
            #save weights
            #print("SAVING")
            #gc.save("2D_s1_s2.h5")
            self.minloss = sum_diff
            return True

        return False

    # Splits training data in batches together with the conditions (called hyperparams here)
    def get_train_data(self) :
        batches=[]
        for en in self.energies:
            sub_batches=[]
            for var in self.variables_of_interest:
                random_num= np.random.randint(0, len(self.training_ds[en][var]), self.batch_size)
                sub_batches.append(self.training_ds[en][var][random_num].reshape(self.batch_size))
            single_batch=np.array(np.matrix(sub_batches).T)
            batches.append(single_batch)
        batches=np.concatenate(batches)

        hyperparams = np.concatenate([np.full(fill_value=float(1/a), shape=(self.batch_size, 1)) for a in self.energies])
        all_hyperparams=[hyperparams]

        for param in self.conditions:
            currentParam=[]
            for en in range(len(self.energies)):
                random_num= np.random.randint(0, len(param[en]), self.batch_size)
                currentParam.append(param[en][random_num].reshape(self.batch_size,1))
            currentParam=np.concatenate(currentParam)
            all_hyperparams+=[currentParam]
        return batches, all_hyperparams

    # Create noise vectors to be fed to generator. These need to be in the same
    # size and shape of training DS
    def get_noise (self) :
        hyperparams = np.concatenate([np.full(fill_value=float(1/a), shape=(self.batch_size, 1)) for a in self.energies])
        all_hyperparams=[hyperparams]
        noise = np.random.normal(size=(len(self.training_ds)*self.batch_size, self.noise_size))
        for param in self.conditions:
            currentParam=[]
            for en in range(len(self.energies)):
                random_num= np.random.randint(0, len(param[en]), self.batch_size)
                currentParam.append(param[en][random_num].reshape(self.batch_size,1))
            currentParam=np.concatenate(currentParam)
            all_hyperparams+=[currentParam]
        return noise, all_hyperparams

    def get_plot_title(self):
        title="Variables: "
        for var in self.variables_of_interest:
            title+=var+", "
        title += "from a "+str(self.dimensionality)+"D"
        title+=" cGAN at Epoch "+str(self.epoch+1)+" and batch size "+str(self.batch_size)
        return title

    def getMoment1(self, y_true, y_pred):
        return np.mean(y_pred)/np.mean(y_true)

    def getMoment2(self, y_true, y_pred):
        return np.var(y_pred)/np.var(y_true)

    def named_logs(self,model,logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result
    # Creates plots used to compare n selected energies from generator w/ training data
    def visualiseCurrentEpoch(self,training_ds,generated_ds,energies,all_gends):
        histograms=[]
        plt.figure(figsize=(10,10),clear=True)

        plt.title(self.get_plot_title())
        cmap = matplotlib.cm.get_cmap('tab10')
        norm = matplotlib.colors.Normalize(vmin=float(energies[0]), vmax=float(energies[-1]))
        for num,var in enumerate(self.variables_of_interest):
            current_histograms=[]
            range_min=min(training_ds[energies[0]][var]*self.normalisation[energies[0]][var])
            range_max=max(training_ds[energies[-1]][var]*self.normalisation[energies[-1]][var])
            bins=np.linspace(range_min, range_max, self.plot_resolution)
            for en in range(0,len(energies)):
                true_histogram=plt.hist(np.clip(training_ds[energies[en]][var]*self.normalisation[energies[en]][var], bins[0], bins[-1]), bins=bins,density = True,color=cmap(norm(float(energies[en]))), alpha = 0.4, label = "G4DS "+energies[en]+"keV")
                generated_histogram=plt.hist(np.clip(generated_ds[energies[en]][var]*self.normalisation[energies[en]][var], bins[0], bins[-1]), bins=bins,density = True, color=cmap(norm(float(energies[en]))), alpha = 1, label = "Gan4DS "+energies[en]+"keV")
                current_histograms.append([true_histogram,generated_histogram])
            histograms.append(current_histograms)

            plt.xlabel(self.variables_of_interest[num].capitalize(), size=11, labelpad=5)
            plt.ylabel(r"$\rho\left(x\right)$", size=11, labelpad=5, rotation="horizontal")
            plt.legend(loc="upper right", fontsize=11)
        if(self.last):
            plt.savefig(self.dir_output+'/figures/final_product/'+self.variables_of_interest[-1]+'_'+str(self.dimensionality)+"D_cGAN"+".png")

        if(self.save):
            plt.savefig(self.dir_output+'/figures/all/reel_f_'+str(int((self.epoch+1)/self.epochCheck))+".png")
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()

            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            with self.file_writer.as_default():
              tf.summary.image("Histogram "+str(int((self.epoch+1)/self.epochCheck)), image, step=self.epoch)

        true_histograms=[]
        generated_histograms=[]
        for a in range(len(self.variables_of_interest)):
            true_histograms.append([b[0] for b in histograms[a]])
            generated_histograms.append([b[1] for b in histograms[a]])
        # Check if this genretor is better than previous and if so save weights
        if(self.metric(true_histograms,generated_histograms)):
            print('\tNew minimum found here.')
            self.final_produced=all_gends
            self.layout.g.save(self.dir_output+'/model/'+self.variables_of_interest[-1]+"_weights.model")

    def initiateTraining(self):
        bar = IncrementalBar('Training', max=self.epochs)
        self.d_x=[]
        self.d_x2=[]
        for e in range(self.epochs) :
            bar.next()
            self.epoch=e
            noise, noise_hyperparams    = self.get_noise()
            batch_DS, batch_hyperparams = self.get_train_data()
            self.generated_ds   = self.layout.g.predict([noise]+noise_hyperparams,batch_size=self.batch_size)
            real_label  = np.array([[1., 0.] for i in range(len(self.energies)*self.batch_size)])
            fake_label  = np.array([[0., 1.] for i in range(len(self.energies)*self.batch_size)])
            train_label = np.array([[1., 0.] for i in range(len(self.energies)*self.batch_size)])
            X  = np.concatenate([batch_DS  , self.generated_ds    ])
            all_Xh=[X]
            for num in range(len(noise_hyperparams)):
                all_Xh.append(np.concatenate([batch_hyperparams[num]  , noise_hyperparams[num]    ]))
            Y = np.concatenate([real_label, fake_label])
            W = np.concatenate([np.ones(shape=(len(self.energies)*self.batch_size,)), np.full(fill_value=1, shape=(len(self.energies)*self.batch_size,))])

            self.layout.d.trainable = True
            d_loss, d_acc = self.layout.d.train_on_batch(all_Xh, Y, sample_weight=W)


            self.layout.d.trainable = False
            logs = self.gan.train_on_batch([noise]+noise_hyperparams, train_label)

            self.layout.tensorboard.on_epoch_end(self.epoch, self.named_logs(self.gan, logs))

            if e == 0 or (e+1) % self.epochCheck == 0 :
                self.generated_ds={}

                noise, noise_hyperparams = self.get_noise()
                temp_generated = self.layout.g.predict([noise]+noise_hyperparams)

                gen_class_length = int(temp_generated.shape[0]/(len(self.energies)))

                for en in range(1,len(self.energies)+1):
                    self.generated_ds[self.energies[en-1]]={}
                    for num,var in enumerate(self.variables_of_interest):
                        current_var = np.asarray(temp_generated)[:,num]
                        gen_energies_var=current_var[(en-1)*gen_class_length:en*gen_class_length]
                        self.generated_ds[self.energies[en-1]][var]=gen_energies_var

                multiples=int(len(self.training_ds[self.energies[0]][self.variables_of_interest[0]])/self.batch_size)
                for i in range(1,multiples) :
                    noise, noise_hyperparams = self.get_noise()
                    temp_generated = self.layout.g.predict([noise]+noise_hyperparams)

                    gen_class_length = int(temp_generated.shape[0]/(len(self.energies)))

                    for en in range(1,len(self.energies)+1):
                        for num,var in enumerate(self.variables_of_interest):
                            current_var = np.asarray(temp_generated)[:,num]
                            gen_energies_var=current_var[(en-1)*gen_class_length:en*gen_class_length]
                            self.generated_ds[self.energies[en-1]][var]=np.concatenate([self.generated_ds[self.energies[en-1]][var],gen_energies_var])

                if(self.epoch+self.epochCheck>=self.epochs):
                  self.last=True
                #Set instead of 3 the number of candidates to show
                indexes=np.round(np.linspace(0, len(self.energies) - 1, 3)).astype(int)
                selected_energies=[self.energies[i] for i in range(len(self.energies)) if i in indexes]
                selected_training_ds=dict(filter(lambda elem: elem[0] in selected_energies,self.training_ds.items()))
                selected_generated_ds=dict(filter(lambda elem: elem[0] in selected_energies,self.generated_ds.items()))
                for num,var in enumerate(self.variables_of_interest):
                    random_energy=  randint(0, len(selected_energies)-1)
                    true_d=selected_training_ds[selected_energies[random_energy]][var]*self.normalisation[selected_energies[random_energy]][var]
                    false_d=selected_generated_ds[selected_energies[random_energy]][var]*self.normalisation[selected_energies[random_energy]][var]
                    self.d_x.append(self.getMoment1(true_d,false_d))
                    self.d_x2.append(self.getMoment2(true_d,false_d))
                self.visualiseCurrentEpoch(selected_training_ds,selected_generated_ds,selected_energies,self.generated_ds)

            self.all_epochs.append(e)
            self.d_loss.append(d_loss)
            self.d_acc.append(d_acc)


        self.layout.tensorboard.on_train_end(None)
        bar.finish()
        return self.final_produced
