import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from progress.bar import IncrementalBar


class NeuralNetworkTraining:

    def __init__(self,layout,tds,fileDir,unseends=[],plotRes=200,epochs=25,epochCheck=1,bs=1000,ns=1000,unseenE='',save=True):
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

        self.training_ds=tds
        self.unseen_ds=unseends
        self.normalisation={}
        self.generated_ds=None
        self.energies=list(tds.keys())
        self.variables_of_interest=list(tds[self.energies[0]].keys())
        self.unseen_energy=unseenE
        self.d_loss=[]
        self.d_acc=[]
        self.reel_images=[]
        self.last=False

        self.normaliseData()
        
    def normaliseData(self):
        #+1 is for the unseen dataset
        for en in range(len(self.training_ds)):
            self.normalisation[self.energies[en]]={}
            for var in self.variables_of_interest:
                temp_maximum=np.max(self.training_ds[self.energies[en]][var])
                self.training_ds[self.energies[en]][var]/=temp_maximum
                self.normalisation[self.energies[en]][var]=temp_maximum
        self.normalisation[self.unseen_energy]={}

        for unseen in self.unseen_ds:
            for var in self.variables_of_interest:
                temp_maximum=np.max(unseen[self.unseen_energy][var])
                unseen[self.unseen_energy][var]/=temp_maximum
                self.normalisation[self.unseen_energy][var]=temp_maximum

        

    #Returns true if genHist < trainHist
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

        hyperparams = np.concatenate([np.full(fill_value=float(a), shape=(self.batch_size, 1)) for a in self.energies])

        return batches, hyperparams


    def get_noise (self) :
        hyperparams = np.concatenate([np.full(fill_value=float(a), shape=(self.batch_size, 1)) for a in self.energies])
        noise = np.random.normal(size=(len(self.training_ds)*self.batch_size, self.noise_size))
        return noise, hyperparams
    
    def get_plot_title(self):
        title="Variables: "
        for var in self.variables_of_interest:
            title+=var+", "
        title += "from a "+str(self.dimensionality)+"D"
        #Add check if GAN or cGAN
        title+=" cGAN at Epoch "+str(self.epoch+1)+" and batch size "+str(self.batch_size)
        return title

    def visualiseCurrentEpoch(self,training_ds,generated_ds,energies):

        histograms=[]
        
        fig, axes = plt.subplots(1,self.dimensionality,squeeze=False)
        
        fh=10
        fw=10*self.dimensionality

        fig.set_figheight(fh)
        fig.set_figwidth(fw)

        fig.suptitle(self.get_plot_title())
        cmap = matplotlib.cm.get_cmap('tab10')
        norm = matplotlib.colors.Normalize(vmin=float(energies[0]), vmax=float(energies[-1]))

        for num,var in enumerate(self.variables_of_interest):
            current_histograms=[]
            for en in range(0,len(energies)):
                true_histogram=axes[0,num].hist(training_ds[energies[en]][var]*self.normalisation[energies[en]][var],density = True, bins = self.plot_resolution, color=cmap(norm(float(energies[en]))), alpha = 0.4, label = "G4DS "+energies[en]+"keV")
                generated_histogram=axes[0,num].hist(generated_ds[energies[en]][var]*self.normalisation[energies[en]][var],density = True, bins = self.plot_resolution, color=cmap(norm(float(energies[en]))), alpha = 1, label = "Gan4DS "+energies[en]+"keV")
                current_histograms.append([true_histogram,generated_histogram])

            for unseen in self.unseen_ds:
                #same but add the unseen energies
                true_histogram=axes[0,num].hist(self.unseen_ds[self.unseen_energy][var]*self.normalisation[self.unseen_energy][var],density = True, bins = self.plot_resolution, color=cmap(norm(float(self.unseen_energy))), alpha = 0.4, label = "G4DS "+self.unseen_energy+"keV")
                generated_histogram=axes[0,num].hist(generated_ds[self.unseen_energy][var]*self.normalisation[self.unseen_energy][var],density = True, bins = self.plot_resolution, color=cmap(norm(float(self.unseen_energy))), alpha = 1, label = "Gan4DS "+self.unseen_energy+"keV")
                current_histograms.append([true_histogram,generated_histogram])

            histograms.append(current_histograms)

            axes[0,num].set_xlabel(self.variables_of_interest[num].capitalize(), size=11, labelpad=5)
            axes[0,num].set_ylabel(r"$\rho\left(x\right)$", size=11, labelpad=5, rotation="horizontal")
            axes[0,num].legend(loc="upper right", fontsize=11)
        if(self.save):
            plt.savefig(self.dir_output+'/figures/all/reel_f_'+str(int(self.epoch+1/self.epochCheck))+".png")
        if(self.last):
            plt.savefig(self.dir_output+'/figures/final_product/'+str(self.dimensionality)+"D_cGAN"+".png")
        
        true_histograms=[]
        generated_histograms=[]
        for a in range(len(self.variables_of_interest)):
            true_histograms.append([b[0] for b in histograms[a]])
            generated_histograms.append([b[1] for b in histograms[a]])

        if(self.metric(true_histograms,generated_histograms)):
            print('\tNew minimum found here.')
            self.gan.save(self.dir_output+'/model/'+"weights.h5")

        plt.close(fig)

    def initiateTraining(self):
        bar = IncrementalBar('Training', max=self.epochs)
        for e in range(self.epochs) :
            bar.next()
            self.epoch=e
            noise, noise_hyperparams    = self.get_noise()
            batch_DS, batch_hyperparams = self.get_train_data()
            self.generated_ds   = self.layout.g.predict([noise, noise_hyperparams],batch_size=self.batch_size)
            real_label  = np.array([[1., 0.] for i in range(len(self.energies)*self.batch_size)])
            fake_label  = np.array([[0., 1.] for i in range(len(self.energies)*self.batch_size)])
            train_label = np.array([[1., 0.] for i in range(len(self.energies)*self.batch_size)])
            X  = np.concatenate([batch_DS  , self.generated_ds    ])
            Xh = np.concatenate([batch_hyperparams  , noise_hyperparams    ])
            Y = np.concatenate([real_label, fake_label])
            W = np.concatenate([np.ones(shape=(len(self.energies)*self.batch_size,)), np.full(fill_value=1, shape=(len(self.energies)*self.batch_size,))])
            
            self.layout.d.trainable = True
            d_loss, d_acc = self.layout.d.train_on_batch([X, Xh], Y, sample_weight=W)

                
            self.layout.d.trainable = False
            self.gan.train_on_batch([noise, noise_hyperparams], train_label)
            
            if e == 0 or (e+1) % self.epochCheck == 0 :
                '''
                #Old Working Code

                noise, noise_hyperparams = self.get_noise()
                self.generated_ds = self.layout.g.predict([noise, noise_hyperparams])
                hyperparams = np.full(fill_value=self.unseen_energy, shape=(self.batch_size, 1))
                z = np.random.normal(size=(self.batch_size, self.noise_size))

                generated_unseen_energy = self.layout.g.predict([z, hyperparams])
                self.generated_ds=np.concatenate([self.generated_ds,generated_unseen_energy],0)
                '''
                self.generated_ds={}
                
                noise, noise_hyperparams = self.get_noise()
                temp_generated = self.layout.g.predict([noise, noise_hyperparams])
                
                for unseen in self.unseen_ds:
                    hyperparams = np.full(fill_value=self.unseen_energy, shape=(self.batch_size, 1))
                    z = np.random.normal(size=(self.batch_size, self.noise_size))
                    generated_unseen_energy = self.layout.g.predict([z, hyperparams])
                
                gen_class_length = int(temp_generated.shape[0]/(len(self.energies)))

                for en in range(1,len(self.energies)+1):
                    self.generated_ds[self.energies[en-1]]={}
                    for num,var in enumerate(self.variables_of_interest):
                        current_var = np.asarray(temp_generated)[:,num]
                        gen_energies_var=current_var[(en-1)*gen_class_length:en*gen_class_length]
                        self.generated_ds[self.energies[en-1]][var]=gen_energies_var
                    for unseen in self.unseen_ds:
                        self.generated_ds[self.unseen_energy][var]=np.asarray(generated_unseen_energy)[:,num]

                '''
                for num,var in enumerate(self.variables_of_interest):
                    current_var = np.asarray(temp_generated)[:,num]
                    self.generated_ds[var]={}
                    for en in range(1,len(self.energies)+1):
                        gen_energies_var=current_var[(en-1)*gen_class_length:en*gen_class_length]
                        self.generated_ds[var][self.energies[en-1]]=gen_energies_var
                    
                    for unseen in self.unseen_ds:
                        self.generated_ds[var][self.unseen_energy]=np.asarray(generated_unseen_energy)[:,num]
                '''

                multiples=int(len(self.training_ds[self.energies[0]][self.variables_of_interest[0]])/self.batch_size)
                for i in range(1,multiples) :
                    noise, noise_hyperparams = self.get_noise()
                    temp_generated = self.layout.g.predict([noise, noise_hyperparams])
                    
                    for unseen in self.unseen_ds:
                        hyperparams = np.full(fill_value=self.unseen_energy, shape=(self.batch_size, 1))
                        z = np.random.normal(size=(self.batch_size, self.noise_size))
                        generated_unseen_energy = self.layout.g.predict([z, hyperparams])

                    gen_class_length = int(temp_generated.shape[0]/(len(self.energies)))

                    for en in range(1,len(self.energies)+1):
                        for num,var in enumerate(self.variables_of_interest):
                            current_var = np.asarray(temp_generated)[:,num]
                            gen_energies_var=current_var[(en-1)*gen_class_length:en*gen_class_length]
                            self.generated_ds[self.energies[en-1]][var]=np.concatenate([self.generated_ds[self.energies[en-1]][var],gen_energies_var])
                        for unseen in self.unseen_ds:
                            self.generated_ds[self.unseen_energy][var]=np.concatenate([self.generated_ds[var][self.unseen_energy],np.asarray(generated_unseen_energy)[:,num]])
                
                if(self.epoch+self.epochCheck>=self.epochs):
                    self.last=True
                
                #Set instead of 3 the number of candidates to show
                indexes=np.round(np.linspace(0, len(self.energies) - 1, 3)).astype(int)
                selected_energies=[self.energies[i] for i in range(len(self.energies)) if i in indexes]
                selected_training_ds=dict(filter(lambda elem: elem[0] in selected_energies,self.training_ds.items()))
                selected_generated_ds=dict(filter(lambda elem: elem[0] in selected_energies,self.generated_ds.items()))
                
                self.visualiseCurrentEpoch(selected_training_ds,selected_generated_ds,selected_energies)

            self.all_epochs.append(e)
            self.d_loss.append(d_loss)
            self.d_acc.append(d_acc)

        bar.finish()