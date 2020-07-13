# MPhys Project
This is the repository for the work done as part of a Masters in Physics project
done by Enrico Zammit Lonardelli and Krishan Jethwa throughout 2019-2020.
For a pdf report check the Reports/Sem_2_Report folder.

## Installing and Running (Containerised)
### To start developing/analysing using docker:
* Download VS Code
* Download the ms-vscode-remote.remote-containers extension
* From the extension tab (or by shift+cmd P) click `Open folder in container`
* Find the GAN4DS repistory folder
* Wait for everything to laod and you can start developing

### To start training on the GPU cluster:
* Ssh with -L 16006:localhost:6006 in the gpu machine
* `mkdir /hepgpu<X>-data<Y>/<your-username>`
* `git clone git@github.com:enricozammitlon/GAN4DS.git`
* `cd /hepgpu<X>-data<Y>/<your-username>/GAN4DS`
* `git checkout deploys` or `git pull origin deploys` as appropriate
* `cd Training`
* ` rm -rf gan4ds.out gan4ds.err out`
* `chmod +x start_running.sh`
* `bash start_running.sh`
* Wait for it to finish, meanwhile open up on your local browser localhost:16006
  When the machine starts training (takes a while to setup) you will see this screen update
* For any problems check the gan4ds.err created in the Training folder
* After training is done, move from the out/ folder to the Saves folder otherwise
  your work will be deleted on the next run! Just move the contents of run1/ and
  name it to something reasonable under the right directory in Saves.

### To start the training runner using docker on any docker enabled machine:
* `git clone` this repository
* `docker pull enricozl/gan4ds:argan-runnerr`
* `docker run --mount type=bind,source=<Insert here absolute path>/GAN4DS/Training,target=/Training -p 0.0.0.0:6006:6006 enricozl/gan4ds:argan-runner`

[To be fixed, right now not working]
To start the trainer on the GPU cluster automatically add "[DEPLOY]" in the
commit message and push to the deploys branch. This will be picked up by the
actions and a job will deployed. Furthermore, running
`ssh -L 16006:localhost:6006` and the gpu cluster will allow to visit the local browser
on 6006 for an interactive tensorboard session.

## Installing and Running (Manual)
### 1. Installing Python3.6.x
   N.B Before doing this make sure you have bzip2-devel installed either as global or by installing it yourself somewhere        locally.
   1. Use the script in Installation/python3_install.sh - it will do everything for you
   2. Running `which python3` should present with the global install and your new python3.6.10 installation

 ### 2. Installing Nvidia Graphics
   1. Move to `cd /hepgpuX-dataY/<your-username>/`
   2. Make a directory called `mkdir CUDA`
   3. Move to the new directory `cd CUDA`
   4. Now download the RPM locally via `wget ftp://ftp.pbone.net/mirror/atrpms.net/sl6-x86_64/atrpms/stable/nvidia-                 graphics302.17-libs-302.17-147.el6.x86_64.rpm` . Again make sure you install the right version for the distro you are         installing it on. If it is different check [1] .
   5. Now use a tool to extract it via `rpm2cpio <whatever-the-rpm-is-called>.rpm | cpio -idmv`. Now you should have a               directory called usr.
   6. Move back to `cd /hepgpuX-dataY/<your-username>/`
   7. Export the location of your install via `export LD_LIBRARY_PATH=/hepgpuX-dataY/<your-username/CUDA/usr/lib64:$PATH`.

 ### 4. Installing Cudnn and TensorRT
   1. Firstly join the Nvidia Developer Program via https://developer.nvidia.com/
   2. Once you have created and confirmed your account go to https://developer.nvidia.com/nvidia-tensorrt-6x-download to             download the tensorrt tar.gz . It is important to note which version of tensorrt and CUDA you are downloading since the       next steps will require you to download the appropriate CUDA version.
   3. Once you complete the mandatory survey you and after accepting the terms you will be able to download the correct             package which you can find under 'Tar File Install Packages For Linux x86'. Download this locally.
   4. Now go to https://developer.nvidia.com/rdp/cudnn-download and repeat the same process to download the same version of         CUDA as the package for tensorrt you downloaded said. This time download the package called 'cuDNN Runtime Library for         RedHat/Centos X.x' again locally.
   5. After both files download, go to your local directory where these are downloaded and open a terminal for us to `scp`           files there. N.B Instructions for windows might vary and you might need to download PuTTY.
   6. To transfer the files, from your local terminal run `scp libcudnn<your version>.rpm <your-                                     username>@pc<20xx>.blackett.manchester.ac.uk:/hepgpuX-dataY/<your-username>/CUDA/`
   7. Again run `scp TensorRT<your-version>.tar.gz <your-username>@pc<20xx>.blackett.manchester.ac.uk:/hepgpuX-dataY/<your-         username>/CUDA/`
   8. When all this is done `ssh` into hepgpuX and go to `cd /hepgpuX-dataY/<your-username>/CUDA`
   9. Extract the TensorRT files via `tar xzvf TensorRT-<your-version>.tar.gz`
   10. Extract the rpm file using `rpm2cpio libcudnn<your version>.rpm | cpio -idmv`
   11. Export the location of your libcuda via `export LD_LIBRARY_PATH=/hepgpuX-dataY/<your-username/CUDA/usr/lib64:$PATH`
   12. Export the location of your TensorRT via `export LD_LIBRARY_PATH=/hepgpuX-dataY/<your-username/CUDA/TensorRt<your-            version>/lib:$PATH`

 ### 4. Creating a virtual environment and installing requirements
   1. Move to `cd /pc20xx-dataX/<your-username>/`
   2. Create the virtual environment via `python3 -m venv py3.6`
   3. Activate the environment via `source ./py3.6/bin/activate`
   4. Now move to the location of the python scripts `cd /pc20xx-dataX/<your-username>/mphys-project/Standalone`
   5. Install dependencies via `pip3 install -r requirements.txt`
   6. Install TensorRT dependencies by going to `cd /pc20xx-dataX/<your-username>/CUDA/TensorRT<your-version>/`
   7. In the directory 'graphsurgeon' run `pip3 install graphsurgeon-<your-version>.whl`
   8. In the directory 'uff' run `pip3 install uff-<your-version>.whl`

### 5. Testing installation and cloning the repository
   1. Move to `cd /pc20xx-dataX/<your-username>/`
   2. Clone the repository via `git clone <https version of the repository>`
   3. Switch to deploys branch for triggers to work via `git checkout deploys`
   4. `cd mphys-project/Standalone`
   5. Here you can run `python3 gpu_test.py` and in the output it should say
      > Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
   6. Run `nohup python3 -u Gan4DS.py > gan4ds.out 2> gan4ds.err < /dev/null &` to run the package

[1]:http://rpm.pbone.net/index.php3/stat/3/limit/9/srodzaj/1/dl/40/search/libcuda.so.1()(64bit)/field[]/1/field[]/2
## Folder Structure Explained
### [Legacy] Jupyter Notebooks
These were initially run locally (found in 'Legacy' folder) but after some initial 'PoC' GANs we moved on to runnng them on Google Colab for some GPU power, coupled with the ease of running on this software. Here one can find all of our notebooks that we used for 'R&D'. For the actual final code please see the Standalone folder

### G4DS_Configs
Contains some important configuration files to be used in G4DS to produce the training data set.

### Generation
Collection of scripts to generate events given a trained GAN.

### Installation
This is just for the manual installation steps.

### Reports
These are our semester reports, one for each semster. In it we explain in detail our master's project and our results.

### Saves
Holds all the saved runs from different GANS

### Training
This folder contains the final production code. It is a complete package producing all of our work over the course of the final year.

### Training_Data
This is a collection of pickled python lists for the training dataset.
It contains S1, S2 and f200 for energies [5,235] keV in steps of 1keV.
