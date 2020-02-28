# MPhys Project
# Table of Contents
1. [First Installation](#first-time-installation)
2. [Folder Structure](#folder-structure)

## First time installation steps on hepgpuX on Manchester Cluster <a name="first-time-installation"></a>
### 1. Installing Python3.6.x
   N.B Before doing this make sure you have bzip2-devel installed either as global or by installing it yourself somewhere        locally.
   1. Login on hepgpuX via `ssh <your-username>@hepgpuX.blackett.manchester.ac.uk`
   2. If you don't already, create a directory in hepgpuX-dataY via `mkdir \hepgpuX-dataY\<your-username>`
   3. Move to that directory via `cd \hepgpuX-dataY\<your-username>`
   4. Create a new directory 'tmp' which will hold our downloads via `mkdir tmp`
   5. Move into tmp via `cd tmp`
   6. Now download the files for your local python version via `wget wget https://www.python.org/ftp/python/3.6.10/Python-           3.6.10.tgz` . In this case we use python 3.6.10 since it is the least version that supports Tensorflow 2.0 . At the           time of writing the current installation on most hepgpuX is python 3.6.x
   7. Now extract via `tar zxvf Python-3.6.10.tgz`
   8. Move to this directory `cd Python-3.6.10`
   9. Now configure it to be put in your directory. This will be where python3.6.10 will be installed for you. Run                   `./configure --prefix=/hepgpuX-dataY/<your-username>/python3.6.10`
   10. Run `make` and then `make install`
   11. When all of that is done move to `cd /hepgpuX-dataY/<your-username>`
   12. Export the location of your install via `export PATH=/hepgpuX-dataY/<your-username>/python3.6.10/bin:$PATH`
   12. Running `which python3` should present with the global install and your new python3.6.10 installation
   
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
   3. `cd mphys-project/Standalone`
   4. Here you can run `python3 gpu_test.py` and in the output it should say
      > Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
   5. Run `python3 Gan4DS.py` to run the package

## Folder Structure <a name="folder-structure"></a>

### 1. Bibliography

The bibliography contains an relevant articles that we have found, or have been sent to us. 

### 2. G4 runs

G4 runs contain the data from any G4 runs we do, as well as the log files to ensure the dat can be reproduced.

### 3. Log/Reports

Just any notes we thought would be useful to make, hoverever these are now largely incorportated into Jupyter notebooks we have produced. 

## 4. Jupyter Notebooks

These were initially ran locally, but after some initial 'PoC' GANs, we moved on to runnng them on Google Colab for some GPU power, coupled with the ease of running on this software.

   
[1]:http://rpm.pbone.net/index.php3/stat/3/limit/9/srodzaj/1/dl/40/search/libcuda.so.1()(64bit)/field[]/1/field[]/2


