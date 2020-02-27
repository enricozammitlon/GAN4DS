# MPhys Project

## First time installation steps on pc20XX on Manchester Cluster
### 1. Installing Python3.6.x
   N.B Before doing this make sure you have bzip2-devel installed either as global or by installing it yourself somewhere        locally. In our case pc20xx didn't have it so we had to ask for it to be installed.
   1. Login on pc-20xx via `ssh <your-username>@pc<20xx>.hep.manchester.ac.uk`
   2. If you don't already, create a directory in pc20xx-dataX via `mkdir \pc20xx-dataX\<your-username>`
   3. Move to that directory via `cd \pc20xx-dataX\<your-username>`
   4. Create a new directory 'tmp' which will hold our downloads via `mkdir tmp`
   5. Move into tmp via `cd tmp`
   6. Now download the files for your local python version via `wget wget https://www.python.org/ftp/python/3.6.10/Python-           3.6.10.tgz` . In this case we use python 3.6.10 since it is the least version that supports Tensorflow 2.0 . At the           time of writing the current installation on most pc20xx is python 3.4.x
   7. Now extract via `tar zxvf Python-3.6.10.tgz`
   8. Move to this directory `cd Python-3.6.10`
   9. Now configure it to be put in your directory. This will be where python3.6.10 will be installed for you. Run                   `./configure --prefix=/pc20xx-dataX/<your-username>/python3.6.10`
   10. Run `make` and then `make install`
   11. When all of that is done move to `cd /pc20xx-dataX/<your-username>`
   12. Export the location of your install via `export PATH=/pc20xx-dataX/<your-username>/python3.6.10/bin:$PATH`
   12. Running `which python3` should present with the global install and your new python3.6.10 installation
   
 ### 2. Installing libcuda.so and other libraries
   1. Move to `cd /pc20xx-dataX/<your-username>/`
   2. Make a directory called `mkdir CUDA`
   3. Move to the new directory `cd CUDA`
   4. Now download the RPM locally via `wget ftp://ftp.pbone.net/mirror/atrpms.net/sl6-x86_64/atrpms/stable/nvidia-                 graphics302.17-libs-302.17-147.el6.x86_64.rpm` . Again make sure you install the right version for the distro you are         installing it on. If it is different check [1] .
   5. Now use a tool to extract it via `rpm2cpio <whatever-the-rpm-is-called>.rpm | cpio -idmv`. Now you should have a               directory called usr.
   6. Move back to `cd /pc20xx-dataX/<your-username>/`
   7. Export the location of your install via `export LD_LIBRARY_PATH=/pc20xx-dataX/<your-username/CUDA/usr/lib64/nvidia-           graphics-302.17:$PATH`. Again if your version is different change the name of nvidia-graphics-xx.xx
   
 ### 3. Creating a virtual environment and installing requirements
   1. Move to `cd /pc20xx-dataX/<your-username>/`
   2. Create the virtual environment via `python3 -m venv py3.6`
   3. Activate the environment via `source ./py3.6/bin/activate`
   4. Now move to the location of the python scripts `cd /pc20xx-dataX/<your-username>/mphys-project/Standalone`
   5. Install dependencies via `pip3 install -r requirements.txt`
   
## Bibliography

The bibliography contains an relevant articles that we have found, or have been sent to us. 

### G4 runs

G4 runs contain the data from any G4 runs we do, as well as the log files to ensure the dat can be reproduced.

## Log/Reports

Just any notes we thought would be useful to make, hoverever these are now largely incorportated into Jupyter notebooks we have produced. 

## Jupyter Notebooks

These were initially ran locally, but after some initial 'PoC' GANs, we moved on to runnng them on Google Colab for some GPU power, coupled with the ease of running on this software.

   
[1]:http://rpm.pbone.net/index.php3/stat/3/limit/9/srodzaj/1/dl/40/search/libcuda.so.1()(64bit)/field[]/1/field[]/2


