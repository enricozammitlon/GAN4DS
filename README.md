# mphys-project

# Bibliography

The bibliography contains an relevant articles that we have found, or have been sent to us. 

# G4 runs

G4 runs contain the data from any G4 runs we do, as well as the log files to ensure the dat can be reproduced.

# Log/Reports

Just any notes we thought would be useful to make, hoverever these are now largely incorportated into Jupyter notebooks we have produced. 

# Jupyter Notebooks

These were initially ran locally, but after some initial 'PoC' GANs, we moved on to runnng them on Google Colab for some GPU power, coupled with the ease of running on this software.
# mphys-project

# Bibliography

The bibliography contains an relevant articles that we have found, or have been sent to us. 

# G4 runs

G4 runs contain the data from any G4 runs we do, as well as the log files to ensure the dat can be reproduced.

# Log/Reports

Just any notes we thought would be useful to make, however these are now largely incorportated into Jupyter notebooks we have produced.

# Jupyter Notebooks

These were initially ran locally, but after some initial 'PoC' GANs, we moved on to runnng them on Google Colab for some GPU power, coupled with the ease of running on this software.

# Installation steps on pc20XX on Manchester Cluster
## 1. Installing Python3.6.x

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


