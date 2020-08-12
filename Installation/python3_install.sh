INSTALL_BASE_PATH="/hepgpu3-data2/ricozl/GAN4DS/python3.6.10"
cd /hepgpu3-data2/ricozl/GAN4DS
mkdir build
cd build
[ -f Python-3.6.10.tgz ] || wget https://www.python.org/ftp/python/3.6.10/Python-3.6.10.tgz
tar -zxvf Python-3.6.10.tgz

[ -f sqlite-autoconf-3240000.tar.gz ] || wget https://www.sqlite.org/2018/sqlite-autoconf-3240000.tar.gz
tar -zxvf sqlite-autoconf-3240000.tar.gz

cd sqlite-autoconf-3240000
./configure --prefix=${INSTALL_BASE_PATH}
make
make install

cd ../Python-3.6.10
LD_RUN_PATH=${INSTALL_BASE_PATH}/lib configure
LDFLAGS="-L ${INSTALL_BASE_PATH}/lib"
CPPFLAGS="-I ${INSTALL_BASE_PATH}/include"
LD_RUN_PATH=${INSTALL_BASE_PATH}/lib make
./configure --prefix=${INSTALL_BASE_PATH}
make
make install

cd /hepgpu3-data2/ricozl/GAN4DS
LINE_TO_ADD="export PATH=${INSTALL_BASE_PATH}/bin:\$PATH"
if grep -q -v "${LINE_TO_ADD}" $HOME/.bash_profile; then echo "${LINE_TO_ADD}" >> $HOME/.bash_profile; fi
source $HOME/.bash_profile
