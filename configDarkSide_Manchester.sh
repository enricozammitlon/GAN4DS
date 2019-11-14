#!/bin/sh

export G4VIS_USE_OPENGLX=1
HOST=`hostname`
CNAF=`hostname | grep cnaf`
echo "configuring environment for $HOST"
if [ $DS_LNGS_CLUSTER ] ; then

 export G4DS=$PWD
 export DSDATA=$PWD/data/physics
 export PATH=$PATH:$PWD/tools
 export G4WORKDIR=$PWD/.g4ds
 export XERCESROOT=/usr/lib

 unset G4VIS_USE_DAWN           
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_OPENGLX  						   
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT

 source /usr/local/share/Geant4-10.1.2/geant4make/geant4make.sh
# source /cern/geant4/geant4.10.00.p01-build/geant4make.sh   

 export G4DATA=/usr/local/share/Geant4-10.1.2/data
 export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
 export G4LEDATA=$G4DATA/G4EMLOW6.41
 export G4NEUTRONHPDATA=$G4DATA/G4NDL4.5
 export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4
 export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.1
 export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.2
 export NeutronHPCrossSections=$G4DATA/G4NDL4.5
 export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
 export G4PIIDATA=$G4DATA/G4PII1.3

 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$G4WORKDIR/tmp/Linux-g++/g4ds:$XERCESROOT

elif [ $HOST == "deathstar" ]
then

 export G4DS=$PWD
 export DSDATA=$PWD/data/physics
 export PATH=$PATH:$PWD/tools
 export G4WORKDIR=$PWD/.g4ds

 unset G4VIS_USE_DAWN           
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_OPENGLX  						   
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT

 export G4DIR=/usr/local/share/Geant4-10.1.2
 export G4DATA=$G4DIR/data
 export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
 export G4LEDATA=$G4DATA/G4EMLOW6.35
 export G4NEUTRONHPDATA=$G4DATA/G4NDL4.4
 export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4
 export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation2.3
 export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.0
 export NeutronHPCrossSections=$G4DATA/G4NDL4.4
 export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
 export G4PIIDATA=$G4DATA/G4PII1.3
 
 source $G4DIR/share/Geant4-10.1.2/geant4make/geant4make.sh

 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$G4WORKDIR/tmp/Linux-g++/g4ds
    

elif [ $HOST == "phy-yoda.Princeton.EDU" ]
then
  export G4DS=$PWD
 export DSDATA=$PWD/data/physics
 export PATH=$PATH:$PWD/tools
 export G4WORKDIR=$PWD/.g4ds

 unset G4VIS_USE_DAWN
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_OPENGLX
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT

 export G4DIR=/usr/local/share/Geant4-10.1.2
 export G4DATA=$G4DIR/data
 export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
 export G4LEDATA=$G4DATA/G4EMLOW6.35
 export G4NEUTRONHPDATA=$G4DATA/G4NDL4.4
 export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4
 export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation2.3
 export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.0
 export NeutronHPCrossSections=$G4DATA/G4NDL4.4
 export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
 export G4PIIDATA=$G4DATA/G4PII1.3

 source $G4DIR/share/Geant4-10.1.2/geant4make/geant4make.sh

 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$G4WORKDIR/tmp/Linux-g++/g4ds:`root-config --glibs --cflags`:/usr/local/root_v5.32.04/lib/root

elif [ $HOST == "$CNAF" ]
then
 #unset ROOTSYS
 export G4DS=${PWD}
 
 
 #export ROOTSYS=/usr/local/root/v5.34.11/
 #export ROOTSYS='/opt/exp_software/darkside/software/root/root_v5.34.32_source'
 
 export DSDATA=$PWD/data/physics

 export PATH=${ROOTSYS}/bin:${PWD}/tools:${PATH}
 export LD_LIBRARY_PATH=${ROOTSYS}/lib:${ROOTSYS}/lib/root:${LD_LIBRARY_PATH}

 source /opt/exp_software/darkside/software/geant4/geant4.10.00.p01/share/Geant4-10.0.1/geant4make/geant4make.sh > /dev/null 2>&1
 #source /usr/local/geant4/geant4.10.00.p01/share/Geant4-10.0.1/geant4make/geant4make.sh > /dev/null
 #source /usr/local/geant4/geant4.9.6.p01/share/Geant4-9.6.1/geant4make/geant4make.sh
 
 export G4VRMLFILE_MAX_FILE_NUM=1
 export G4VIS_USE=1
 unset G4VIS_USE_DAWN           
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT
 
 #export G4DATA=/usr/local/geant4/share
 export G4DATA=/opt/exp_software/darkside/software/geant4/g4data
						
 export G4SAIDXSDATA=$G4DATA/G4SAIDDATA1.1
 export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
 export G4LEDATA=$G4DATA/G4EMLOW6.41
 export G4NEUTRONHPDATA=$G4DATA/G4NDL4.4
 export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4 
 export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.1
 export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.2
 export NeutronHPCrossSections=$G4DATA/G4NDL4.4
 export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
 export G4PIIDATA=$G4DATA/G4PII1.3

 export G4WORKDIR=$PWD/.g4ds


 export LD_LIBRARY_PATH=$G4WORKDIR/tmp/Linux-g++/g4ds/:$XERCESCROOT/lib:$LD_LIBRARY_PATH

elif [ $HOST == "palpatine.Princeton.EDU" ]
then
# unset ROOTSYS                                                                                                                                                                                                   
 export ROOTSYS=/usr/local/root_v5.34.21/
 export G4DS=$PWD
 export DSDATA=$PWD/data/physics
 export PATH=$PATH:$PWD/tools
 export G4WORKDIR=$PWD/.g4ds

# unset G4VIS_USE_DAWN
# unset G4VIS_USE_OPENGLQT
# unset G4VIS_USE_OPENGLX
# unset G4VIS_USE_RAYTRACERX
# unset G4VIS_USE_VRML                                                                                                                                                                                            
# unset G4VIS_USE_OIX
# unset G4VIS_USE_OPENGLXM                                                                                                                                                                                      
# unset G4UI_USE_QT

 export G4DIR=/usr/local/geant4.10.00.p02

 export G4DATA=$G4DIR/data
 export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
 export G4LEDATA=$G4DATA/G4EMLOW6.35
 export G4NEUTRONHPDATA=$G4DATA/G4NDL4.4
 export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4
 export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.0
  export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.0
#  export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.0
 export NeutronHPCrossSections=$G4DATA/G4NDL4.4
 export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
 export G4PIIDATA=$G4DATA/G4PII1.3

# export G4DATA01=/usr/local/geant4.10.01/data/
# export G4LEVELGAMMADATA=$G4DATA01/PhotonEvaporation3.1
# export G4RADIOACTIVEDATA=$G4DATA01/RadioactiveDecay4.2

 source $G4DIR/share/Geant4-10.0.2/geant4make/geant4make.sh > /dev/null
 export LD_LIBRARY_PATH=$G4WORKDIR/tmp/Linux-g++/g4ds/:$LD_LIBRARY_PATH

 
 elif [ $HOST == "ds50srv01.fnal.gov" ]
 then

# unset ROOTSYS
 export G4DS=${PWD}
# export ROOTSYS=/usr/local/root/v5.34.01/

#do some basic setups
#. /ds50/app/ds50/setup_ds50
#disabled in April 2014, since it causes conflicts while submitting jobs. It is anyway already called in one of my (Bernd) other setup-scripts (my_bashrc).

#now start setting up products
setup clhep v2_1_4_1 -q e4:prof #http://www.geant4.org/geant4/support/ReleaseNotes4.10.0.html
#setup geant4 v4_9_6_p02 -q e4:prof
#setup geant4 v4_10_0_p01 -q e4:prof
#setup geant4 v4_10_0_0 -q e4:prof
setup geant4 v4_10_0_0_p01 -q e4:prof #had v4_10_0_0_p01 before
 setup root v5_34_12 -q e4:prof
#source $GEANT4_FQ_DIR/share/Geant4-9.6.1/geant4make/geant4make.sh

#copy these lines from the geant4make script
export G4SYSTEM=Linux-g++
export G4INSTALL=
export G4INCLUDE=$GEANT4_FQ_DIR/include/Geant4
#export G4LIB=$GEANT4_FQ_DIR/lib64/Geant4-9.6.2
#export G4LIB=$GEANT4_FQ_DIR/lib64/Geant4-10.0.0/
export G4LIB=$GEANT4_FQ_DIR/lib64/Geant4-10.0.1/

 export DSDATA=$PWD/data/physics

 export PATH=${ROOTSYS}/bin:${PWD}/tools:${PATH}
 export LD_LIBRARY_PATH=${GEANT4_FQ_DIR}/lib64:${ROOTSYS}/lib:${ROOTSYS}/lib/root:${LD_LIBRARY_PATH}
export G4WORKDIR=$PWD/.g4ds
export G4LIB_BUILD_SHARED=1
export G4LIB_USE_ZLIB=1
export G4LIB_USE_GDML=1
export G4UI_USE_TCSH=1

 #source /usr/local/geant4/geant4.10.00.p01/share/Geant4-10.0.1/geant4make/geant4make.sh > /dev/null
 #source /usr/local/geant4/geant4.9.6.p01/share/Geant4-9.6.1/geant4make/geant4make.sh
 
 export G4VRMLFILE_MAX_FILE_NUM=1
 export G4VIS_USE=1

 unset G4VIS_USE_DAWN           
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT
						
 export G4DATA=$GEANT4_FQ_DIR ### not sure
#setup g4emlow v6_32
setup g4emlow v6_35
#this is set by the above setup
# export G4LEDATA=$G4DATA/G4EMLOW6.23

#setup g4photon v2_3 
setup g4photon v3_0 
#export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation2.2

# setup g4radiative v3_6
 setup g4radiative v4_0
# export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay3.4
 
#setup g4neutron v4_2
setup g4neutron v4_4
#export NeutronHPCrossSections=$G4DATA/G4NDL4.0
# export G4NEUTRONHPDATA=$G4DATA/G4NDL4.0

#setup g4neutronxs v1_2
setup g4neutronxs v1_4
# export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.1

setup g4nucleonxs v1_1

#also done by setup
#stays the same in g4.10
setup g4surface v1_0
# export G4REALSURFACEDATA=$G4DATA/RealSurface1.0

#stays the same in g4.10
setup g4pii v1_3
# export G4PIIDATA=$G4DATA/G4PII1.3
 
#<<<<<<< configDarkSide.sh
#=======
# export G4DATA=/usr/local/geant4/share
						
# export G4SAIDXSDATA=$G4DATA/G4SAIDDATA1.1
# export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
# export G4LEDATA=$G4DATA/G4EMLOW6.35
# export G4NEUTRONHPDATA=$G4DATA/G4NDL4.4
# export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4 
# export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.1
# export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.2
# export NeutronHPCrossSections=$G4DATA/G4NDL4.4
# export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
# export G4PIIDATA=$G4DATA/G4PII1.3
#>>>>>>> 1.9

 #G4NEUTRONHPDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4NDL4.2
 #G4LEDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4EMLOW6.32
 #G4LEVELGAMMADATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/PhotonEvaporation2.3
 #G4RADIOACTIVEDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/RadioactiveDecay3.6
 #G4NEUTRONXSDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4NEUTRONXS1.2
 #G4PIIDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4PII1.3
 #G4SAIDXSDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4SAIDDATA1.1

#for some reason this doesn't get set properly, so set it by hand
export XERCESCROOT=${XERCES_C_DIR}/Linux64bit+2.6-2.12-e4-prof

#fix the CLHEP environment variables to be the ones that G4 wants
 export CLHEP_BASE_DIR=${CLHEP_BASE}
 export CLHEP_INCLUDE_DIR=${CLHEP_INC}
 export CLHEP_LIB_DIR=${CLHEP_BASE_DIR}/lib

 export LD_LIBRARY_PATH=${CLHEP_LIB_DIR}:$G4WORKDIR/tmp/Linux-g++/g4ds/:$XERCESCROOT/lib:$LD_LIBRARY_PATH


 elif [ $HOST == "pc2012.hep.manchester.ac.uk" ] || [ $HOST == "hepgpu1.hep.manchester.ac.uk" ] || [ $HOST == "pc2014.hep.manchester.ac.uk" ]
 then

#source /cvmfs/fermilab.opensciencegrid.org/products/larsoft/setup
source /cvmfs/larsoft.opensciencegrid.org/products/setup
#setup geant4 v4_10_1_p02 -q e7:prof
setup geant4 v4_10_1_p03a -q e10:prof
#setup root v6_04_02 -q e7:prof
setup root v6_06_04b -q e10:nu:prof
#setup marley v0_9_5 -q e10:prof             # e10 version to fit with other programs since Nov 2018
#setup marley v1_0_0b -q e14:prof            # old working marley version 

#setup marley v1_0_0b -q e15:prof             # version that now works!
#setup geant4 and root version 6 from cvmfs fermilab
#also setup marley version 1.0.0b from cvmfs femilab (need e14 version)

# !!  to include marley, leave commented out for sourceing prior to compilation	 !! #
# !! of g4ds, then after compilation is complete uncomment setup marley and redo !! #
# !! ~$ source configDarkSide_Manchester 										 !! #

#setting up manchester machines using Fermilab UPS system.

# unset ROOTSYS
 export G4DS=${PWD}
# export ROOTSYS=/usr/local/root/v5.34.01/

#do some basic setups
#. /ds50/app/ds50/setup_ds50
#disabled in April 2014, since it causes conflicts while submitting jobs. It is anyway already called in one of my (Bernd) other setup-scripts (my_bashrc).

#now start setting up products
#setup clhep v2_1_4_1 -q e4:prof #http://www.geant4.org/geant4/support/ReleaseNotes4.10.0.html
#setup geant4 v4_9_6_p02 -q e4:prof
#setup geant4 v4_10_0_p01 -q e4:prof
#setup geant4 v4_10_0_0 -q e4:prof
#setup geant4 v4_10_0_0_p01 -q e4:prof #had v4_10_0_0_p01 before
# setup root v5_34_12 -q e4:prof
#source $GEANT4_FQ_DIR/share/Geant4-9.6.1/geant4make/geant4make.sh

#copy these lines from the geant4make script
export G4SYSTEM=Linux-g++
export G4INSTALL=
export G4INCLUDE=$GEANT4_FQ_DIR/include/Geant4
#export G4LIB=$GEANT4_FQ_DIR/lib64/Geant4-9.6.2
#export G4LIB=$GEANT4_FQ_DIR/lib64/Geant4-10.0.0/
export G4LIB=$GEANT4_FQ_DIR/lib64/Geant4-10.1.3/


 export DSDATA=$PWD/data/physics

 export PATH=${ROOTSYS}/bin:${PWD}/tools:${PATH}
 export LD_LIBRARY_PATH=${GEANT4_FQ_DIR}/lib64:${ROOTSYS}/lib:${ROOTSYS}/lib/root:${LD_LIBRARY_PATH}
export G4WORKDIR=$PWD/.g4ds
export G4LIB_BUILD_SHARED=1
export G4LIB_USE_ZLIB=1
export G4LIB_USE_GDML=1
export G4UI_USE_TCSH=1

 #source /usr/local/geant4/geant4.10.00.p01/share/Geant4-10.0.1/geant4make/geant4make.sh > /dev/null
 #source /usr/local/geant4/geant4.9.6.p01/share/Geant4-9.6.1/geant4make/geant4make.sh
 
 export G4VRMLFILE_MAX_FILE_NUM=1
 export G4VIS_USE=1

 unset G4VIS_USE_DAWN           
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT

export G4CVMFSDATA=/cvmfs/fermilab.opensciencegrid.org/products/larsoft/



						
export G4DATA=$GEANT4_FQ_DIR ### not sure
#setup g4emlow v6_32
setup g4emlow v6_41
#this is set by the above setup
# export G4LEDATA=$G4DATA/G4EMLOW6.23

#setup g4photon v2_3 
setup g4photon v3_0 	
#export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation2.2

# setup g4radiative v3_6
 setup g4radiative v4_0
# export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay3.4
 
#setup g4neutron v4_2
setup g4neutron v4_5
#export NeutronHPCrossSections=$G4DATA/G4NDL4.0
# export G4NEUTRONHPDATA=$G4DATA/G4NDL4.0

#setup g4neutronxs v1_2
setup g4neutronxs v1_4
# export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.1

setup g4nucleonxs v1_1

#also done by setup
#stays the same in g4.10
setup g4surface v1_0
# export G4REALSURFACEDATA=$G4DATA/RealSurface1.0

#stays the same in g4.10
setup g4pii v1_3
# export G4PIIDATA=$G4DATA/G4PII1.3
 
#<<<<<<< configDarkSide.sh
#=======
# export G4DATA=/usr/local/geant4/share
						
# export G4SAIDXSDATA=$G4DATA/G4SAIDDATA1.1
# export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
# export G4LEDATA=$G4DATA/G4EMLOW6.35
# export G4NEUTRONHPDATA=$G4DATA/G4NDL4.4
# export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4 
# export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.1
# export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.2
# export NeutronHPCrossSections=$G4DATA/G4NDL4.4
# export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
# export G4PIIDATA=$G4DATA/G4PII1.3
#>>>>>>> 1.9

 #G4NEUTRONHPDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4NDL4.2
 #G4LEDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4EMLOW6.32
 #G4LEVELGAMMADATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/PhotonEvaporation2.3
 #G4RADIOACTIVEDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/RadioactiveDecay3.6
 #G4NEUTRONXSDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4NEUTRONXS1.2
 #G4PIIDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4PII1.3
 #G4SAIDXSDATA=/work/GEANT4/geant4.9.6/share/Geant4-9.6.0/data/G4SAIDDATA1.1

#for some reason this doesn't get set properly, so set it by hand
export XERCESCROOT=${XERCES_C_DIR}/Linux64bit+2.6-2.12-e7-prof


#fix the CLHEP environment variables to be the ones that G4 wants
 export CLHEP_BASE_DIR=${CLHEP_BASE}
 export CLHEP_INCLUDE_DIR=${CLHEP_INC}
 export CLHEP_LIB_DIR=${CLHEP_BASE_DIR}/lib

 export LD_LIBRARY_PATH=${CLHEP_LIB_DIR}:$G4WORKDIR/tmp/Linux-g++/g4ds/:$XERCESCROOT/lib:$LD_LIBRARY_PATH


else

 unset ROOTSYS
 export G4DS=${PWD}
 
 
 export ROOTSYS=/usr/local/root/v5.34.11/

 export DSDATA=$PWD/data/physics

 export PATH=${ROOTSYS}/bin:${PWD}/tools:${PATH}
 export LD_LIBRARY_PATH=${ROOTSYS}/lib:${ROOTSYS}/lib/root:${LD_LIBRARY_PATH}

 source /usr/local/share/Geant4-10.1.2/geant4make/geant4make.sh > /dev/null
 #source /usr/local/geant4/geant4.9.6.p01/share/Geant4-9.6.1/geant4make/geant4make.sh
 
 export G4VRMLFILE_MAX_FILE_NUM=1
 export G4VIS_USE=1
 unset G4VIS_USE_DAWN           
 unset G4VIS_USE_OPENGLQT
 unset G4VIS_USE_RAYTRACERX
 unset G4VIS_USE_VRML
 unset G4VIS_USE_OIX
 unset G4VIS_USE_OPENGLXM
 unset G4UI_USE_QT
 
 export G4DATA=/usr/local/share/Geant4-10.1.2/data
						
 export G4SAIDXSDATA=$G4DATA/G4SAIDDATA1.1
 export G4SAIDDATA=$G4DATA/G4SAIDDATA1.1
 export G4LEDATA=$G4DATA/G4EMLOW6.41
 export G4NEUTRONHPDATA=$G4DATA/G4NDL4.5
 export G4NEUTRONXSDATA=$G4DATA/G4NEUTRONXS1.4 
 export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation3.1
 export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay4.2
 export NeutronHPCrossSections=$G4DATA/G4NDL4.5
 export G4REALSURFACEDATA=$G4DATA/RealSurface1.0
 export G4PIIDATA=$G4DATA/G4PII1.3

 export G4WORKDIR=$PWD/.g4ds


 export LD_LIBRARY_PATH=$G4WORKDIR/tmp/Linux-g++/g4ds/:$XERCESCROOT/lib:$LD_LIBRARY_PATH
						
fi
