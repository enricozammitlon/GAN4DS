# This script will accept root files and automagically transform them
# to pickles containing lists of the variables interested in w/ the appropriate data
import pickle
import uproot
import numpy as np
from os import listdir,makedirs
from os.path import isfile, join,exists,splitext
import sys, getopt

def main(argv):
    inputFolder='./in/root/'
    outputFolder='./in/pickles/'
    variables=['s1','s2','f200like']

    try:
        opts, args = getopt.getopt(argv,"hi:o:v:",["idir=","odir=","vars="])
    except getopt.GetoptError:
        print('RootConverter.py -i inputfolder -o outputfolder -v [var1,...]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('RootConverter.py -i inputfolder -o outputfolder -v [var1,var2,...]')
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputFolder = arg
        elif opt in ("-o", "--odir"):
            outputFolder = arg
        elif opt in ("-v", "--vars"):
            variables = list(arg.strip('[]').split(','))
    print(inputFolder)
    print(outputFolder)
    print(variables)
    allFiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f)) and '.root' in f]
    if not exists(outputFolder):
        makedirs(outputFolder)
    for file in allFiles:
        currentTree=uproot.open(inputFolder+file)["dstree"]
        allVariables={}
        for var in variables:
            allVariables[var]=np.array(currentTree.array(f"{var}"))
        pickle.dump( allVariables, open(outputFolder+'/'+splitext(file)[0]+".p", "wb" ) )

if __name__ == "__main__":
   main(sys.argv[1:])
