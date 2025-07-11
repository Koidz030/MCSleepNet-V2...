import pandas as pd
import glob
import numpy as np
import scipy.io
import os

#files = glob.glob("/data2/clearData20210724/pkl/eeg*.pkl")
#files.extend(glob.glob("/data2/clearData20210724/pkl/emg*.pkl"))
#files.extend(glob.glob("/data2/clearData20210724/pkl/rms*.pkl"))

files = glob.glob("/data2/clearData20210727/pkl/eeg*.pkl")
files.extend(glob.glob("/data2/clearData20210727/pkl/emg*.pkl"))
files.extend(glob.glob("/data2/clearData20210727/pkl/rms*.pkl"))

#files = glob.glob("/data2/clearData20190330/pkl/eeg_Y0*.pkl")
#files.extend(glob.glob("/data2/clearData20190330/pkl/emg_Y0*.pkl"))

for f in files:
    if "rms" in f:
        a = pd.read_pickle(f).values.flatten().astype('float32')
        scipy.io.savemat(f[0:29]+"rms"+f[32:-3]+'mat',{"value":a})
    else:
        a = pd.read_pickle(f).values.flatten().astype('int16')
        scipy.io.savemat(f[:-4]+".mat",{"value":a})
    print "save to " + f[:-4] + ".mat"
    #os.remove(f)
    #print "remove " + f

"""
files = glob.glob("/data2/clearData20190330/pkl/emg*.pkl")#rms*")

for f in files:
    a = pd.read_pickle(f).values.flatten().astype('float32')
    scipy.io.savemat(f[:-4]+".mat",{"value":a})
    print "save to " + f[:-4] + ".mat"
    os.remove(f)
    print "remove " + f
   
""" 
