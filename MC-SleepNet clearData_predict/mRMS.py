#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import sys
import pickle
import csv

window_size = 250

#emglist = glob.glob('/data2/clearData20210724/pkl/emg_*.pkl')
emglist = glob.glob('/data2/clearData20210727/pkl/emg_*.pkl')

#@profile
def main():
    for emgpkl in emglist:
        print(emgpkl)
        with open(emgpkl, "rb") as f:
            emg = pickle.load(f)
        emg = np.power(np.array(emg).flatten(),2)
        window = np.ones(window_size)/float(window_size)
        rms = np.sqrt(np.convolve(emg,window,"same"))
        rms = pd.Series(rms)
        f = open(emgpkl.replace('emg', 'rms'),"wb")
        pickle.dump(rms, f, protocol=2)
        f.close()
        del rms
        del emg
        del window

if __name__ == "__main__":
    main()
