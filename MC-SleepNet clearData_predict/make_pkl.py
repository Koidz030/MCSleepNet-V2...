import pandas as pd
import pickle 
import numpy as np
import os

#flist = pd.read_csv('dataset_20190405_clearMouse.csv')
#flist = pd.read_csv('dataset_20190405_clearMouse4.csv')
#flist = pd.read_csv('/data2/clearData20190330/Y0mouselist.csv')
#eeglist = glob.glob('/data2/clearData20210724/raw/eeg_*.csv') # pd.read_csv('/data2/clearData20190330/train_mouse_all.csv')
eeglist = glob.glob('/data2/clearData20210727/raw/eeg_*.csv') #
print(eeglist)
#emglist = glob.glob('/data2/clearData20210724/raw/emg_*.csv')
emglist = glob.glob('/data2/clearData20210727/raw/emg_*.csv')
print(emglist)

#if not os.path.exists('/data2/clearData20210724/pkl/'):
#    os.makedirs('/data2/clearData20210724/pkl')

if not os.path.exists('/data2/clearData20210727/pkl/'):
    os.makedirs('/data2/clearData20210727/pkl')
    
#for fname in flist['name']:#i in range(1,15):#4314):
#    print fname#i
    #eeg = pd.read_csv('/data2/clearData20190330/raw/eeg_%04d.csv'% i,header=None, index_col=None, dtype=np.int32)
    #emg = pd.read_csv('/data2/clearData20190330/raw/emg_%04d.csv'% i,header=None, index_col=None, dtype=np.int32)
#    eeg = pd.read_csv('/data2/clearData20190724/raw/eeg_%s.csv'% fname, header=None, index_col=None, dtype=np.int32)
#    emg = pd.read_csv('/data2/clearData20190724/raw/emg_%s.csv'% fname, header=None, index_col=None, dtype=np.int32)

    #pickle.dump(eeg, open('/data2/clearData20190330/pkl/eeg_%04d.pkl'% i, 'w'),protocol=2)
    #pickle.dump(emg, open('/data2/clearData20190330/pkl/emg_%04d.pkl'% i, 'w'),protocol=2)
#    pickle.dump(eeg, open('/data2/clearData20210724/pkl/eeg_%s.pkl'% fname, 'w'),protocol=2)
#    pickle.dump(emg, open('/data2/clearData20210724/pkl/emg_%s.pkl'% fname, 'w'),protocol=2)

for eegfile, emgfile in zip(eeglist, emglist):
    eeg = pd.read_csv(eegfile, header=None, index_col=None, dtype=np.int32)
    emg = pd.read_csv(emgfile, header=None, index_col=None, dtype=np.int32)
    with open(eegfile.replace('raw', 'pkl').replace('.csv', '.pkl'), 'w') as f:
        pickle.dump(eeg, f, protocol=2)
    with open(emgfile.replace('raw', 'pkl').replace('.csv', '.pkl'), 'w') as f:
        pickle.dump(emg, f, protocol=2)
