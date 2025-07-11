import os

import numpy as np
import pandas as pd
import pickle
from scipy import fftpack
import scipy.io
import csv

from deepsleep.sleep_stage import print_n_samples_each_class
from deepsleep.utils import get_balance_class_downsample
from deepsleep.utils import get_balance_class_oversample

from deepsleep.sleep_stage import (class_dict)

import re
import sys
import time
import math # 20190806 LO added.
# Upper limit for data loading so as to be analyzed by LSTM layer which uses consecutive 25 epochs.
#clearDataFlag = 1#0#1 # For analysis of clear data, set this flag 1. For analysis of noisy data, please set this flag 0
#if clearDataFlag:
    # 2019/08/17 LO changed upper limit for analysis.
#    upperLimit = 17275#17000 # epoch (This value must be rational to 25, and less than the length of input eeg, emg epoch number.)
#else:
#    upperLimit = 8500 # epoch (This value must be rational to 25, and less than the length of input eeg, emg epoch number.)
# In small data size analysis : 17000 -> 17275 is upper limit.
# In large data size analysis and noisy data(4 mice) : 8500

class NonSeqDataLoader(object):

    def __init__(self, data_dir, upperLimit):
        self.data_dir = data_dir
        self.upperLimit = upperLimit

    def _load_csv_file(self, mouseid): # csv_file):
        with open("/data1/mouse_4313/pkl/eeg_%s.pkl"%(mouseid), "rb") as f: # raw/eeg_%s.csv"%(csv_file), "r") as f:
            eeg = pickle.load(f) # reader = csv.reader(f)
            #eeg = []
            #for row in reader:
            #    eeg.append(row)
            #eeg = np.array(eeg)
        with open("/data1/mouse_4313/pkl/emg_%s.pkl"%(mouseid), 'rb') as f: # raw/emg_%s.csv"%(csv_file), "r") as f:
            emg = pickle.load(f) # reader = csv.reader(f)
            #emg = []
            #for row in reader:
            #    emg.append(row)
            #emg = np.array(emg)
        #eeg = np.array(pickle.load(open("%s/pkl/eeg_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #emg = np.array(pickle.load(open("%s/pkl/rms_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(self.data_dir,csv_file))['value']
        #emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        #if clearDataFlag:
        #eeg = scipy.io.loadmat("%s/pkl/eeg_%s.mat"%(self.data_dir,csv_file))['value']
        #emg = scipy.io.loadmat("%s/pkl/rms_%s.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        #else:
        #    eeg = scipy.io.loadmat("%s/pkl/Noisy/eeg_%s.mat"%(self.data_dir,csv_file))['value']
        #    emg = scipy.io.loadmat("%s/pkl/Noisy/rms_%s.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        print('shape of eeg : '+str(eeg.shape))
        eeg = np.reshape(eeg,(-1,5000)) # 20 second epoch data in sampling frequency is 250Hz.
        emg = np.reshape(emg,(-1,5000))
        #if clearDataFlag:
        #    labels = pd.read_csv("%s/training_20s/%04d_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten()
        #labels = pd.read_csv("%s/training_20s/%s_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten()
        #else:
        #    labels = pd.read_csv("%s/training_20s/madeByLO/%s_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten()
        with open("/data1/mouse_4313/pkl/stg20s_%s.pkl"%(mouseid), 'rb') as f:
            labels=pickle.load(f)
        #labels = pd.read_csv("/data1/training_20s/%s_training.csv"%(csv_file))
        #labels[labels=="W"] = 0
        #labels[labels=="NR"] = 1
        #labels[labels=="R"] = 2
        return list(eeg[:self.upperLimit]), list(emg[:self.upperLimit]), list(labels[:self.upperLimit])
        #return list(eeg[:8500]), list(emg[:8500]), list(labels[:8500])

    def _load_csv_list_files(self, csv_files):
        """Load data and labels from list of npz files."""
        eeg = []
        emg = []
        labels = []
        print('called csv file list in _load_csv_list_files()')
        print(csv_files)
        for csv_f in [csv_files]:
            #sys.stdout.write("\rLoding %04d"%csv_f)
            sys.stdout.write("\rLoding %s"%csv_f)
            sys.stdout.flush()
            tmp_eeg, tmp_emg, tmp_labels = self._load_csv_file(csv_f)
            eeg = eeg + tmp_eeg
            emg = emg + tmp_emg
            labels = labels + tmp_labels
        print ""
        eeg = np.array(eeg)
        emg = np.array(emg)
        labels = np.array(labels)

        return eeg, emg, labels

    def load_cv_data(self, files, is_train):
        # Load a npz file
        print "Load :"
        eeg, emg, label = self._load_csv_list_files(files)
        print " "
        # Reshape the data to match the input of the model - conv2d
        eeg = eeg[:, :, np.newaxis, np.newaxis]
        emg = emg[:, :, np.newaxis, np.newaxis]

        # Casting
        eeg = eeg.astype(np.float32)
        emg = emg.astype(np.float32)
        label = label.astype(np.int32)

        print "data set: {}, {}, {}".format(eeg.shape, emg.shape, label.shape)
        print_n_samples_each_class(label)
        print " "

        # Use balanced-class, oversample training set
        if is_train:
            #eeg, emg, label = get_balance_class_downsample(
            eeg, emg, label = get_balance_class_oversample(
                x1=eeg, x2=emg, y=label
            )
            #print "downsampled training set: {}, {}, {}".format(
            print "oversampled training set: {}, {}, {}".format(
                eeg.shape, emg.shape, label.shape
            )
            print_n_samples_each_class(label)
            print " "


        return eeg, emg, label
    '''
    def load_cv_data2(self, files, is_train):
        # Load a npz file
        print "Load :"
        eeg, emg, label = self._load_csv_list_files(files)
        print " "
        # Reshape the data to match the input of the model - conv2d
        eeg = eeg[:, :, np.newaxis, np.newaxis]
        emg = emg[:, :, np.newaxis, np.newaxis]

        # Casting
        eeg = eeg.astype(np.float32)
        emg = emg.astype(np.float32)
        label = label.astype(np.int32)

        print "data set: {}, {}, {}".format(eeg.shape, emg.shape, label.shape)
        print_n_samples_each_class(label)
        print " "

        # Use balanced-class, oversample training set
        if is_train:
            #eeg, emg, label = get_balance_class_downsample(
            eeg, emg, label = get_balance_class_oversample(
                x1=eeg, x2=emg, y=label
            )
            #print "downsampled training set: {}, {}, {}".format(
            print "oversampled training set: {}, {}, {}".format(
                eeg.shape, emg.shape, label.shape
            )
            print_n_samples_each_class(label)
            print " "

        return eeg, emg, label
    '''
    # 2019/7/8 Leo Ota added function to validate the accuracy.
    def load_csv_data_valid(self, files, is_train):
        # load a npz file
        print("Start Loading csv files for validation test: ")
        print(files)
        for no_file, fname in enumerate(files):
            print("File Number: "+str(no_file))
            print("File Name: "+str(fname))
            if no_file == 0:
                eeg, emg, label = self._load_csv_list_files(fname)
                print(eeg)
                print(emg)
                print(label)
                print(eeg.shape)
                print(emg.shape)
                print(len(label))
                #eeg = eeg.reshape(1,5000,1,1)
                #emg = emg.reshape(1,5000,1,1)
                label = np.array(label)
            else:
                eeg_tmp, emg_tmp, label_tmp = self._load_csv_list_files(fname)
                label_tmp = np.array(label)
                eeg = np.concatenate([eeg,eeg_tmp], axis=0)
                emg = np.concatenate([emg,emg_tmp], axis=0)
                label = np.concatenate([label,label_tmp], axis=0)
        print(" ")
        # Reshape the data to match the input of the model - conv1d
        eeg = eeg.astype(np.float32)
        emg = emg.astype(np.float32)
        label = label.astype(np.int32)
        print("data set: {}, {}, {}".format(eeg.shape, emg.shape, label.shape))
        print_n_samples_each_class(label)
        print(" ")
        return eeg, emg, label

class SeqDataLoader(object):
    def __init__(self, data_dir, upperLimit):
        self.data_dir = data_dir
        self.upperLimit = upperLimit

    # These functions were not used.
    def _load_csv_file(self, csv_file):
        print('csv file name : '+str(csv_file))
        with open("/data1/raw/eeg_%s.csv"%(csv_file), "r") as f:
            reader = csv.reader(f)
            eeg = []
            for row in reader:
                eeg.append(row)
            eeg = np.array(eeg)
        with open("/data1/raw/emg_%s.csv"%(csv_file), "r") as f:
            reader = csv.reader(f)
            emg = []
            for row in reader:
                emg.append(row)
            emg = np.array(emg)
        #print('csv file name in class variable : '+str(self.csv_file)) # no attribute self.csv_file
        #eeg = np.array(pickle.load(open("%s/pkl/eeg_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #emg = np.array(pickle.load(open("%s/pkl/rms_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(self.data_dir,csv_file))['value']
        #emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        #eeg = scipy.io.loadmat("%s/pkl/eeg_%s.mat"%(self.data_dir,csv_file))['value']
        #emg = scipy.io.loadmat("%s/pkl/rms_%s.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        eeg = np.reshape(eeg,(-1,5000)) # for 20 second in the sampling frequency is 250Hz.
        emg = np.reshape(emg,(-1,5000))
        #labels = pd.read_csv("%s/training_20s/%04d_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised save folder
        labels = pd.read_csv("/data1/training_20s/%s_training.csv"%(csv_file))
        labels[labels=="W"] = 0
        labels[labels=="NR"] = 1
        labels[labels=="R"] = 2
        upperLimit = len(labels)
        return eeg[:self.upperLimit], emg[:self.upperLimit], labels[:self.upperLimit]

    def _load_csv_list_files(self, csv_file):
        """Load data and labels from list of npz files."""
        eeg = []
        emg = []
        labels = []
        fs = None
        for csv_f in [csv_file]:
            sys.stdout.write("\rLoading {} ...".format(csv_f))
            sys.stdout.flush()
            tmp_eeg, tmp_emg, tmp_labels = self._load_csv_file(csv_f)

            # Reshape the data to match the input of the model - conv2d
            tmp_eeg = tmp_eeg[:, :, np.newaxis, np.newaxis]
            tmp_emg = tmp_emg[:, :, np.newaxis, np.newaxis]

            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_eeg = tmp_eeg.astype(np.float32)
            tmp_emg = tmp_emg.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            eeg.append(tmp_eeg)
            emg.append(tmp_emg)
            labels.append(tmp_labels)

        return eeg, emg, labels

    def load_cv_data(self,list_files,is_train):

        # Load a csv file
        print "Load data set:"
        eeg, emg, label = self._load_csv_list_files(list_files)
        print " "

        if is_train:
            print "Training set: n_subjects={}".format(len(eeg))
            n_train_examples = 0
            for d in eeg[:3]:
                print d.shape
                n_train_examples += d.shape[0]
            print "Number of examples = {}".format(n_train_examples)
            print_n_samples_each_class(np.hstack(label))
            print " "
        else:
            print "Validation set: n_subjects={}".format(len(eeg))
            n_valid_examples = 0
            for d in eeg:
                print d.shape
                n_valid_examples += d.shape[0]
            print "Number of examples = {}".format(n_valid_examples)
            print_n_samples_each_class(np.hstack(label))
            print " "

        return eeg, emg, label

    def _load_csv_file2(self, csv_file):
        print('csv file name : '+str(csv_file))
        with open("/data1/raw/eeg_%s.csv"%(csv_file), "r") as f:
            reader = csv.reader(f)
            eeg = []
            for row in reader:
                eeg.append(row)
            eeg = np.array(eeg)
        with open("/data1/raw/emg_%s.csv"%(csv_file), "r") as f:
            reader = csv.reader(f)
            emg = []
            for row in reader:
                emg.append(row)
            emg = np.array(emg)
        #print('csv file name in class variable : '+str(self.csv_file)) # no attribute self.csv_file
        #eeg = np.array(pickle.load(open("%s/pkl/eeg_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #emg = np.array(pickle.load(open("%s/pkl/rms_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(self.data_dir,csv_file))['value']
        #emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        #if clearDataFlag:
        #eeg = scipy.io.loadmat("%s/pkl/eeg_%s.mat"%(self.data_dir,csv_file))['value']
        #emg = scipy.io.loadmat("%s/pkl/rms_%s.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        #else:
        #    eeg = scipy.io.loadmat("%s/pkl/Noisy/eeg_%s.mat"%(self.data_dir,csv_file))['value']
        #    emg = scipy.io.loadmat("%s/pkl/Noisy/rms_%s.mat"%(self.data_dir,csv_file))['value'] # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
        eeg = np.reshape(eeg,(-1,5000)) # for 20 second in the sampling frequency is 250Hz.
        emg = np.reshape(emg,(-1,5000))
        #if clearDataFlag:
        #labels = pd.read_csv("%s/training_20s/%s_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised save folder
        #    #print('data_loading : %s'%self.csv_file)
        #else:
        #    #labels = pd.read_csv("%s/training_20s/madeByLO/%s_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised save folder
        labels = pd.read_csv("/data1/training_20s/%s_training.csv"%(csv_file))
        labels[labels=="W"] = 0
        labels[labels=="NR"] = 1
        labels[labels=="R"] = 2
        return eeg[:upperLimit], emg[:upperLimit], labels[:upperLimit]

    def _load_csv_list_files2(self, csv_file):
        """Load data and labels from list of npz files."""
        eeg = []
        emg = []
        labels = []
        fs = None
        for csv_f in [csv_file]:
            sys.stdout.write("\rLoading {} ...".format(csv_f))
            sys.stdout.flush()
            tmp_eeg, tmp_emg, tmp_labels = self._load_csv_file2(csv_f)

            # Reshape the data to match the input of the model - conv2d
            tmp_eeg = tmp_eeg[:, :, np.newaxis, np.newaxis]
            tmp_emg = tmp_emg[:, :, np.newaxis, np.newaxis]

            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_eeg = tmp_eeg.astype(np.float32)
            tmp_emg = tmp_emg.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            eeg.append(tmp_eeg)
            emg.append(tmp_emg)
            labels.append(tmp_labels)

        return eeg, emg, labels

    def load_cv_data2(self,list_files,is_train):

        # Load a csv file
        print "Load data set:"
        eeg, emg, label = self._load_csv_list_files2(list_files)
        print " "

        if is_train:
            print "Training set: n_subjects={}".format(len(eeg))
            n_train_examples = 0
            for d in eeg[:3]:
                print d.shape
                n_train_examples += d.shape[0]
            print "Number of examples = {}".format(n_train_examples)
            print_n_samples_each_class(np.hstack(label))
            print " "
        else:
            print "Validation set: n_subjects={}".format(len(eeg))
            n_valid_examples = 0
            for d in eeg:
                print d.shape
                n_valid_examples += d.shape[0]
            print "Number of examples = {}".format(n_valid_examples)
            print_n_samples_each_class(np.hstack(label))
            print " "

        return eeg, emg, label

    # 2019/7/8 Leo Ota added function to validate the accuracy.
    def load_csv_data_valid(self, files, is_train):
        # load a npz file
        print("Start Loading csv files for validation test: ")
        print(files)
        for no_file, fname in enumerate(files):
            print("File Number: "+str(no_file))
            print("File Name: "+str(fname))
            if no_file == 0:
                eeg, emg, label = self._load_csv_list_files(fname)
                print(eeg)
                print(emg)
                print(label)
                print(eeg.shape)
                print(emg.shape)
                print(len(label))
                #eeg = eeg.reshape(1,5000,1,1)
                #emg = emg.reshape(1,5000,1,1)
                label = np.array(label)
            else:
                eeg_tmp, emg_tmp, label_tmp = self._load_csv_list_files(fname)
                label_tmp = np.array(label)
                eeg = np.concatenate([eeg,eeg_tmp], axis=0)
                emg = np.concatenate([emg,emg_tmp], axis=0)
                label = np.concatenate([label,label_tmp], axis=0)
        print(" ")
        # Reshape the data to match the input of the model - conv1d
        eeg = eeg.astype(np.float32)
        emg = emg.astype(np.float32)
        label = label.astype(np.int32)
        print("data set: {}, {}, {}".format(eeg.shape, emg.shape, label.shape))
        print_n_samples_each_class(label)
        print(" ")
        return eeg, emg, label


    @staticmethod
    def load_subject_data(data_dir, subject_idx, mouse):
        mouse = mouse[subject_idx]

        def remove_noise(x):
            N = 5000
            dt = 0.004
            freq = np.linspace(0, 1.0/dt, N)
            F = fftpack.fft(x)
            F[(freq < 2)] = 0
            F[(freq > 1/(dt*2))] = 0
            #F[(freq > 70)] = 0
            f = fftpack.ifft(F)
            f = np.real(f*2)
            return f

        def load_sub_file(csv_file,data_dir):
            print('csv file name : '+str(csv_file))
            with open("/data1/raw/eeg_%s.csv"%(csv_file), "rb") as f:
                eeg = pickle.load(f)
            #with open("/data1/raw/eeg_%s.csv"%(csv_file), "r") as f:
            #    reader = csv.reader(f)
            #    eeg = []
            #    for row in reader:
            #        eeg.append(row)
            #    eeg = np.array(eeg)
            with open("/data1/raw/emg_%s.csv"%(csv_file), "rb") as f:
                emg = pickle.load(f)
            #with open("/data1/raw/emg_%s.csv"%(csv_file), "r") as f:
            #    reader = csv.reader(f)
            #    emg = []
            #    for row in reader:
            #        emg.append(row)
            #    emg = np.array(emg)
            #print('csv file name in class variable : '+str(self.csv_file))
            #eeg = np.array(pickle.load(open("%s/pkl/eeg_%04d.pkl"%(data_dir,csv_file),'r'))).reshape(-1,5000)
            #emg = np.array(pickle.load(open("%s/pkl/rms_%04d.pkl"%(data_dir,csv_file),'r'))).reshape(-1,5000)
            #eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(data_dir,csv_file))['value'].reshape(-1,5000)
            #emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(data_dir,csv_file))['value'].reshape(-1,5000) # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
            #if clearDataFlag:
            #eeg = scipy.io.loadmat("%s/pkl/eeg_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000)
            #emg = scipy.io.loadmat("%s/pkl/rms_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000) # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
            #else:
            #    eeg = scipy.io.loadmat("%s/pkl/Noisy/eeg_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000)
            #    emg = scipy.io.loadmat("%s/pkl/Noisy/rms_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000) # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
            #removed = np.zeros(eeg.shape)
            #print eeg[0][:10]
            #t1 = time.time()
            #for i,x in enumerate(eeg):
            #    removed[i] = remove_noise(x)
            #t2 = time.time()
            #print t2 - t1
            #eeg = removed
            #print eeg[0][:10]
            #if clearDataFlag:
            #    #labels = pd.read_csv("%s/training_20s/%04d_training.csv"%(data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised.
            #labels = pd.read_csv("%s/training_20s/%s_training.csv"%(data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised.
            #    #print('data_loading : %s'%self.csv_file)
            #else:
            #    labels = pd.read_csv("%s/training_20s/madeByLO/%s_training.csv"%(data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised.
            with open("/data1/mouse_4313/pkl/stg20s_%s.pkl"%(csv_file), "rb") as f:
                labels = pickle.load(f)
            #labels = pd.read_csv("/data1/training_20s/%s_training.csv"%(csv_file))
            #labels[labels=="W"] = 0
            #labels[labels=="NR"] = 1
            #labels[labels=="R"] = 2
            return eeg, emg, labels

        def load_sub_list_files(csv_file,data_dir):
            """Load data and labels from list of npz files."""
            eeg = []
            emg = []
            labels = []
            fs = None
            for csv_f in [csv_file]:
                sys.stdout.write("\rLoading {} ...".format(csv_f))
                sys.stdout.flush()
                tmp_eeg, tmp_emg, tmp_labels = load_sub_file(csv_f,data_dir)

                # Reshape the data to match the input of the model - conv2d
                tmp_eeg = tmp_eeg[:, :, np.newaxis, np.newaxis]
                tmp_emg = tmp_emg[:, :, np.newaxis, np.newaxis]

                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_eeg = tmp_eeg.astype(np.float32)
                tmp_emg = tmp_emg.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                eeg.append(tmp_eeg)
                emg.append(tmp_emg)
                labels.append(tmp_labels)

            return eeg, emg, labels

        print "Load data from: {}".format(mouse)
        eeg,emg, labels = load_sub_list_files(mouse,data_dir)

        return eeg,emg, labels

    # 20190806 LO added to load noisy data.
    @staticmethod
    def load_subject_data6(data_dir, mouse):
        ''' input data format is mat signal files and csv stage file. '''
        ''' only for 20 second stage information. '''
        mouse = mouse

        def remove_noise(x):
            #N = 5000
            N = 1000
            dt = 0.004
            freq = np.linspace(0, 1.0/dt, N)
            F = fftpack.fft(x)
            F[(freq < 2)] = 0
            F[(freq > 1/(dt*2))] = 0
            #F[(freq > 70)] = 0
            f = fftpack.ifft(F)
            f = np.real(f*2)
            return f

        def load_sub_file(csv_file,data_dir):
            with open("/data1/raw/eeg_%s.csv"%(csv_file), "r") as f:
                reader = csv.reader(f)
                eeg = []
                for row in reader:
                    eeg.append(row)
                eeg = np.array(eeg)
            with open("/data1/raw/emg_%s.csv"%(csv_file), "r") as f:
                reader = csv.reader(f)
                emg = []
                for row in reader:
                    emg.append(row)
                emg = np.array(emg)
            #if not clearDataFlag:
            #with open("%s/pkl/Noisy/eeg_%s.pkl"%(data_dir,csv_file), "rb") as fd:
            #    eeg = pickle.load(fd)
            #with open("%s/pkl/Noisy/rms_%s.pkl"%(data_dir,csv_file), "rb") as fd:
            #    emg = pickle.load(fd)
            #else:
            #eeg = scipy.io.loadmat("%s/pkl/Noisy/eeg_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000)
            #emg = scipy.io.loadmat("%s/pkl/Noisy/rms_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000) # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
            #if upperLimit==4500:
            #    eeg = np.array(eeg)[0:int(math.floor(float(eeg.shape[0])/float(1000))*1000)]
            #    emg = np.array(emg)[0:int(math.floor(float(emg.shape[0])/float(1000))*1000)]#,1]
            #    eeg = np.array(eeg.flatten()).reshape(eeg.shape[0]//1000, 1000)
            #    emg = np.array(emg.flatten()).reshape(emg.shape[0]//1000, 1000)
            #elif upperLimit==8500 or upperLimit==4300:
            #    #eeg = np.array(eeg)[0:int(math.floor(float(eeg.shape[0])/float(5000))*5000)]
            #    #emg = np.array(emg)[0:int(math.floor(float(emg.shape[0])/float(5000))*5000)]#,1]
            #    #eeg = np.array(eeg.flatten()).reshape(eeg.shape[0]//5000, 5000)
            #    #emg = np.array(emg.flatten()).reshape(emg.shape[0]//5000, 5000)
            epoch_num = eeg.shape[0]
            if upperLimit < epoch_num:
                eeg = eeg[0:upperLimit, :]
                emg = emg[0:upperLimit, :]
            #with open("%s/training/madeByLO/%s_training.pkl"%(data_dir,csv_file), "rb") as fd:
            #    labels = pickle.load(fd)
            #labels = pd.read_csv("%s/training_20s/madeByLO/%s_training.csv"%(data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised.
            labels = pd.read_csv("/data1/training_20s/%s_training.csv"%(csv_file))
            labels[labels=="W"] = 0
            labels[labels=="NR"] = 1
            labels[labels=="R"] = 2
            if upperLimit < epoch_num:
                labels = labels[0:upperLimit]
            return eeg, emg, labels

        def load_sub_list_files(csv_file,data_dir):
            """Load data and labels from list of npz files."""
            eeg = []
            emg = []
            labels = []
            fs = None
            for csv_f in csv_file:#[csv_file]:
                sys.stdout.write("\rLoading {} ...".format(csv_f))
                sys.stdout.flush()
                tmp_eeg, tmp_emg, tmp_labels = load_sub_file(csv_f,data_dir)

                # Reshape the data to match the input of the model - conv2d
                tmp_eeg = tmp_eeg[:, :, np.newaxis, np.newaxis]
                tmp_emg = tmp_emg[:, :, np.newaxis, np.newaxis]

                # Casting
                tmp_eeg = tmp_eeg.astype(np.float32)
                tmp_emg = tmp_emg.astype(np.float32)
                tmp_labels = np.array(tmp_labels).astype(np.int32)

                eeg.append(tmp_eeg)
                emg.append(tmp_emg)
                labels.append(tmp_labels)
            return eeg,emg, labels

        print "Load data from: {}".format(mouse)
        eeg,emg, labels = load_sub_list_files(mouse,data_dir)

        return eeg,emg, labels

    @staticmethod
    def load_subject_data7(data_dir, mouse):
        ''' input data format is mat signal files and csv stage file. '''
        ''' only for 20 second stage information. '''
        mouse = mouse

        # It seems not to be use.
        def remove_noise(x):
            #N = 5000
            N = 1000
            dt = 0.004
            freq = np.linspace(0, 1.0/dt, N)
            F = fftpack.fft(x)
            F[(freq < 2)] = 0
            F[(freq > 1/(dt*2))] = 0
            #F[(freq > 70)] = 0
            f = fftpack.ifft(F)
            f = np.real(f*2)
            return f

        def load_sub_file(csv_file,data_dir):
            with open("/data1/mouse_4313/pkl/eeg_%s.pkl"%(csv_file), "rb") as f: #raw/eeg_%s.csv"%(csv_file), "r") as f:
                eeg = pickle.load(f) # reader = csv.reader(f)
                #eeg = []
                #for row in reader:
                #    eeg.append(row)
                #eeg = np.array(eeg)
            with open("/data1/mouse_4313/pkl/emg_%s.pkl"%(csv_file), "rb") as f: #raw/emg_%s.csv"%(csv_file), "r") as f:
                emg = pickle.load(f) # reader = csv.reader(f)
                #emg = []
                #for row in reader:
                #    emg.append(row)
                #emg = np.array(emg)
            #if not clearDataFlag:
            #with open("%s/pkl/Noisy/eeg_%s.pkl"%(data_dir,csv_file), "rb") as fd:
            #    eeg = pickle.load(fd)
            #with open("%s/pkl/Noisy/rms_%s.pkl"%(data_dir,csv_file), "rb") as fd:
            #    emg = pickle.load(fd)
            #else:
            #    #eeg = scipy.io.loadmat("%s/pkl/eeg_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000)
            #    #emg = scipy.io.loadmat("%s/pkl/rms_%s.mat"%(data_dir,csv_file))['value'].reshape(-1,5000) # 20190325 LO revised. These 'pkl' files were the result of preprocess of moving root mean square using 'mRMS.py'.
            #if upperLimit==4500:
            #    eeg = np.array(eeg)[0:int(math.floor(float(eeg.shape[0])/float(1000))*1000)]
            #    emg = np.array(emg)[0:int(math.floor(float(emg.shape[0])/float(1000))*1000)]#,1]
            #    eeg = np.array(eeg.flatten()).reshape(eeg.shape[0]//1000, 1000)
            #    emg = np.array(emg.flatten()).reshape(emg.shape[0]//1000, 1000)
            #elif upperLimit==8500 or upperLimit==4300:
            #    #eeg = np.array(eeg)[0:int(math.floor(float(eeg.shape[0])/float(5000))*5000)]
            #    #emg = np.array(emg)[0:int(math.floor(float(emg.shape[0])/float(5000))*5000)]#,1]
            #    #eeg = np.array(eeg.flatten()).reshape(eeg.shape[0]//5000, 5000)
            #    #emg = np.array(emg.flatten()).reshape(emg.shape[0]//5000, 5000)
            epoch_num = eeg.shape[0]
            if upperLimit < epoch_num:
                eeg = eeg[0:upperLimit, :]
                emg = emg[0:upperLimit, :]
            #if clearDataFlag:
            #labels = pd.read_csv("%s/training_20s/madeByLO/%s_training.csv"%(data_dir,csv_file), header = 0, index_col=0).values.flatten()p
            with open("/data1/mouse_4313/pkl/stg20s_%s.pkl"%(csv_file), "r") as f:
                labels = pickle.load(f)
            #labels = pd.read_csv("/data1/training_20s/%s_training.csv"%(csv_file))
            #labels[labels=="W"] = 0
            #labels[labels=="NR"] = 1
            #labels[labels=="R"] = 2
            #    #with open("%s/training_20s/madeByLO/%s_training.pkl"%(data_dir,csv_file), "rb") as fd:
            #    #    labels = pickle.load(fd)
            #else:
            #    #labels = pd.read_csv("%s/training_20s/madeByLO/%s_training.csv"%(data_dir,csv_file), header = 0, index_col=0).values.flatten() # 20190322 LO revised.
            #    #labels[labels=="W"] = 0
            #    #labels[labels=="NR"] = 1
            #    #labels[labels=="R"] = 2
            if upperLimit < epoch_num:
                labels = labels[0:upperLimit]
            return eeg, emg, labels

        def load_sub_list_files(csv_file,data_dir):
            """Load data and labels from list of npz files."""
            eeg = []
            emg = []
            labels = []
            fs = None
            for csv_f in csv_file:#[csv_file]:
                sys.stdout.write("\rLoading {} ...".format(csv_f))
                sys.stdout.flush()
                tmp_eeg, tmp_emg, tmp_labels = load_sub_file(csv_f,data_dir)

                # Reshape the data to match the input of the model - conv2d
                tmp_eeg = tmp_eeg[:, :, np.newaxis, np.newaxis]
                tmp_emg = tmp_emg[:, :, np.newaxis, np.newaxis]

                # Casting
                tmp_eeg = tmp_eeg.astype(np.float32)
                tmp_emg = tmp_emg.astype(np.float32)
                tmp_labels = np.array(tmp_labels).astype(np.int32)

                eeg.append(tmp_eeg)
                emg.append(tmp_emg)
                labels.append(tmp_labels)
            return eeg,emg, labels

        print "Load data from: {}".format(mouse)
        eeg,emg, labels = load_sub_list_files(mouse,data_dir)

        return eeg,emg, labels
