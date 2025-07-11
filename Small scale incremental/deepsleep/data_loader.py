import os

import numpy as np
import pandas as pd
import pickle
import scipy.io
from deepsleep.sleep_stage import print_n_samples_each_class
from deepsleep.utils import get_balance_class_downsample
from deepsleep.utils import get_balance_class_oversample

from deepsleep.sleep_stage import (class_dict)

import re
import sys
import psutil
import sqlite3

class NonSeqDataLoader(object):

    def __init__(self, data_dir_mat, data_dir_pkl, upperLimit):
        self.data_dir_mat = data_dir_mat
        self.data_dir_pkl = data_dir_pkl
        self.upperLimit = upperLimit

    def _load_csv_file(self, csv_file):
        #print csv_file
        eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(self.data_dir_mat,int(csv_file)))['value']
        emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(self.data_dir_mat,int(csv_file)))['value']
        #with open("%s/pkl/eeg_%04d.pkl"%(self.data_dir,csv_file), 'rb') as f:
        #    eeg = pickle.load(f)
        #with open("%s/pkl/rms_%04d.pkl"%(self.data_dir,csv_file), 'rb') as f:
        #    emg = pickle.load(f)
        eeg = np.reshape(np.array(eeg),(-1,5000))
        emg = np.reshape(np.array(emg),(-1,5000))
        #labels = pd.read_csv("%s/training/%04d_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten()
        #with open("%s/pkl/%04d_training.pkl"%(self.data_dir,csv_file), 'rb') as f:
        #    labels = pd.read_csv(f)
        labels = np.array(pickle.load(open("/data1/mouse_4313/pkl/stg20s_%04d.pkl"%(csv_file),'r'))).astype(np.int32).reshape(-1,1)
        #a = np.load("/data2/output/mc-sleep_pred/cross1/output_subject_%d.npz"%csv_file)
        #print len(a["y_pred"])
        #pos = np.where((a["y_pred"] == 1) & (a["NR_prob"]<0.95))[1]
        #eeg = eeg[pos]
        #emg = emg[pos]
        #labels = labels[pos]

        #eeg = pd.DataFrame(eeg,index=None,columns=None)
        #eeg['label'] = labels
        #eeg = eeg[eeg.label != 'W']
        #del eeg['label']
        #eeg = np.array(eeg)
        #emg = pd.DataFrame(emg,index=None,columns=None)
        #emg['label'] = labels
        #emg = emg[emg.label!='W']
        #labels = emg['label']
        #del emg['label']
        #emg = np.array(emg)

        #labels[labels=="NR"] =1
        #labels[labels=="R"] = 2
        #labels[labels=="W"] = 0

        return list(eeg[:self.upperLimit]), list(emg[:self.upperLimit]), list(labels[:self.upperLimit])

    def _load_csv_list_files(self, csv_files):
        """Load data and labels from list of npz files."""
        eeg = []
        emg = []
        labels = []

        for csv_f in csv_files:
            sys.stdout.write("\rLoding %04d\n"%csv_f)
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

        #Use balanced-class, oversample training set
        if is_train:
            eeg, emg, label = get_balance_class_oversample(
                x1=eeg, x2=emg, y=label
            )
            print "oversampled training set: {}, {}, {}".format(
                eeg.shape, emg.shape, label.shape
            )
            print_n_samples_each_class(label)
            print " "


        return eeg, emg, label


class SeqDataLoader(object):

    def __init__(self, data_dir_mat, data_dir_pkl, upperLimit):
        self.data_dir_mat = data_dir_mat
        self.data_dir_pkl = data_dir_pkl
        self.upperLimit = upperLimit

    def _load_csv_file(self, csv_file):
        #eeg = np.array(pickle.load(open("%s/pkl/eeg_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #emg = np.array(pickle.load(open("%s/pkl/rms_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #eeg = np.array(pickle.load(open("%s/pkl/eeg_%04d.pkl"%(self.data_dir,csv_file),'r')))
        #emg = np.array(pickle.load(open("%s/pkl/rms_%04d.pkl"%(self.data_dir,csv_file),'r')))
        eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(self.data_dir_mat,int(csv_file)))['value']
        emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(self.data_dir_mat,int(csv_file))['value']
        eeg = np.reshape(eeg,(-1,5000))
        emg = np.reshape(emg,(-1,5000))
        #labels = pd.read_csv("%s/training/%04d_training.csv"%(self.data_dir,csv_file), header = 0, index_col=0).values.flatten()
        #with open("%s/training/%04d_training.csv"%(self.data_dir,csv_file), 'rb') as f:
        #    labels = pd.read_csv(f)
        #labels = np.array(pickle.load(open("%s/pkl/stg20s_%04d.pkl"%(self.data_dir_pkl,csv_file),'r'))).astype(np.int32)
        labels = np.array(pickle.load(open("/data1/mouse_4313/pkl/stg20s_%04d.pkl" %(csv_file), 'r'))).astype(np.int32).reshape(-1, 1)
        #labels = np.array(pickle.load(open("/data1/mouse_4313/pkl/stg20s_%04d.pkl"%(self.data_dir_pkl,csv_file),'r'))).astype(np.int32)
        #labels = np.array(pickle.load(open("%s/pkl/stg20s_%04d.pkl"%(self.data_dir_pkl,csv_file),'r'))).astype(np.int32).reshape(-1,1)
        #eeg = pd.DataFrame(eeg,index=None,columns=None)
        #eeg['label'] = labels
        #eeg = eeg[eeg.label!='W']
        #del eeg['label']
        eeg = np.array(eeg)
        #emg = pd.DataFrame(emg,index=None,columns=None)
        #emg['label'] = labels
        #eeg = eeg[eeg.label!='W']
        #labels = emg['label']
        #del emg['label']
        emg = np.array(emg)

        #print emg.shape
        #print eeg.shape
        #print labels.shape
        labels = np.array(labels)

        #labels[labels=="NR"] = 0
        #labels[labels=="R"] = 1
        return eeg[:self.upperLimit], emg[:self.upperLimit], labels[:self.upperLimit]

    def _load_csv_list_files(self, csv_file):
        """Load data and labels from list of npz files."""
        eeg = []
        emg = []
        labels = []
        fs = None
        for csv_f in csv_file:
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

    # This function is called in retraining program.
    def load_cv_data(self,list_files,is_train):
        #print("check the data length.")
        #print(list_files)
        # Load a csv file
        #print "Load data set:"
        eeg, emg, label = self._load_csv_list_files(list_files)
        #print " "

        if is_train:
            print "Training set: n_subjects={}".format(len(eeg))
            n_train_examples = 0
            for d in eeg: # [:3]: # <= why does last 2 epochs removed?
                #print d.shape
                n_train_examples += d.shape[0]
            #print "Number of examples = {}".format(n_train_examples)
            print_n_samples_each_class(np.hstack(label))
            #print " "
        else:
            print "Validation set: n_subjects={}".format(len(eeg))
            n_valid_examples = 0
            for d in eeg:
                #print d.shape
                n_valid_examples += d.shape[0]
            #print "Number of examples = {}".format(n_valid_examples)
            print_n_samples_each_class(np.hstack(label))
            #print " "

        return eeg, emg, label

    # This function is called in prediction program.
    @staticmethod
    def load_subject_data(data_dir_pkl, subject_idx, mouse):
        mouse = mouse[subject_idx]

        def load_sub_file(csv_file, data_dir_pkl):
            preprocess_flag=0 # if preprocessing is operated, flag changes to 1.
            #eeg = scipy.io.loadmat("%s/pkl/eeg_%04d.mat"%(data_dir,csv_file))['value']
            #emg = scipy.io.loadmat("%s/pkl/rms_%04d.mat"%(data_dir,csv_file))['value']
            eeg = scipy.io.loadmat("/data2/data/pkl/eeg_%04d.mat"%(csv_file))['value']
            emg = scipy.io.loadmat("/data2/data/pkl/rms_%04d.mat"%(csv_file))['value']
            eeg = np.array(eeg).astype(np.float32).flatten()
            emg = np.array(emg).astype(np.float32).flatten()
            meanEEG=eeg.mean()
            stdEEG=eeg.std()
            meanEMG=emg.mean()
            stdEMG=emg.std()
            leneeg=len(eeg)
            lenemg=len(emg)
            '''
            if os.path.exists("%s/pkl/eeg_%04d.pkl"%(data_dir,csv_file)):
                with open("%s/pkl/eeg_%04d.pkl"%(data_dir,csv_file), 'rb') as f:
                    eeg = np.array(pickle.load(f)).astype(np.float32)
                with open("%s/pkl/rms_%04d.pkl"%(data_dir,csv_file), 'rb') as f:
                    emg = np.array(pickle.load(f)).astype(np.float32)
            elif os.path.exists("%s/raw/eeg_%04d.csv"%(data_dir,csv_file)):
                preprocess_flag=1 # preprocessing were done.
                eeg, emg = [], []
                with open("%s/raw/eeg_%04d.csv"%(data_dir,csv_file), 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        eeg.append(row[1])
                    eeg = np.array(eeg).astype(np.float32)
                with open("%s/raw/emg_%04d.pkl"%(data_dir,csv_file), 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        emg.append(row[1])
                    emg = np.array(emg).astype(np.float32)
                    emg = np.power(np.array(emg).flatten(), 2)
                    window_size=250
                    window=np.ones(window_size)/float(window_size)
                    rms = np.sqrt(np.convolve(emg, window, "same"))
                    rms = pd.Series(rms)
                # check the free disk
                hdd_data1 = psutil.disk_usage('/data1')
                if hdd_data1.free>600000000:
                    with open("%s/pkl/eeg_%04d.pkl"%(data_dir,csv_file), 'wb') as f:
                        pickle.dump(eeg, f, protocol=2)
                    with open("%s/pkl/rms_%04d.pkl"%(data_dir,csv_file), 'wb') as f:
                        pickle.dump(rms, f, protocol=2)
                else:
            '''
            eeg = np.reshape(eeg,(-1,5000))
            emg = np.reshape(emg,(-1,5000))
            print("check the existense of %s/training_20s/%04d_training.csv"%(data_dir_pkl,csv_file))
            print(os.path.exists("%s/training_20s/%04d_training.csv"%(data_dir_pkl,csv_file)))
            #if os.path.exists("%s/pkl/stg20s_%04d.pkl"%(data_dir,csv_file)):
            with open("%s/pkl/stg20s_%04d.pkl"%(data_dir_pkl,csv_file), 'rb') as f:
                labels = np.array(pickle.load(f)).astype(np.int32)
            '''elif os.path.exists("%s/training_20s/%04d_training.csv"%(self.data_dir_pkl,csv_file)):
                labels = pd.read_csv("%s/training_20s/%04d_training.csv"%(data_dir,csv_file), header=0, index_col=0).values.flatten()
                labels[labels=="W"] = 0
                labels[labels=="NR"] = 1
                labels[labels=="R"] = 2
                labels = np.array(labels).astype(np.int32)
                # check the free disk
                hdd_data1 = psutil.disk_usage('/data1')
                if hdd_data1.free>100000000: # preprocess_flag==1:
                    with open("%s/pkl/stg20s_%04d.pkl"%(data_dir,csv_file), 'wb') as f:
                        pickle.dump(labels, f, protocol=2)
                #pos = np.where(np.load("/data2/output/cross3/output_subject_%d.npz"%csv_file)["y_pred"] == 1)[1]
                #print pos
                #eeg = eeg[pos]
                #emg = emg[pos]
                #labels = labels[pos]
                conn = sqlite3.connect("/data1/mouse_4313/sqlite_mean_var_datalen_delta_theta.db")
                c = conn.cursor()'''
            """
                CREATE_TABLE='''CREATE TABLE IF NOT EXISTS retrainingTest20210202_mean_std
                (set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mouse_id INTEGER,
                meanEEG NUMERIC,
                meanEMG NUMERIC,
                stdEEG NUMERIC,
                stdEMG NUMERIC,
                lenEEG INTEGER,
                lenEMG INTEGER,
                stg20s INTEGER)
                '''
                c.execute(CREATE_TABLE)
                conn.commit()
                SELECT_DATA="SELECT count(*) FROM retrainingTest20210202_mean_std;"
                processed=c.execute(SELECT_DATA)
                for res in processed:
                    counter=int(res[0])
                print('counter: '+str(counter))
                meanEEG=eeg.mean()
                stdEEG=eeg.std()
                leneeg=len(eeg)
                meanEMG=emg.mean()
                stdEMG=emg.std()
                lenemg=emg.std()
                INSERT_DATA="INSERT INTO retrainingTest20210202_mean_std VALUES(%d,%d,%f,%f,%f,%f,%d,%d,%d);"%(
                    int(counter),int(csv_file),float(meanEEG),float(meanEMG),float(stdEEG),float(stdEMG),int(leneeg),int(lenemg),int(len(labels))
                )
            """

            return eeg, emg, labels

        def load_sub_list_files(csv_file, data_dir_pkl):
            """Load data and labels from list of npz files."""
            eeg = []
            emg = []
            labels = []
            fs = None
            for csv_f in [csv_file]:
                sys.stdout.write("\rLoading {} ...".format(csv_f))
                sys.stdout.flush()
                tmp_eeg, tmp_emg, tmp_labels = load_sub_file(csv_f, data_dir_pkl)

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
        eeg,emg, labels = load_sub_list_files(mouse,data_dir_pkl)

        return eeg,emg, labels
