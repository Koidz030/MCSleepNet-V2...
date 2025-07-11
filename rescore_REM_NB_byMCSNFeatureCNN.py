import os
import glob
import time
import csv
import pandas as pd
import numpy as np
import pickle as pk
from datetime import datetime
from datetime import datetime as dt
from deepsleep.sleep_stage import (NUM_CLASSES, EPOCH_SEC_LEN, SAMPLING_RATE)

import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from deepsleep.nn import *
from deepsleep.model import DeepFeatureNet
from deepsleep.data_loader import NonSeqDataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report

np.random.seed(0)
tf.set_random_seed(1)

from train_NB import customDeepFeatureNet

def main(argv=None):
    # Definition of Naive Bayes Classifier
    NBmodel=GaussianNB()
    with open("/home/juan/mc-sleepnet_incrementallearning/outputNB_params_13.pkl", "rb") as f:
        NBmodel=pk.load(f)
    # Training Naive Bayes Classifier
    sess=tf.Session()
    #with tf.Graph().as_default(), tf.Session() as sess:
    # Definition of MCSNmodel (MC-SleepNet)
    MCSNmodel=customDeepFeatureNet(batch_size=1,input_dims=20*250,n_classes=3,is_train=False,reuse_params=False,use_dropout=True)
    # batch size: 1 sleep epoch
    # input_dims: 20 scond, 250 Hz sampling
    # n_class: number of classes is 3 (Wake, NR, R
    # is_train: False (Because it is just prediction output of the feature map of MC-SleepNet CNN part)
    # use_dropout: This part is ignored, because of the is_train is set to False. 
    # Initialization operation (make place holders, etc. ) of MCSNmodel (MC-SleepNet)
    MCSNmodel.init_ops()
    # Load 672 mice for training. 
    #with open("/data1/mouse_4313/retrainEXP_valid_datalist.csv", "r") as f: #Here I changed the path to load the 168 mice sleep recordings
    with open("/data1/mouse_4313/retrain_datalist.csv", "r") as f: #Here I changed the path to load the 168 mice sleep recordings => 2021/3/1 LO revised to 168 mice data list.
        reader=csv.reader(f)
        data=[]
        for row in reader:
            data.extend(row)
    data=list(np.array(data).astype(np.int32))
    # Get trainable variables of the pretrained, and new ones
    train_vars1 = [v for v in tf.all_variables() # trainable_variables()
                        if v.name.find("conv")!=-1] # in train_params] # .replace(train_net.name, "network") in train_params]
    # Create a saver object
    saver = tf.train.Saver(train_vars1, max_to_keep=None)
    # Initialize variables in the graph
    sess.run(tf.initialize_all_variables())
    #train_vars2 = list(set(tf.trainable_variables())-set(train_vars1))
    # load model file. 
    saver.restore(sess, "/home/ota/cross_1/fold1/deepsleepnet/model_epoch19.ckpt-20")
    # Make save directory
    save_dir_name="/home/juan/mc-sleepnet_incrementallearning/output/result_CNN_gaussNB_20210301/"
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    #data_loader=NonSeqDataLoader(data_dir_mat="/data2/data/", data_dir_pkl="/data1/mouse_4313/") # 2.6 sec / a record, 
    #x_train=[]
    y_pred_mcsn_featureCNN=[]
    for i in range(672): # Devided 50 mice data block for one training update calculation. 
        #for i in range(3): # Devided 50 mice data block for one training update calculation. => 2021/3/1 LO revised to adapt to 168 mice training. 
        #eeg, emg, y_train = data_loader.load_cv_data(files=data[i], is_train=False) # 31 GB /62 GB
        # MCSN features
        #for j in range(eeg.shape[0]):
        #    if len(x_train)==0:
        #        x_train=sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )})
        #    elif len(x_train)>0:
        #        x_train=np.concatenate([x_train, sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )})], axis=0)
        # 2021/2/22 LO changed to the list comprehension from python plain for loop. 
        #x_train=np.array([sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )}) for j in range(eeg.shape[0])])
        #X=x_train.reshape(x_train.shape[0],-1)
        #pred_tmp=NBmodel.predict(X)
        #class_prob=NBmodel.predict_proba(X)
        if len(y_pred_mcsn_featureCNN)==0:
            #prob=class_prob
            #pred=pred_tmp
            y_pred_mcsn_featureCNN=np.load("/data2/mcsleepnet_incrementalLearning/output_20210201featurenet/output_subject_%s.npz"%(data[i]))
            y_pred_mcsn_featureCNN=y_pred_mcsn_featureCNN["y_pred"]
            y_true=y_pred_mcsn_featureCNN["y_true"]
        else:
            #prob=np.concatenate([prob, class_prob], axis=0)
            #pred=np.concatenate([pred, pred_tmp], axis=0)
            y_pred_mcsn_featureCNN=np.concatenate([y_pred_mcsn_featureCNN, np.load("/data2/mcsleepnet_incrementalLearning/output_20210201featurenet/output_subject_%s.npz"%(data[i]))], axis=0)
            y_true=np.concatenate([y_true, y_pred_mcsn_featureCNN["y_true"]], axis=0)

    pred=np.load("pred_certainty_NBmodel.npz")["pred"]
    #REMprob=np.load("pred_certainty_NBmodel.npz")["REMprob"]
    # cases 1
    pred[pred==2]=y_pred_mcsn_featureCNN[pred==2] # All of REM for rescoring
    # cases 2
    #pred[REMprob[:,2]<0.95]=y_pred_mcsn_featureCNN[REMprob[:,2]<0.95]
    cm=confusion_matrix(y_true, pred)
    acc=accuracy_score(y_true, pred)
    kappa=cohen_kappa_score(y_true, pred)
    report=classification_report(y_true, pred)
    print("accuracy: ", acc)
    print("Kappa: ", kappa)
    print(cm)
    print(report)

if __name__=="__main__":
    main()

