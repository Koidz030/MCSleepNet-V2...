import os
import glob
import time
import csv
import pandas as pd
import numpy as np
import pickle
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

class customDeepFeatureNet(DeepFeatureNet):
    def __init__(
        self, 
        batch_size, 
        input_dims, 
        n_classes, 
        is_train, 
        reuse_params, 
        use_dropout, 
        name="deepsleepnet"
    ):#naivebayes"):

        super (self.__class__, self).__init__(
            batch_size=batch_size, 
            input_dims=input_dims, 
            n_classes=n_classes, 
            is_train=is_train, 
            reuse_params=reuse_params, 
            use_dropout=use_dropout, 
            name=name
        )

        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.use_dropout = use_dropout
        self.name = name
        
    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_eeg_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs_eeg"
        )
        self.input_emg_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs_emg"
        )
        # Target
        self.target_var = tf.placeholder(
            tf.int32,
            shape=[self.batch_size],
            name=name + "_targets"
        )
    def build_model(self, input_eeg_var, input_emg_var):
        # List to store the output of each calculation
        output_conns = []
        
        ######### EEG CNNs with small filter size at the first layer #########
        # Convolution
        network=self._conv1d_layer(input_var=input_eeg_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)
        
        # Max pooling
        name="l{}_pool".format(self.layer_idx)
        network=max_pool_1d(name=name, input_var=network, pool_size=8, stride=8)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1
        
        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1
        output_conns.append(network)
        
        ######### EMG CNNs with large filter size at the first layer #########
        
        # Convolution
        network = self._conv1d_layer(input_var=input_emg_var, filter_size=500, n_filters=64, stride=50)
        
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1
        
        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
       
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=2, stride=2)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1
        output_conns.append(network)
        
        
        ######### Aggregate and link two CNNs #########
        
        # Concat
        name = "l{}_concat".format(self.layer_idx)
        network = tf.concat(1, output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1
        
        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1
        return network


    def init_ops(self):
        self._build_placeholder()
        
        # Get loss and prediction operations
        with tf.variable_scope(self.name) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()
            # Build model
            network = self.build_model(input_eeg_var=self.input_eeg_var, input_emg_var=self.input_emg_var)
            self.features = network

def main(argv=None):
    # Definition of Naive Bayes Classifier
    NBmodel=GaussianNB()
    
    # Training Naive Bayes Classifier
    sess=tf.Session()
    #with tf.Graph().as_default(), tf.Session() as sess:
    
    # Definition of MCSNmodel (MC-SleepNet)
    MCSNmodel=customDeepFeatureNet(
        batch_size=1,
        input_dims=20*250,
        n_classes=3,
        is_train=False,
        reuse_params=False,
        use_dropout=True
    )
    # batch size: 1 sleep epoch
    # input_dims: 20 scond, 250 Hz sampling
    # n_class: number of classes is 3 (Wake, NR, R
    # is_train: False (Because it is just prediction output of the feature map of MC-SleepNet CNN part)
    # use_dropout: This part is ignored, because of the is_train is set to False. 
    
    # Initialization operation (make place holders, etc. ) of MCSNmodel (MC-SleepNet)
    MCSNmodel.init_ops()
    
    # Load 672 mice for training. 
    with open("/data1/mouse_4313/retrainEXP_valid_datalist.csv", "r") as f: #2021/5/17 JN Changed to path of 672 datalist
    #with open("/data1/mouse_4313/retrain_datalist.csv", "r") as f: #Here I changed the path to load the 168 mice sleep recordings => 2021/3/1 LO revised to 168 mice data list.
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
    save_dir_name="/home/juan/mc-sleepnet_incrementallearning/output/result_CNN_gaussNB_20210518/"
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    data_loader=NonSeqDataLoader(data_dir_mat="/data2/data/", data_dir_pkl="/data1/mouse_4313/") # 2.6 sec / a record, 
    
    #x_train=[]
    y_pred_train=[]
    y_true_train=[]
    
    #for i in range(13): # Devided 50 mice data block for one training update calculation. 
    for i in range(13): # Devided 50 mice data block for one training update calculation. => 2021/3/1 LO revised to adapt to 168 mice training. 
        eeg, emg, y_train = data_loader.load_cv_data(files=data[i*50:(i+1)*50], is_train=False) # 31 GB /62 GB
        # MCSN features
        #for j in range(eeg.shape[0]):
        #    if len(x_train)==0:
        #        x_train=sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )})
        #    elif len(x_train)>0:
        #        x_train=np.concatenate([x_train, sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )})], axis=0)
        # 2021/2/22 LO changed to the list comprehension from python plain for loop. 
        x_train=np.array([sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )}) for j in range(eeg.shape[0])])
        x_train=x_train.reshape(x_train.shape[0],-1)
        #np.savez_compressed("/data2/mcsleepnet_incrementalLearning/output_20210215_incrementalLearn_NB/x_train_%d.npz"%(i), x_train)
        #if len(features)==0:
        #    features=x_train
        #elif len(features)>0:
        #    featues=np.concatenate([features, x_train], axis=0)
        # Fit model
        NBmodel.partial_fit(x_train, y_train, np.unique(y_train)) # It takes about a minite.
        # Save the model check point file
        with open(os.path.join(save_dir_name,"NB_params_%d.pkl"%(i)), "wb") as f:
            pickle.dump(NBmodel, f)
        # For checking
        y_pred_tmp=NBmodel.predict(x_train) # This line for several minites (probably 5 minutes)
        if len(y_pred_train)==0:
            y_pred_train=y_pred_tmp
            y_true_train=y_train
        elif len(y_pred_train)!=0:
            y_pred_train=np.concatenate([y_pred_train, y_pred_tmp], axis=0)
            y_true_train=np.concatenate([y_true_train, y_train], axis=0)
        cm=confusion_matrix(y_true_train, y_pred_train)
        acc=accuracy_score(y_true_train, y_pred_train)
        kappa=cohen_kappa_score(y_true_train, y_pred_train)
        print("check point ", str(i))
        print("Batch partial fit %d: acc.: %f, kappa: %f"%(i, acc, kappa))
        print("confusion matrix: ")
        print(cm)
    i+=1

    eeg, emg, y_train = data_loader.load_cv_data(files=data[i*50:],is_train=False) # from 650 to 672 (22 mice), probably use about 15 GB memory
    #x_train=np.concatenate([x_train, sess.run(MCSNmodel.features, feed_dict={network.input_eeg_var:eeg, network.input_emg:emg, network.target_var: y_train})], axis=0)
    # Please reshape input (batch size, timestep, 1(damy axis to reduce dimension), 1 channel)
    x_train=np.array([sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )}) for j in range(eeg.shape[0])])
    x_train=x_train.reshape(x_train.shape[0],-1)
    #np.savez_compressed("/data2/mcsleepnet_incrementalLearning/output_20210215_incrementalLearn_NB/x_train_%d.npz"%(i), x_train)
    # Fit model
    NBmodel.partial_fit(x_train, y_train, np.unique(y_train))
    
    # Save the model check point file (Final version)
    with open(os.path.join(save_dir_name,"NB_params_%d.pkl"%(i)), "wb") as f:
        pickle.dump(NBmodel, f)
    #with open("/data2/mcsleepnet_incrementalLearning/output_20210215_incrementalLearn_NB/NB_params_%d.pkl"%(i), "rb") as f:
    #    NBmodel=pickle.load(f)
    #print("predicted stage for training data: ", NBmodel.predict(x_train), ", true label: ", y_train[0])
    
    y_pred_train=np.concatenate([y_pred_train, NBmodel.predict(x_train)], axis=0)
    y_true_train=np.concatenate([y_true_train, y_train], axis=0)
    cm=confusion_matrix(y_true_train, y_pred_train)
    acc=accuracy_score(y_true_train, y_pred_train)
    kappa=cohen_kappa_score(y_true_train, y_pred_train)
    result_tr=classification_report(y_true_train, y_pred_train)
    print("check the result for training data. ")
    print(cm)
    #[[10488   375   222]
    # [  192 10220   187]
    # [   12    11  1082]]
    print("training accuracy: ", acc, "training cohen kappa score: ", kappa)
    print(result_tr)
    # ('training accuracy: ', 0.95616306112598182, 'training cohen kappa score: ', 0.92060804467746982)
    # save model parameters
    #a=NBmodel.get_params()
    #np.savez_compressed("/data2/mcsleepnet_incrementalLearning/output_20210215_incrementalLearn_NB/NB_params.npz", a)
    
    
    #################################################################
    # Validation part (Testing the model)
    # Load 168 mice for test(validation). 
    with open("/data1/mouse_4313/retrain_datalist.csv", "r") as f: #2021/5/17 JN Changed to path of 168 datalist
    #with open("/data1/mouse_4313/retrainEXP_valid_datalist.csv", "r") as f: #Here I exchanged the file with the 672 mice sleep recordings => 2021/3/1 LO revised to 672 mice list. 
        reader=csv.reader(f)
        data=[]
        for row in reader:
            data.extend(row)
    data=np.array(data).astype(np.int32)
    y_pred_valid=[]
    y_true_valid=[]
    for i in range(3):
        eeg, emg, y_valid = data_loader.load_cv_data(files=data[i*50:(i+1)*50], is_train=False)
        x_valid=np.array([sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1,5000,1,1), MCSNmodel.input_emg_var:emg[j].reshape(1,5000,1,1), MCSNmodel.target_var: y_valid[j].reshape(1,)}) for j in range(eeg.shape[0])]) # probably about 21 minutes
        x_valid=x_valid.reshape(x_valid.shape[0], -1)
        if len(y_pred_valid)==0:
            y_pred_valid=NBmodel.predict(x_valid)
            y_true_valid=y_valid
        elif len(y_pred_valid)!=0:
            y_pred_valid=np.concatenate([y_pred_valid, NBmodel.predict(x_valid)], axis=0)
            y_true_valid=np.concatenate([y_true_valid, y_valid], axis=0)
    i+=1
    eeg, emg, y_valid = data_loader.load_cv_data(files=data[i*50:], is_train=False) # from 150 to 168 (18 mice) => 2021/3/1 LO comment. This line load data from 651st data to 672nd data(21 mice).
    x_valid=np.array([sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1,5000,1,1), MCSNmodel.input_emg_var:emg[j].reshape(1,5000,1,1), MCSNmodel.target_var: y_valid[j].reshape(1,)}) for j in range(eeg.shape[0])]) # probably about 20 minutes
    x_valid=x_valid.reshape(x_valid.shape[0], -1)
    y_pred_valid=np.concatenate([y_pred_valid, NBmodel.predict(x_valid)], axis=0)
    y_true_valid=np.concatenate([y_true_valid, y_valid], axis=0)
    cm_val=confusion_matrix(y_true_valid, y_pred_valid)
    acc_val=accuracy_score(y_true_valid, y_pred_valid)
    kappa_val=cohen_kappa_score(y_true_valid, y_pred_valid)
    print("confusion matrix for validation data")
    print(cm_val)
    print("validation accuracy: ", acc_val, "validation cohen kappa score: ", kappa_val)
    print(classification_report(y_true_valid, y_pred_valid))
    result=classification_report(y_true_valid, y_pred_valid)
    
    # save the result
    with open(os.path.join(save_dir_name,"result_tr_val_cm_acc_kappa.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["acc_tr", "kappa_tr", "acc_val", "kappa_val"])
        writer.writerow([acc, kappa, acc_val, kappa_val])
        writer.writerow(["training confusion matrix: "])
        writer.writerows(cm)
        writer.writerow(["validation(test) confusion matrix: "])
        writer.writerows(cm_val)
        #writer.writerow([result.split('\n')[0][:12], result.split('\n')[0][12:27], result.split('\n')[0][27:35], result.split('\n')[0][35:45], result.split('\n')[0][45:]])
        writer.writerow(["", "precision tr.", "recall tr.", "f1-score tr.", "support"])
        writer.writerow([int(float(result_tr.split('\n')[2][:12])), float(result_tr.split('\n')[2][12:27]), float(result_tr.split('\n')[2][27:35]), float(result_tr.split('\n')[2][35:45]), int(result_tr.split('\n')[2][45:])])
        writer.writerow([int(float(result_tr.split('\n')[3][:12])), float(result_tr.split('\n')[3][12:27]), float(result_tr.split('\n')[3][27:35]), float(result_tr.split('\n')[3][35:45]), int(result_tr.split('\n')[3][45:])])
        writer.writerow([int(float(result_tr.split('\n')[4][:12])), float(result_tr.split('\n')[4][12:27]), float(result_tr.split('\n')[4][27:35]), float(result_tr.split('\n')[4][35:45]), int(result_tr.split('\n')[4][45:])])
        writer.writerow([result_tr.split('\n')[6][:12], float(result_tr.split('\n')[6][12:27]), float(result_tr.split('\n')[6][27:35]), float(result_tr.split('\n')[6][35:45]), int(result_tr.split('\n')[6][45:])])
        writer.writerow([result_tr.split('\n')[7][:12], float(result_tr.split('\n')[7][12:27]), float(result_tr.split('\n')[7][27:35]), float(result_tr.split('\n')[7][35:45]), int(result_tr.split('\n')[7][45:])])
        writer.writerow([result_tr.split('\n')[8][:12], float(result_tr.split('\n')[8][12:27]), float(result_tr.split('\n')[8][27:35]), float(result_tr.split('\n')[8][35:45]), int(result_tr.split('\n')[8][45:])])
        writer.writerow(["", "precision val.", "recall val.", "f1-score val.", "support"])
        writer.writerow([int(float(result.split('\n')[2][:12])), float(result.split('\n')[2][12:27]), float(result.split('\n')[2][27:35]), float(result.split('\n')[2][35:45]), int(result.split('\n')[2][45:])])
        writer.writerow([int(float(result.split('\n')[3][:12])), float(result.split('\n')[3][12:27]), float(result.split('\n')[3][27:35]), float(result.split('\n')[3][35:45]), int(result.split('\n')[3][45:])])
        writer.writerow([int(float(result.split('\n')[4][:12])), float(result.split('\n')[4][12:27]), float(result.split('\n')[4][27:35]), float(result.split('\n')[4][35:45]), int(result.split('\n')[4][45:])])
        writer.writerow([result.split('\n')[6][:12], float(result.split('\n')[6][12:27]), float(result.split('\n')[6][27:35]), float(result.split('\n')[6][35:45]), int(result.split('\n')[6][45:])])
        writer.writerow([result.split('\n')[7][:12], float(result.split('\n')[7][12:27]), float(result.split('\n')[7][27:35]), float(result.split('\n')[7][35:45]), int(result.split('\n')[7][45:])])
        writer.writerow([result.split('\n')[8][:12], float(result.split('\n')[8][12:27]), float(result.split('\n')[8][27:35]), float(result.split('\n')[8][35:45]), int(result.split('\n')[8][45:])])
    sess.close()
    del sess

if __name__=="__main__":
    main()


#################
"""
# To load the model file of Naive Bayes classifier, use this command. 
with open("/data2/mcsleepnet_incrementalLearning/output_20210215_incrementalLearn_NB/NB_params_13.pkl", "rb") as f:
    NBmodel=pickle.load(f)
"""

"""
# 2021/2/22 LO check the result. 
# Check the result for 50 training mice
    # data set: (431571, 5000, 1, 1), (431571, 5000, 1, 1), (431571,)
    # W: 222208
    # NR: 188738
    # R: 20625
print(cm)
array([[209652,   6602,   5954],
       [  4768, 174858,   9112]
       [   418,    650,  19557]])
acc
0.93627004594840713
kappa
0.88523666885517804
print(result_tr)
              precision    recall  f1-score   support
              0       0.98      0.94      0.96    216446(W)
              1       0.96      0.93      0.94    194074(NR)
              2       0.56      0.95      0.71     21072(R <- REM precision and f1 score value is low, especially in precision
      micro avg       0.94      0.94      0.94    431592
      macro avg       0.83      0.94      0.87    431592
   weighted avg       0.95      0.94      0.94    431592

############################3
# Check the result for 50 mice
# validation
431592 sleep epochs
acc_val
0.91028100613542418
kappa_val
0.84112888982555356
print(result)
              precision    recall  f1-score   support
              0       0.97      0.90      0.94    216446(W)
              1       0.94      0.92      0.93    194074(NR)
              2       0.47      0.93      0.63     21072(R) <- This value is low, especially in precision
      micro avg       0.91      0.91      0.91    431592
      macro avg       0.79      0.92      0.83    431592
   weighted avg       0.93      0.91      0.92    431592
"""

"""
# Please connect to GPU server at first.
# Copy files to your home directory
cp -R /home/ota/mc-sleepnet_incrementallearning /home/juan
# make tmux session on the GPU server
tmux
# open python interactively. 
python
# After close ssh session from note PC and reconnect to the GPU server, you can attach to the tmux session by following command.
tmux a

############################

# 2021/03/02

# Results of training with 168 and testing with 672 mice datasets

# Testing with 672 

data set: (189853, 5000, 1, 1), (189853, 5000, 1, 1), (189853,)
W: 95452
NR: 85112
R: 9289

confusion matrix for validation data
[[2680837  126550  120943]
 [  74871 2326867  193830]
 [   5640   11274  259705]]
('validation accuracy: ', 0.90809301998425318, 'validation cohen kappa score: ', 0.83727873307078249)

              precision    recall  f1-score   support

           0       0.97      0.92      0.94   2928330
           1       0.94      0.90      0.92   2595568
           2       0.45      0.94      0.61    276619

   micro avg       0.91      0.91      0.91   5800517
   macro avg       0.79      0.92      0.82   5800517
weighted avg       0.93      0.91      0.92   5800517

#############################

# 2021/05/18

# Results of training with 672 and testing with 168 mice datasets

# Testing with 168 

data set: (155358, 5000, 1, 1), (155358, 5000, 1, 1), (155358, 1)
W: 78008
NR: 70133
R: 7217

confusion matrix for validation data
[[676935  29629  20568]
 [ 18142 600494  34747]
 [  1318   3945  64325]]
('validation accuracy: ', 0.92528185928861606, 'validation cohen kappa score: ', 0.86614799074607718)
              precision    recall  f1-score   support

           0       0.97      0.93      0.95    727132
           1       0.95      0.92      0.93    653383
           2       0.54      0.92      0.68     69588

   micro avg       0.93      0.93      0.93   1450103
   macro avg       0.82      0.92      0.85   1450103
weighted avg       0.94      0.93      0.93   1450103



"""
