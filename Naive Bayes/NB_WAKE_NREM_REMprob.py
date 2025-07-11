import os
import csv
import matplotlib
import numpy as np
import pickle as pk
import pandas as pd
import tensorflow as tf
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from deepsleep.sleep_stage import (NUM_CLASSES, EPOCH_SEC_LEN, SAMPLING_RATE)
from train_NB import customDeepFeatureNet
from deepsleep.data_loader import NonSeqDataLoader

"""
with open("/data2/mcsleepnet_incrementalLearning/output_20210215_incrementalLearn_NB/NB_params_13.pkl", "rb") as f:
    NBmodel=pickle.load(f)

input X = 
class_prob=gaussNB.predict_proba(X)
REMprob=class_prob[:, 2]
fig = plt.figure(figsize=(6,4))
ax=fig.add_subplot(1,1,1)
ax.hist(df_REMprob['REMprob'])
ax.hist(REMprob)
plt.savefig("hist_REMprob.png")
"""

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
    # input_dims: 20 scond, 250 Hz samplingf
    # n_class: number of classes is 3 (Wake, NR, R)
    # is_train: False (Because it is just prediction output of the feature map of MC-SleepNet CNN part)
    # use_dropout: This part is ignored, because of the is_train is set to False. 
    # Initialization operation (make place holders, etc. ) of MCSNmodel (MC-SleepNet)
    MCSNmodel.init_ops()
    # Load 672 mice for training. 
    with open("/data1/mouse_4313/retrainEXP_valid_datalist.csv", "r") as f: #Here I changed the path to load the 168 mice sleep recordings
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
    save_dir_name="/home/juan/mc-sleepnet_incrementallearning/output/result_CNN_gaussNB_20210301/"
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    data_loader=NonSeqDataLoader(data_dir_mat="/data2/data/", data_dir_pkl="/data1/mouse_4313/") # 2.6 sec / a record, 
    #x_train=[]
    id_mouse=[]
    WAKEprob=[]
    REMprob=[]
    NREMprob=[]
    for i in range(672): # Devided 50 mice data block for one training update calculation.
        #for i in range(3): # Devided 50 mice data block for one training update calculation. => 2021/3/1 LO revised to adapt to 168 mice training. 
        eeg, emg, y_train = data_loader.load_cv_data(files=data[i:(i+1)], is_train=False) # 31 GB /62 GB
        # MCSN features
        #for j in range(eeg.shape[0]):
        #    if len(x_train)==0:
        #        x_train=sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )})
        #    elif len(x_train)>0:
        #        x_train=np.concatenate([x_train, sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )})], axis=0)
        # 2021/2/22 LO changed to the list comprehension from python plain for loop. 
        x_train=np.array([sess.run(MCSNmodel.features, feed_dict={MCSNmodel.input_eeg_var:eeg[j].reshape(1, 5000, 1, 1), MCSNmodel.input_emg_var:emg[j].reshape(1, 5000, 1, 1), MCSNmodel.target_var: y_train[j].reshape(1, )}) for j in range(eeg.shape[0])])
        X=x_train.reshape(x_train.shape[0],-1)
        pred_tmp=NBmodel.predict(X)
        class_prob=NBmodel.predict_proba(X)
        
        if len(WAKEprob)==0:
            WAKEprob=class_prob[:,0].flatten()
            pred=pred_tmp.flatten()
            true=y_train.flatten()
            id_mouse=np.repeat(int(data[i]), len(true))

        elif len(NREMprob)==0:
            NREMprob=class_prob[:,1].flatten()           
            pred=pred_tmp.flatten()
            true=y_train.flatten()
            id_mouse=np.repeat(int(data[i]), len(true))
        
        elif len(REMprob)==0:
            REMprob=class_prob[:,2].flatten()
            pred=pred_tmp.flatten()
            true=y_train.flatten()
            id_mouse=np.repeat(int(data[i]), len(true))

        else:
            WAKEprob=np.concatenate([WAKEprob, class_prob[:,0].flatten()], axis=0)
            NREMprob=np.concatenate([NREMprob, class_prob[:,1].flatten()], axis=0)
            REMprob=np.concatenate([REMprob, class_prob[:,2].flatten()], axis=0)
            pred=np.concatenate([pred, pred_tmp.flatten()], axis=0)
            true=np.concatenate([true, y_train.flatten()], axis=0)
            id_mouse=np.concatenate([id_mouse, np.repeat(int(data[i]), len(y_train.flatten()))])
    
    df_WAKEprob=pd.DataFrame({"WAKE":WAKEprob, "y_pred":pred, "y_true":true})
    x1=df_WAKEprob[(df_WAKEprob["y_pred"]==0)&(df_WAKEprob["y_true"]==0)]["WAKE"]
    x2=df_WAKEprob[(df_WAKEprob["y_pred"]==0)&(df_WAKEprob["y_true"]==1)]["WAKE"]
    x3=df_WAKEprob[(df_WAKEprob["y_pred"]==0)&(df_WAKEprob["y_true"]==2)]["WAKE"]
    fig = plt.figure(figsize=(13,8))
    ax=fig.add_subplot(1,1,1)
    ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
    plt.xlabel("Certainty ")
    plt.ylabel("Frequency [#]")
    ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
    plt.title("Predicted label = WAKE")
    plt.savefig("hist_WAKE_prob.png")
    #np.savez_compressed("pred_certainty_NBmodel.npz", REMprob=REMprob, pred=pred)
    #df_WAKEprob.to_csv("NBmodel_outputs_true_pred_WAKEprob.csv")

    df_NREMprob=pd.DataFrame({"NREM":NREMprob, "y_pred":pred, "y_true":true})
    x1=df_NREMprob[(df_NREMprob["y_pred"]==1)&(df_NREMprob["y_true"]==0)]["NREM"]
    x2=df_NREMprob[(df_NREMprob["y_pred"]==1)&(df_NREMprob["y_true"]==1)]["NREM"]
    x3=df_NREMprob[(df_NREMprob["y_pred"]==1)&(df_NREMprob["y_true"]==2)]["NREM"]
    fig = plt.figure(figsize=(13,8))
    ax=fig.add_subplot(1,1,1)
    ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
    plt.xlabel("Certainty ")
    plt.ylabel("Frequency [#]")
    ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
    plt.title("Predicted label = NREM")
    plt.savefig("hist_NREM_prob.png")
    #np.savez_compressed("pred_certainty_NBmodel.npz", REMprob=REMprob, pred=pred)
    #df_NREMprob.to_csv("NBmodel_outputs_true_pred_NREMprob.csv")
    df_prob=pd.DataFrame({"id":id_mouse, "WAKE":WAKEprob, "NREM":NREMprob, "REM":REMprob, "y_pred":pred, "y_true":true})
    df_prob.to_csv("NBmodel_outputs_true_pred_prob.csv")
    x1=df_prob[(df_prob["y_pred"]==2)&(df_prob["y_true"]==0)]["REM"]
    x2=df_prob[(df_prob["y_pred"]==2)&(df_prob["y_true"]==1)]["REM"]
    x3=df_prob[(df_prob["y_pred"]==2)&(df_prob["y_true"]==2)]["REM"]
    fig = plt.figure(figsize=(13,8))
    ax=fig.add_subplot(1,1,1)
    ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
    plt.xlabel("Certainty ")
    plt.ylabel("Frequency [#]")
    ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
    plt.title("Predicted label = REM")
    plt.savefig("hist_REM_prob.png")


if __name__ == "__main__":
    main()
