import numpy as np
import pandas as pd
from xgboost import XGBRegressor as XGBreg
import csv

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report

with open('/data1/mouse_4313/retrainEXP_valid_datalist.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        datalist=row

folder="/data2/mcsleepnet_incrementalLearning/output_20210301_retrain_5epoch/"
remprob=[]
nremprob=[]
wprob=[]
for fname in datalist:
    data=np.load(folder+"output_subject_%s.npz"%(fname))
    if len(wprob)==0:
        remprob=data['R_prob'][0]
        nremprob=data['NR_prob'][0]
        wprob=data['W_prob'][0]
    else:
        remprob=np.concatenate([remprob, data['R_prob'][0][data['y_pred'][0]==2]])
        nremprob=np.concatenate([nremprob, data['NR_prob'][0][data['y_pred'][0]==1]])
        wprob=np.concatenate([wprob, data['W_prob'][0][data['y_pred'][0]==0]])

# Result of the NB model
df_=pd.read_csv("/home/juan/mc-sleepnet_incrementallearning/NBmodel_outputs_true_pred_prob.csv")
# total epoch: 5610663

# Result of the retraining condition
remprob_all=[]
nremprob_all=[]
wprob_all=[]
y_true_all=[]
Wprob=[]
NRprob=[]
Rprob=[]
for fname in datalist:
    data=np.load(folder+"output_subject_%s.npz"%(fname))
    if len(wprob)==0:
        if len(data['R_prob'][0])==len(df_[df_["id"]==int(fname)]["REM"]):
            remprob_all=data['R_prob'][0]
            nremprob_all=data['NR_prob'][0]
            wprob_all=data['W_prob'][0]
            y_true_all=data['y_true'][0]
            Wprob=df_[df_["id"]==int(fname)]["WAKE"]
            NRprob=df_[df_["id"]==int(fname)]["NREM"]
            Rprob=df_[df_["id"]==int(fname)]["REM"]
        elif len(data['R_prob'][0])>len(df_[df_["id"]==int(fname)]["REM"]):
            length_min=len(df_[df_["id"]==int(fname)]["REM"])
            remprob_all=data['R_prob'][0][:length_min]
            nremprob_all=data['NR_prob'][0][:length_min]
            wprob_all=data['W_prob'][0][:length_min]
            y_true_all=data['y_true'][0][:length_min]
            Wprob=df_[df_["id"]==int(fname)]["WAKE"]
            NRprob=df_[df_["id"]==int(fname)]["NREM"]
            Rprob=df_[df_["id"]==int(fname)]["REM"]
        elif len(data['R_prob'][0])<len(df_[df_["id"]==int(fname)]["REM"]):
            length_min=len(data['R_prob'][0])
            remprob_all=data['R_prob'][0]
            nremprob_all=data['NR_prob'][0]
            wprob_all=data['W_prob'][0]
            y_true_all=data['y_true'][0]
            Wprob=df_[df_["id"]==int(fname)]["WAKE"][:length_min]
            NRprob=df_[df_["id"]==int(fname)]["NREM"][:length_min]
            Rprob=df_[df_["id"]==int(fname)]["REM"][:length_min]
    else:
        if len(data['R_prob'][0])==len(df_[df_["id"]==int(fname)]["REM"]):
            remprob_all=np.concatenate([remprob_all, data['R_prob'][0]], axis=0)
            nremprob_all=np.concatenate([nremprob_all, data['NR_prob'][0]], axis=0)
            wprob_all=np.concatenate([wprob_all, data['W_prob'][0]], axis=0)
            y_true_all=np.concatenate([y_true_all, data['y_true'][0]], axis=0)
            Wprob=np.concatenate([Wprob, df_[df_["id"]==int(fname)]["WAKE"]], axis=0)
            NRprob=np.concatenate([NRprob, df_[df_["id"]==int(fname)]["NREM"]], axis=0)
            Rprob=np.concatenate([Rprob, df_[df_["id"]==int(fname)]["REM"]], axis=0)
        elif len(data['R_prob'][0])>len(df_[df_["id"]==int(fname)]["REM"]):
            length_min=len(df_[df_["id"]==int(fname)]["REM"])
            remprob_all=np.concatenate([remprob_all, data['R_prob'][0][:length_min]], axis=0)
            nremprob_all=np.concatenate([nremprob_all, data['NR_prob'][0][:length_min]], axis=0)
            wprob_all=np.concatenate([wprob_all, data['W_prob'][0][:length_min]], axis=0)
            y_true_all=np.concatenate([y_true_all, data['y_true'][0][:length_min]], axis=0)
            Wprob=np.concatenate([Wprob, df_[df_["id"]==int(fname)]["WAKE"]], axis=0)
            NRprob=np.concatenate([NRprob, df_[df_["id"]==int(fname)]["NREM"]], axis=0)
            Rprob=np.concatenate([Rprob, df_[df_["id"]==int(fname)]["REM"]], axis=0)
        elif len(data['R_prob'][0])<len(df_[df_["id"]==int(fname)]["REM"]):
            length_min=len(data['R_prob'][0])
            remprob_all=np.concatenate([remprob_all, data['R_prob'][0]], axis=0)
            nremprob_all=np.concatenate([nremprob_all, data['NR_prob'][0]], axis=0)
            wprob_all=np.concatenate([wprob_all, data['W_prob'][0]], axis=0)
            y_true_all=np.concatenate([y_true_all, data['y_true'][0]], axis=0)
            Wprob=np.concatenate([Wprob, df_[df_["id"]==int(fname)]["WAKE"][:length_min]], axis=0)
            NRprob=np.concatenate([NRprob, df_[df_["id"]==int(fname)]["NREM"][:length_min]], axis=0)
            Rprob=np.concatenate([Rprob, df_[df_["id"]==int(fname)]["REM"][:length_min]], axis=0)

# total epoch: 5796000


# Consolidate the outputs
prob_lstm=np.concatenate([wprob_all.reshape(-1,1), nremprob_all.reshape(-1,1), remprob_all.reshape(-1,1)], axis=1)
prob_nb=np.concatenate([Wprob.reshape(-1,1), NRprob.reshape(-1,1), Rprob.reshape(-1,1)], axis=1)

X, y=([prob_lstm, prob_nb],y_true_all)

################################################
# XGBoost consolidation model
################################################

#Establish parameters
param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', 'num_class': 3}

#Defining the model of XGBoost
model = XGBreg()

#Fitting the model on the whole dataset
model.fit(X, y)

#Saving the model
model.save_model('Consolidate_output_XGB.model')

#################################################

#Resutls
cm=confusion_matrix(y, X)
acc=accuracy_score(y, X)
kappa=cohen_kappa_score(y, X)
report=classification_report(y, X)

print("accuracy: ", acc)
print("Kappa: ", kappa)
print("confusion_matrix: ")
print(cm) 
print(report)

#########################
"""


"""