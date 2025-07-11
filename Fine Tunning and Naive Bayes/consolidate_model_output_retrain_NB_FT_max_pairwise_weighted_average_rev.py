import numpy as np
import pandas as pd
import csv

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report

with open('/data1/mouse_4313/retrainEXP_valid_datalist.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        datalist=row

folder="/home/ota/mc-sleepnet_incrementallearning/output_finetune20210406/retraining2/output_20210426_fineTuene_20sepoch/"
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
            y_true_all=np.concatenate([y_true_all, data['y_true'][0][:length_min].flatten()], axis=0)
            Wprob=np.concatenate([Wprob, df_[df_["id"]==int(fname)]["WAKE"]], axis=0)
            NRprob=np.concatenate([NRprob, df_[df_["id"]==int(fname)]["NREM"]], axis=0)
            Rprob=np.concatenate([Rprob, df_[df_["id"]==int(fname)]["REM"]], axis=0)
        elif len(data['R_prob'][0])<len(df_[df_["id"]==int(fname)]["REM"]):
            length_min=len(data['R_prob'][0])
            remprob_all=np.concatenate([remprob_all, data['R_prob'][0]], axis=0)
            nremprob_all=np.concatenate([nremprob_all, data['NR_prob'][0]], axis=0)
            wprob_all=np.concatenate([wprob_all, data['W_prob'][0]], axis=0)
            y_true_all=np.concatenate([y_true_all, data['y_true'][0].flatten()], axis=0)
            Wprob=np.concatenate([Wprob, df_[df_["id"]==int(fname)]["WAKE"][:length_min]], axis=0)
            NRprob=np.concatenate([NRprob, df_[df_["id"]==int(fname)]["NREM"][:length_min]], axis=0)
            Rprob=np.concatenate([Rprob, df_[df_["id"]==int(fname)]["REM"][:length_min]], axis=0)

# total epoch: 5796000


# Consolidate the outputs
def define_stage_forEachSleepEpoch(prob1, prob2):
    # Find max value of an average between two terms
    average=0.33*((2*prob1)+prob2)
    #average=0.25*((3*prob1)+prob2)
    #average=0.2*((4*prob1)+prob2)
    argmax_average=np.argmax(average)
    return argmax_average


prob_lstm=np.concatenate([wprob_all.reshape(-1,1), nremprob_all.reshape(-1,1), remprob_all.reshape(-1,1)], axis=1)
prob_nb=np.concatenate([Wprob.reshape(-1,1), NRprob.reshape(-1,1), Rprob.reshape(-1,1)], axis=1)

y_pred_consol=[define_stage_forEachSleepEpoch(prob1, prob2) for prob1, prob2 in zip(prob_lstm, prob_nb)]

cm=confusion_matrix(y_true_all, y_pred_consol)
acc=accuracy_score(y_true_all, y_pred_consol)
kappa=cohen_kappa_score(y_true_all, y_pred_consol)
report=classification_report(y_true_all, y_pred_consol)

print("accuracy: ", acc)
print("Kappa: ", kappa)
print("confusion_matrix: ")
print(cm) 
print(report)

#########################
"""
# 2021/4/27
# Results of analysis (unweighted average)

Accuracy: 0.92
Kappa: 0.85

confusion_matrix:
[[2688200  124715  102546]
 [  76607 2373463  137434]
 [   5990   14350  255445]]

              precision    recall  f1-score   support

         0.0       0.97      0.92      0.95   2915461
         1.0       0.94      0.92      0.93   2587504
         2.0       0.52      0.93      0.66    275785

   micro avg       0.92      0.92      0.92   5778750
   macro avg       0.81      0.92      0.85   5778750
weighted avg       0.94      0.92      0.93   5778750

#########################

# 2021/4/27
# Results of analysis (weighted average (2))

Accuracy: 0.966
Kappa: 0.937

confusion_matrix:
[[2846488   58157   10816]
 [  53819 2514859   18826]
 [   2264   50785  222736]]

              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98   2915461
         1.0       0.96      0.97      0.97   2587504
         2.0       0.88      0.81      0.84    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.94      0.92      0.93   5778750
weighted avg       0.97      0.97      0.97   5778750



#########################

# 2021/4/27
# Results of analysis (weighted average (3))

Accuracy: 0.965
Kappa: 0.936

confusion_matrix:
[[2846219   59550    9692]
 [  53930 2518183   15391]
 [   2152   56422  217211]]

              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98   2915461
         1.0       0.96      0.97      0.96   2587504
         2.0       0.90      0.79      0.84    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.94      0.91      0.93   5778750
weighted avg       0.97      0.97      0.97   5778750


#########################

# 2021/4/27
# Results of analysis (weighted average (4))

Accuracy: 0.965
Kappa: 0.936

confusion_matrix:
[[2845980   60144    9337]
 [  54295 2519015   14194]
 [   2151   58990  214644]]
              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98   2915461
         1.0       0.95      0.97      0.96   2587504
         2.0       0.90      0.78      0.84    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.95      0.91      0.93   5778750
weighted avg       0.97      0.97      0.97   5778750

"""