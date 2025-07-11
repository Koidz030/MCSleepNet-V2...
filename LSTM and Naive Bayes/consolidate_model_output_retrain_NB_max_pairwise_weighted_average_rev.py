import numpy as np
import pandas as pd
import csv

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report

with open('/data1/mouse_4313/retrain_datalist.csv') as f: #2021/5/25 JN Changed to path of 168 datalist
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
# 2021/3/24
# Results of analysis (unweighted average) 672 mice test

confusion_matrix:
[[2691098  124647   99716]
 [  76557 2373525  137422]
 [   5993   14329  255463]]

              precision    recall  f1-score   support

         0.0       0.97      0.92      0.95   2915461
         1.0       0.94      0.92      0.93   2587504
         2.0       0.52      0.93      0.66    275785

   micro avg       0.92      0.92      0.92   5778750
   macro avg       0.81      0.92      0.85   5778750
weighted avg       0.94      0.92      0.93   5778750


#########################

# 2021/3/25
# Results of analysis (weighted average (2)) 672 mice test

accuracy: 0.972
Kappa: 0.949

confusion_matrix:
[[2858263   48002    9196]
 [  38631 2515878   32995]
 [   2400   28717  244668]]

              precision    recall  f1-score   support

         0.0       0.99      0.98      0.98   2915461
         1.0       0.97      0.97      0.97   2587504
         2.0       0.85      0.89      0.87    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.94      0.95      0.94   5778750
weighted avg       0.97      0.97      0.97   5778750


#########################

# 2021/3/25
# Results of analysis (weighted average (3)) 672 mice test

accuracy: 0.973
Kappa: 0.950

confusion_matrix:
[[2858585   48764    8112]
 [  36131 2523765   27608]
 [   2304   32429  241052]]

              precision    recall  f1-score   support

         0.0       0.99      0.98      0.98   2915461
         1.0       0.97      0.98      0.97   2587504
         2.0       0.87      0.87      0.87    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.94      0.94      0.94   5778750
weighted avg       0.97      0.97      0.97   5778750

#########################

# 2021/3/26
# Results of analysis (weighted average (4)) 672 mice test

accuracy: 0.973
Kappa: 0.950

confusion_matrix:
[[2858382   49311    7768]
 [  35244 2526773   25487]
 [   2295   34247  239243]]

              precision    recall  f1-score   support

         0.0       0.99      0.98      0.98   2915461
         1.0       0.97      0.98      0.97   2587504
         2.0       0.88      0.87      0.87    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.94      0.94      0.94   5778750
weighted avg       0.97      0.97      0.97   5778750


#########################

# 2021/3/29
# Results of analysis (weighted average (5)) 672 mice test

accuracy: 0.973
Kappa: 0.950

confusion_matrix:
[[2858272   49599    7590]
 [  34748 2528355   24401]
 [   2302   35364  238119]]

              precision    recall  f1-score   support

         0.0       0.99      0.98      0.98   2915461
         1.0       0.97      0.98      0.97   2587504
         2.0       0.88      0.86      0.87    275785

   micro avg       0.97      0.97      0.97   5778750
   macro avg       0.95      0.94      0.94   5778750
weighted avg       0.97      0.97      0.97   5778750


#########################

# 2021/5/25




"""