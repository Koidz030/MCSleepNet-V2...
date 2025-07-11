import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from deepsleep.sleep_stage import (NUM_CLASSES, EPOCH_SEC_LEN, SAMPLING_RATE)


#df_REMprob=pd.DataFrame({"REM":REMprob, "y_pred":pred, "y_true":true})
df_WAKEprob=pd.read_csv('NBmodel_outputs_true_pred_prob.csv')
x1=df_WAKEprob[(df_WAKEprob["y_pred"]==0)&(df_WAKEprob["y_true"]==0)]["WAKE"]
x2=df_WAKEprob[(df_WAKEprob["y_pred"]==0)&(df_WAKEprob["y_true"]==1)]["WAKE"]
x3=df_WAKEprob[(df_WAKEprob["y_pred"]==0)&(df_WAKEprob["y_true"]==2)]["WAKE"]
fig = plt.figure(figsize=(13,8))
ax=fig.add_subplot(1,1,1)
ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
plt.ylim([0, 5000])
plt.xlabel("Certainty ")
plt.ylabel("Frequency [#]")
ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
plt.title("Predicted label = WAKE")
plt.savefig("hist_WAKE_prob_v2.png")

df_NREMprob=pd.read_csv('NBmodel_outputs_true_pred_prob.csv')
x1=df_NREMprob[(df_NREMprob["y_pred"]==1)&(df_NREMprob["y_true"]==0)]["NREM"]
x2=df_NREMprob[(df_NREMprob["y_pred"]==1)&(df_NREMprob["y_true"]==1)]["NREM"]
x3=df_NREMprob[(df_NREMprob["y_pred"]==1)&(df_NREMprob["y_true"]==2)]["NREM"]
fig = plt.figure(figsize=(13,8))
ax=fig.add_subplot(1,1,1)
ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
plt.ylim([0, 5000])
plt.xlabel("Certainty ")
plt.ylabel("Frequency [#]")
ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
plt.title("Predicted label = NREM")
plt.savefig("hist_NREM_prob_v2.png")

df_REMprob=pd.read_csv('NBmodel_outputs_true_pred_prob.csv')
x1=df_REMprob[(df_REMprob["y_pred"]==2)&(df_REMprob["y_true"]==0)]["REM"]
x2=df_REMprob[(df_REMprob["y_pred"]==2)&(df_REMprob["y_true"]==1)]["REM"]
x3=df_REMprob[(df_REMprob["y_pred"]==2)&(df_REMprob["y_true"]==2)]["REM"]
fig = plt.figure(figsize=(13,8))
ax=fig.add_subplot(1,1,1)
ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
plt.ylim([0, 5000])
plt.xlabel("Certainty of REM")
plt.ylabel("Frequency [#]")
ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
plt.title("Predicted label = REM")
plt.savefig("hist_REM_prob_v2.png")