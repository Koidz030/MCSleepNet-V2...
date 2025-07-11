import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from deepsleep.sleep_stage import (NUM_CLASSES, EPOCH_SEC_LEN, SAMPLING_RATE)


#df_REMprob=pd.DataFrame({"REM":REMprob, "y_pred":pred, "y_true":true})
df_REMprob=pd.read_csv('NBmodel_outputs_true_pred_REMprob.csv')
x1=df_REMprob[(df_REMprob["y_pred"]==2)&(df_REMprob["y_true"]==0)]["REM"]
x2=df_REMprob[(df_REMprob["y_pred"]==2)&(df_REMprob["y_true"]==1)]["REM"]
x3=df_REMprob[(df_REMprob["y_pred"]==2)&(df_REMprob["y_true"]==2)]["REM"]
fig = plt.figure(figsize=(13,8))
ax=fig.add_subplot(1,1,1)
ax.hist((x1, x2, x3), bins=20, color=["blue", "green", "red"], label=["W", "NR", "R"], stacked=False, log=False, normed=False)
plt.xlabel("Certainty of REM")
plt.ylabel("Frequency [#]")
ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize=14)
plt.title("Predicted label = REM")
plt.savefig("hist_REM_prob_v2.png")
