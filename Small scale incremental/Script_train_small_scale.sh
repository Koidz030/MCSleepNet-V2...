#!/bin/sh

#  Script_train_small_scale.sh
#  
#
#  Created by Leo Ota on 2021/06/14.
#
#sed --i.bak 's/trainlist[0]/trainlist[0]/g' deepsleep/trainer.py
CUDA_VISIBLE_DEVICES='0' python train_smallScale_LSTMupdate.py 1

#sed --i.bak 's/trainlist[0]/trainlist[training_step_number-1]/g' deepsleep/trainer.py
sed --i.bak 's/trainlist[0]/trainlist[2-1]/g' deepsleep/trainer.py
#    param_file_path = '/home/ota/mc-sleepnet_incrementallearning/output/retrain20210611/fold1/deepsleepnet/params_epoch19.npz'
sed --i.bak 's/ota\/mc-sleepnet_incrementallearning\/output\/retrain20210611\/fold1/juan\/mc-sleepnet_incrementallearning\/output\/retrain20210611\/session1/g' train_smallScale.py
CUDA_VISIBLE_DEVICES='0' python train_smallScale_LSTMupdate.py 2

sed --i.bak 's/trainlist[2-1]/trainlist[3-1]/g' deepsleep/trainer.py
CUDA_VISIBLE_DEVICES='0' python train_smallScale_LSTMupdate.py 3

sed --i.bak 's/trainlist[3-1]/trainlist[4-1]/g' deepsleep/trainer.py
CUDA_VISIBLE_DEVICES='0' python train_smallScale_LSTMupdate.py 4

j=4
for i in $(eval echo {5..20})
do
sed --i.bak 's/trainlist[$j-1]/trainlist[$i-1]/g' deepsleep/trainer.py
sed --i.bak 's/juan\/mc-sleepnet_incrementallearning\/output\/retrain20210611\/session$j/juan\/mc-sleepnet_incrementallearning\/output\/retrain20210611\/session$i/g' train_smallScale.py
j=$i
CUDA_VISIBLE_DEVICES='0' python train_smallScale_LSTMupdate.py $i
done

