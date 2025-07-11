#! /usr/bin/env python
# -*- coding: utf-8 -*-
from deepsleep.utils import iterate_batch_seq_minibatches, iterate_minibatches
import sys
#train_file_list = sys.argv[1] # str(input("Please input train file list name: "))
#valid_file_list = sys.argv[2] # str(input("Please input validation file list name: "))

###! Check here!
# sys.argv[1]: first command line arguement
# training step number 1, 2, ..., 21     # train_file_list[-1])
training_step_number = int(sys.argv[1])
if training_step_number > 1:
    #model_file_path = input("Please input model file path of model_epoch%d.npz: "%((training_step_number-1)*20))
    param_file_path = '/home/ota/mc-sleepnet_incrementallearning/output/retrain20210611/fold1/deepsleepnet/params_epoch19.npz'
    #input("Please input model file path of params_epoch%d.npz"%((training_step_number-1)*20))
else:
    #model_file_path = ""
    param_file_path = ""

import itertools
import os
import time
import pickle
import numpy as np
import tensorflow as tf

from deepsleep.trainer import DeepFeatureNetTrainer, DeepSleepNetTrainer
from deepsleep.model import DeepSleepNet, DeepFeatureNet # DeepSleepNet2
from deepsleep.optimize import adam, adam_clipping_list_lr, adam_clipping
from deepsleep.data_loader import SeqDataLoader
from deepsleep.nn import *

from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from progressbar import ProgressBar
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir_mat', '/data2/data/',
                           """Directory where to load eeg and rms signal data.""")
tf.app.flags.DEFINE_string('data_dir_pkl', '/data1/mouse_4313/',
                           """Directory where to stage 20 second label data.""")
# raw data directory: /data1/mouse_4313/raw/
# training for retraining:
# test
"""
f = open("/data1/mouse_4313/retrain_datalist.csv")
data = f.read()
data_list = data.replace("\r\n","").split(",")
trainlist = df_retrain_mouse = list(np.array(data_list).astype(np.int32)[0:16])
print("training mouse list: "+str(df_retrain_mouse))
f.close()
"""
# Juan-san's plan
#trainlist = pickle.load('train_subsets.pkl')
with open('train_subsets.pkl', 'rb') as f:
    trainlist=pickle.load(f)
# validation fo retraining: /data1/mouse_4313/retrainEXP_valid_datalist.csv
#f = open("/data1/mouse_4313/retrainEXP_valid_datalist.csv")
f=open('testlist1.csv', 'r')
data = f.read()
data_list = data.replace("\r\n","").split(",")
validlist = df_retrain_mouse_valid = list(np.array(data_list).astype(np.int32)) # [0:16])
#print("validation mouse list: "+str(df_retrain_mouse_valid))
f.close()

tf.app.flags.DEFINE_string('output_dir', 'output/retrain20210614/', # 1/', # 0129/',
                           """Directory where to save trained models """
                           """and outputs.""")
#tf.app.flags.DEFINE_string('model_dir', '/home/ota/cross_1/fold1/deepsleepnet/params_epoch19.npz', # model_epoch19.ckpt-20',
#                            """model load path of original MC-SleepNet learned model""")
#tf.app.flags.DEFINE_string('model_file', model_file_path, # '/home/ota/cross_1/fold1/deepsleepnet/model_epoch19.ckpt-20',
#                            """model load path of original MC-SleepNet learned model""")
#tf.app.flags.DEFINE_string('model_params', '/home/ota/cross_1/fold1/deepfeaturenet/params_epoch9.npz',
tf.app.flags.DEFINE_string('model_params', param_file_path, #'/home/ota/cross_1/fold1/deepsleepnet/params_epoch19.npz',
                            """model load path of original MC-SleepNet learned model""")
tf.app.flags.DEFINE_integer('n_folds', 1,
                           """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 1,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 10,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_integer('finetune_epochs', 20,
                            """Number of epochs for fine-tuning DeepSleepNet.""")
tf.app.flags.DEFINE_boolean('resume', None,
                            """Whether to resume the training process.""")

#train_file_list = '/data1/mouse_4313/retrain_datalist.csv'
#with open(train_file_list, 'r') as f:
#    reader = csv.reader(f)
#    for row in reader:
#        #retrain_mouse = row
#        trainlist = np.array(row).astype(np.int32)[:16]
#with open('train_subsets.pkl', 'rb') as f:
#    pickle.dump(f, new_train_subset)

#valid_file_list = '/data1/mouse_4313/retrainEXP_valid_datalist.csv'
#with open(valid_file_list, 'r') as f:
#    reader = csv.reader(f)
#    for row in reader:
#        #retrain_mouse = row
#        validlist = np.array(row).astype(np.int32)[:16]

def pretrain(n_epochs):
    trainer = DeepFeatureNetTrainer(
        data_dir=FLAGS.data_dir_mat,
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=FLAGS.fold_idx,
        batch_size=100,
        input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
        n_classes=NUM_CLASSES,
        interval_plot_filter=5,
        interval_save_model=1, # 5,
        interval_print_cm=1
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs,
        resume=FLAGS.resume
    )
    return pretrained_model_path


class DeepFeatureNetRetrainer(DeepFeatureNetTrainer):
    def __init__(self, data_dir_mat, data_dir_pkl, output_dir, batch_size=100,
                 input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
                 n_classes=NUM_CLASSES, # seq_length=25,
                 #n_rnn_layer=2, return_last=False,
                 interval_plot_filter=20, interval_save_model=1,
                 interval_print_cm=1, fold_idx=FLAGS.fold_idx,
                 model_dir=FLAGS.model_params,
                 hist_save_dir='/home/ota/mc-sleepnet_incrementallearning/hist/',
                 fname_loss_acc2="lossAcc_log_train_rewindtrain16mouse_session%d.csv" % (
                     training_step_number)  # 8.csv"
                 ):
        self.data_dir_mat=data_dir_mat
        self.data_dir_pkl=data_dir_pkl
        self.output_dir=output_dir
        self.batch_size=batch_size
        self.input_dims=input_dims
        self.n_classes=n_classes
        #self.seq_length=seq_length
        #self.n_rnn_layer=n_rnn_layer
        #self.return_last=return_last
        self.interval_plot_filter=interval_plot_filter
        self.interval_save_model=interval_save_model
        self.interval_print_cm=interval_print_cm
        self.fold_idx=fold_idx
    def _run_epoch(self, sess, network, inputs, targets, train_op, is_train):
        start_time=time.time()
        y=[]
        y_true=[]
        total_loss, n_batches=0.0, 0
        is_shuffle=True if is_train else False
        for eeg_batch, emg_batch, y_batch in iterate_minibatches(inputs[0],
                                                                 inputs[1],
                                                                 targets,
                                                                 self.batch_size,
                                                                 shuffle=is_shuffle):
            #print("eeg_batch shape: ", str(eeg_batch.shape))
            #print("emg_batch shape: ", str(emg_batch.shape))
            #print("y_batch shape: ", str(y_batch.shape))
            y_batch=y_batch.reshape(-1,)
            feed_dict={
                network.input_eeg_var: eeg_batch,
                network.input_emg_var: emg_batch,
                network.target_var: y_batch
            }
            _, loss_value, y_pred=sess.run(
                [train_op, network.loss_op, network.pred_op],
                feed_dict=feed_dict
            )
            total_loss += loss_value
            n_batches += 1
            y.append(y_pred)
            y_true.append(y_batch)
            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"
        duration=time.time() - start_time
        #print("n_batches: ", str(n_batches))
        total_loss /= n_batches
        total_y_pred=np.hstack(y)
        total_y_true=np.hstack(y_true)
        return total_y_true, total_y_pred, total_loss, duration
    def retrain(self, finetuned_model_path, n_epochs, training_step_number, resume):
        finetuned_model_name="deepsleepnet"
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation network
            train_net=DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=True,
                reuse_params=False,
                use_dropout=True
            )
            valid_net=DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=True,
                use_dropout=False
            )
            # Initialize parameters
            train_net.init_ops()
            valid_net.init_ops()
            print("Network (layers={})".format(len(train_net.activations)))
            print("input eeg ({}): {}".format(
                train_net.input_eeg_var.name, train_net.input_eeg_var.get_shape()
            ))
            print("input emg ({}): {}".format(
                train_net.input_emg_var.name, train_net.input_emg_var.get_shape()
            ))
            print("targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            ))
            for name, act in train_net.activations:
                print("{} ({}): {}".format(name, act.name, act.get_shape()))
            print(" ")
            # Make subdirectory for training MC-SleepNet
            output_dir = os.path.join(self.output_dir, "session%d" % (
                training_step_number), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Get list of all pretrained parameters
            with np.load(finetuned_model_path) as f:
                finetuned_params=f.keys()
            # Remove the network-name-prefix
            for i in range(len(finetuned_params)):
                finetuned_params[i] = finetuned_params[i].replace(
                    finetuned_model_name, "network")
            # Get trainable variables of the finetuned, and pretrain model
            #train_vars1=[v for v in tf.trainable_variables() if v.name.replace(
            #    train_net.name, "network") in finetuned_params and "conv" in v.name]
            #train_vars1=list(set(tf.trainable_variables())-set(train_vars2))
            # Make subdirectory for pretraining
            #output_dir=os.path.join(
            #    self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            output_dir=os.path.join(
                self.output_dir, "session{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step=tf.Variable(
                    0, name="global_step", trainable=False)
            # print "Pretrained parameters:"
            # for v in train_vars1:
            #     print v.name
            # print " "
            # print "Optimizing parameters:"
            # for v in train_vars2:
            #     print v.name
            # print " "
            # print "Trainable Variables:"
            # for v in tf.trainable_variables():
            #     print v.name, v.get_shape()
            # print " "
            # print "All Variables:"
            # for v in tf.global_variables():
            #     print v.name, v.get_shape()
            # print " "
            # Create a saver
            saver=tf.train.Saver(tf.all_variables(), max_to_keep=0)
            # Initialize variables in the graph
            # Define optimization operations
            train_op, grads_and_vars_op = adam(
                loss=train_net.loss_op,
                lr=1e-4,
                train_vars=tf.trainable_variables()
            )
            # Define optimization operations
            #train_op, grads_and_vars_op = adam_clipping(
            #    loss=train_net.loss_op,
            #    lr=1e-4,
            #    train_vars=tf.trainable_variables()
            #)
            sess.run(tf.initialize_all_variables())
            # Add the graph structure into the Tensorboard writer
            train_summary_wrt=tf.train.SummaryWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )
            # Resume the training if applicable
            load_pretrain=False
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(
                            sess, tf.train.latest_checkpoint(output_dir))
                        print "Model restored"
                        print "[{}] Resume pretraining ...\n".format(datetime.now())
                    else:
                        load_pretrain=True
            else:
                load_pretrain=True
            if load_pretrain:
                # Load pre-trained model
                print "Loading fine-tuned parameters in the previous session to the model ..."
                print " | --> {} from {}".format(finetuned_model_name, finetuned_model_path)
                with np.load(finetuned_model_path) as f:
                    for k, v in f.iteritems():
                        if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k or "LSTM" in k or "fc" in k:
                            continue
                        prev_k=k
                        k=k.replace(finetuned_model_name, train_net.name)
                        tmp_tensor=tf.get_default_graph().get_tensor_by_name(k)
                        sess.run(
                            tf.assign(
                                tmp_tensor,
                                v
                            )
                        )
                        print "assigned {}: {} to {}: {}".format(
                            prev_k, v.shape, k, tmp_tensor.get_shape()
                        )
                print " "
                print "[{}] Start pre-training ...\n".format(datetime.now())
            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader=SeqDataLoader(
                    data_dir_mat=self.data_dir_mat,
                    data_dir_pkl=self.data_dir_pkl,
                    upperLimit=8625
                )
                # retrain mouse list
                train_mouse=trainlist[training_step_number-1]
                print('fold_idx : '+str(self.fold_idx))
                print('training mouse list')
                print(train_mouse)
                valid_mouse=validlist
                print('validation mouse list')
                print(valid_mouse)
                # Performance history
                all_train_loss=np.zeros(n_epochs)
                all_train_acc=np.zeros(n_epochs)
                all_train_f1=np.zeros(n_epochs)
                all_valid_loss=np.zeros(n_epochs)
                all_valid_acc=np.zeros(n_epochs)
                all_valid_f1=np.zeros(n_epochs)
                train_losses=[]
                valid_losses=[]
            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # Update parameters and compute loss of training set
                y_true_train=np.array([])
                y_pred_train=np.array([])
                train_loss=0
                train_duration=0
                count=0
                #train_mouse = np.reshape(train_mouse,(-1,40)) # 20190323 LO removed.
                for i, mouseid in enumerate(list(train_mouse)):  # ['name']):
                    print "train:" + str(epoch) + "-" + str(count)
                    print('mouse id : '+str(mouseid))
                    csv_file_id=mouseid  # [i] # 20190323 LO added.
                    print(str(i)+'th training data loading...(DeepFeatureNet)')
                    eeg_train, emg_train, y_train=data_loader.load_cv_data([csv_file_id], is_train=True)  # i,is_train=True)
                    #eeg_train, emg_train, y_train = data_loader.load_cv_data(csv_file_id, is_train=True)#i,is_train=True)
                    #print("eeg: ", str(np.array(eeg_train[0]).shape))
                    #print("emg: ", str(np.array(emg_train[0]).shape))
                    #print("y_train shape: ", str(np.array(y_train[0]).shape))
                    true_train, pred_train, loss, duration=self._run_epoch(
                            sess=sess, network=train_net,
                            inputs=(eeg_train[0], emg_train[0]), 
                            targets=y_train[0],
                            train_op=train_op,
                            is_train=True
                    )
                    y_true_train=np.hstack((y_true_train, true_train))
                    y_pred_train=np.hstack((y_pred_train, pred_train))
                    print "loss:" + str(loss)
                    train_losses.append(loss)
                    train_loss=train_loss + loss
                    train_duration=train_duration + duration
                    count += 1
                train_loss=train_loss/count
                n_train_examples=len(y_true_train)
                train_cm=confusion_matrix(y_true_train, y_pred_train)
                train_acc=np.mean(y_true_train == y_pred_train)
                train_f1=f1_score(y_true_train, y_pred_train, average="macro")
                y_true_val=np.array([])
                y_pred_val=np.array([])
                valid_loss=0
                valid_duration=0
                count=0
                # Evaluate the model on the validation set
                for i, mouseid in enumerate(list(valid_mouse)):  # ['name']):
                    print "valid:" + str(count)
                    csv_file_id=mouseid  # [i] # 20190323 LO added.
                    print('loaded csv file name : '+str(csv_file_id))
                    print(str(i)+'th validation data loading...(DeepFeatureNet)')
                    eeg_valid, emg_valid, y_valid=data_loader.load_cv_data(
                        [csv_file_id], is_train=False)
                    true_val, pred_val, loss, duration=self._run_epoch(
                            sess=sess, network=valid_net,
                            inputs=(eeg_valid[0], emg_valid[0]), targets=y_valid[0],
                            train_op=tf.no_op(),
                            is_train=False
                    )
                    y_true_val=np.hstack((y_true_val, true_val))
                    y_pred_val=np.hstack((y_pred_val, pred_val))
                    valid_losses.append(loss)
                    valid_loss += loss
                    valid_duration += duration
                    count += 1
                valid_loss=valid_loss/count
                n_valid_examples=len(y_true_val)
                valid_cm=confusion_matrix(y_true_val, y_pred_val)
                valid_acc=np.mean(y_true_val == y_pred_val)
                valid_f1=f1_score(y_true_val, y_pred_val, average="macro")
                all_train_loss[epoch]=train_loss
                all_train_acc[epoch]=train_acc
                all_train_f1[epoch]=train_f1
                all_valid_loss[epoch]=valid_loss
                all_valid_acc[epoch]=valid_acc
                all_valid_f1[epoch]=valid_f1
                # Report performance
                self.print_performance(
                    sess, output_dir, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss, train_acc, train_f1,
                    valid_duration, valid_loss, valid_acc, valid_f1
                )
                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_epoch{}.npz".format(epoch)),
                    train_loss=all_train_loss, valid_loss=all_valid_loss,
                    train_acc=all_train_acc, valid_acc=all_valid_acc,
                    train_f1=all_train_f1, valid_f1=all_valid_f1,
                    y_true_val=np.asarray(y_true_val),
                    y_pred_val=np.asarray(y_pred_val)
                )
                #Save loss file
                output_train_losses=np.array(train_losses)
                #np.savetxt(self.output_dir+"train_loss_pretrain_%s.csv" %
                np.savetxt(output_dir+"train_loss_pretrain_%s.csv" %
                           training_step_number, output_train_losses, delimiter=",")
                output_valid_losses=np.array(valid_losses)
                #np.savetxt(self.output_dir+"valid_loss_pretrain_%s.csv" %
                np.savetxt(output_dir+"valid_loss_pretrain_%s.csv" %
                           training_step_number, output_valid_losses, delimiter=",")
                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(
                        sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir, 16)
                    self.plot_filters(
                        sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)", output_dir, 16)
                # epoch++, Save checkpoint
                sess.run(tf.assign(global_step, epoch+1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time=time.time()
                    save_path=os.path.join(
                        output_dir, "model_epoch{}.ckpt".format(epoch)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration=time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)
                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time=time.time()
                    save_dict={}
                    for v in tf.all_variables():
                        save_dict[v.name]=sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_epoch{}.npz".format(epoch)),
                        **save_dict
                    )
                    duration=time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)
        print "Finish pre-training"
        return os.path.join(output_dir, "params_epoch{}.npz".format(epoch))

def pretrain_restart(model_path, training_step_number, n_epochs):
    trainer = DeepFeatureNetRetrainer(
        data_dir_mat=FLAGS.data_dir_mat,
        data_dir_pkl=FLAGS.data_dir_pkl,
        output_dir=FLAGS.output_dir,
        batch_size=100,
        input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
        n_classes=NUM_CLASSES, #seq_length=25,
        #n_rnn_layer=2, return_last=False,
        interval_plot_filter=10, interval_save_model=1,
        interval_print_cm=1, fold_idx=FLAGS.fold_idx,
        model_dir=FLAGS.model_params,
        hist_save_dir='/home/ota/mc-sleepnet_incrementallearning/hist/',
        fname_loss_acc2="lossAcc_log_train_rewindtrain16mouse_session%d.csv" % (
            training_step_number)
    )
    pretrained_model_path = trainer.retrain(
        finetuned_model_path=model_path,
        n_epochs=n_epochs,
        training_step_number=training_step_number,
        resume=FLAGS.resume
    )
    return pretrained_model_path


def finetune(model_path, n_epochs):
    trainer = DeepSleepNetTrainer(
        data_dir=FLAGS.data_dir_mat,
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=FLAGS.fold_idx,
        batch_size=10,
        input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=10,
        interval_save_model=1,
        interval_print_cm=1
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path,
        n_epochs=n_epochs,
        resume=FLAGS.resume
    )
    return finetuned_model_path


class DeepSleepNetRetrainer(DeepSleepNetTrainer):
    def __init__(self, data_dir_mat, data_dir_pkl, output_dir, batch_size=100,
                 input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
                 n_classes=NUM_CLASSES,  seq_length=25,
                 n_rnn_layer=2, return_last=False,
                 interval_plot_filter=20, interval_save_model=1,
                 interval_print_cm=1, fold_idx=FLAGS.fold_idx,
                 model_dir=FLAGS.model_params,
                 hist_save_dir='/home/ota/mc-sleepnet_incrementallearning/hist/',
                 fname_loss_acc2="lossAcc_log_train_rewindtrain16mouse_session%d.csv" % (
                     training_step_number)  # 8.csv"
                 ):
        self.data_dir_mat=data_dir_mat
        self.data_dir_pkl=data_dir_pkl
        self.output_dir=output_dir
        self.batch_size=batch_size
        self.input_dims=input_dims
        self.n_classes=n_classes
        self.seq_length=seq_length
        self.n_rnn_layer=n_rnn_layer
        self.return_last=return_last
        self.interval_plot_filter=interval_plot_filter
        self.interval_save_model=interval_save_model
        self.interval_print_cm=interval_print_cm
        self.fold_idx=fold_idx
    def _run_epoch(self, sess, network, inputs, targets, train_op, is_train):
        start_time=time.time()
        y=[]
        y_true=[]
        total_loss, n_batches=0.0, 0
        for sub_idx, each_data in enumerate(itertools.izip(inputs[0], inputs[1], targets)):
            each_eeg, each_emg, each_y = each_data
            # Initialize state of LSTM - Bidirectional LSTM
            fw_state = sess.run(network.fw_initial_state)
            bw_state = sess.run(network.bw_initial_state)
            for eeg_batch, emg_batch, y_batch in iterate_minibatches(inputs[0],
                                                                    inputs[1],
                                                                    targets,
                                                                    self.batch_size,
                                                                    shuffle=is_shuffle):
                #print("eeg_batch shape: ", str(eeg_batch.shape))
                #print("emg_batch shape: ", str(emg_batch.shape))
                #print("y_batch shape: ", str(y_batch.shape))
                y_batch=y_batch.reshape(-1,)
                feed_dict={
                    network.input_eeg_var: eeg_batch,
                    network.input_emg_var: emg_batch,
                    network.target_var: y_batch
                }
                for i, (c, h) in enumerate(network.fw_initial_state):
                    feed_dict[c] = fw_state[i].c
                    feed_dict[h] = fw_state[i].h
                for i, (c, h) in enumerate(network.bw_initial_state):
                    feed_dict[c] = bw_state[i].c
                    feed_dict[h] = bw_state[i].h
                _, loss_value, y_pred =sess.run(
                    [train_op, network.loss_op, network.pred_op],
                    feed_dict=feed_dict
                )
                total_loss += loss_value
                n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)
                # Check the loss value
                assert not np.isnan(loss_value), \
                    "Model diverged with loss = NaN"
        duration=time.time() - start_time
        #print("n_batches: ", str(n_batches))
        total_loss /= n_batches
        total_y_pred=np.hstack(y)
        total_y_true=np.hstack(y_true)
        return total_y_true, total_y_pred, total_loss, duration
    def finetune(self, pretrained_model_path, finetuned_model_path, n_epochs, training_step_number, resume):
        pretrained_model_name = "deepfeaturenet"
        finetuned_model_name = "deepsleepnet"
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation network
            train_net = DeepSleepNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                n_rnn_layers=self.n_rnn_layers,
                return_last=self.return_last,
                is_train=True,
                reuse_params=False,
                use_dropout_feature=True,
                use_dropout_sequence=True
            )
            valid_net = DeepSleepNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                n_rnn_layers=self.n_rnn_layers,
                return_last=self.return_last,
                is_train=False,
                reuse_params=True,
                use_dropout_feature=True,
                use_dropout_sequence=True
            )
            # Initialize parameters
            train_net.init_ops()
            valid_net.init_ops()
            print("Network (layers={})".format(len(train_net.activations)))
            print("input eeg ({}): {}".format(
                train_net.input_eeg_var.name, train_net.input_eeg_var.get_shape()
            ))
            print("input emg ({}): {}".format(
                train_net.input_emg_var.name, train_net.input_emg_var.get_shape()
            ))
            print("targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            ))
            for name, act in train_net.activations:
                print("{} ({}): {}".format(name, act.name, act.get_shape()))
            print(" ")
            # Make subdirectory for training MC-SleepNet
            output_dir = os.path.join(self.output_dir, "session%d" % (
                training_step_number), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Get list of all pretrained parameters
            with np.load(finetuned_model_path) as f:
                finetuned_params=f.keys()
            # Get list of all pretrained parameters
            with np.load(pretrained_model_path) as f:
                pretrain_params = f.keys()
            # Remove the network-name-prefix
            for i in range(len(finetuned_params)):
                finetuned_params[i] = finetuned_params[i].replace(
                    finetuned_model_name, "network")
            for i in range(len(pretrain_params)):
                pretrained_params[i] = finetuned_params[i].replace(
                    pretrained_model_name, "network")
            # Get trainable variables of the finetuned, and pretrain model
            #train_vars1=[v for v in tf.trainable_variables() if v.name.replace(
            #    train_net.name, "network") in finetuned_params and "conv" in v.name]
            #train_vars1=list(set(tf.trainable_variables())-set(train_vars2))
            # Make subdirectory for pretraining
            #output_dir=os.path.join(
            #    self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            output_dir=os.path.join(
                self.output_dir, "session{}".format(training_step_number), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step=tf.Variable(
                    0, name="global_step", trainable=False)
            # print "Pretrained parameters:"
            # for v in train_vars1:
            #     print v.name
            # print " "
            # print "Optimizing parameters:"
            # for v in train_vars2:
            #     print v.name
            # print " "
            # print "Trainable Variables:"
            # for v in tf.trainable_variables():
            #     print v.name, v.get_shape()
            # print " "
            # print "All Variables:"
            # for v in tf.global_variables():
            #     print v.name, v.get_shape()
            # print " "
            # Create a saver
            saver=tf.train.Saver(tf.all_variables(), max_to_keep=0)
            # Initialize variables in the graph
            # Define optimization operations
            #train_op, grads_and_vars_op = adam(
            #    loss=train_net.loss_op,
            #    lr=1e-4,
            #    train_vars=tf.trainable_variables()
            #)
            # Define optimization operations
            #train_op, grads_and_vars_op = adam_clipping(
            #    loss=train_net.loss_op,
            #    lr=1e-4,
            #    train_vars=tf.trainable_variables()
            #)
            # Get trainable variables of the pretrained, and new ones
            train_vars1 = [v for v in tf.trainable_variables()
                           if v.name.replace(train_net.name, "network") in pretrain_params]
            train_vars2 = list(set(tf.trainable_variables()) - set(train_vars1))

            # Optimizer that use different learning rates for each part of the network
            train_op, grads_and_vars_op = adam_clipping_list_lr(
                loss=train_net.loss_op,
                list_lrs=[1e-6, 1e-4],
                list_train_vars=[train_vars1, train_vars2],
                clip_value=10.0
            )
            sess.run(tf.initialize_all_variables())
            # Add the graph structure into the Tensorboard writer
            train_summary_wrt=tf.train.SummaryWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )
            # Resume the training if applicable
            load_pretrain=False
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(
                            sess, tf.train.latest_checkpoint(output_dir))
                        print "Model restored"
                        print "[{}] Resume pretraining ...\n".format(datetime.now())
                    else:
                        load_pretrain=True
            else:
                load_pretrain=True
            if load_pretrain:
                # Load pre-trained model
                print "Loading fine-tuned parameters in the previous session to the model ..."
                print " | --> {} from {}".format(finetuned_model_name, finetuned_model_path)
                with np.load(finetuned_model_path) as f:
                    for k, v in f.iteritems():
                        if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                            continue
                        prev_k=k
                        k=k.replace(finetuned_model_name, train_net.name)
                        tmp_tensor=tf.get_default_graph().get_tensor_by_name(k)
                        sess.run(
                            tf.assign(
                                tmp_tensor,
                                v
                            )
                        )
                        print "assigned {}: {} to {}: {}".format(
                            prev_k, v.shape, k, tmp_tensor.get_shape()
                        )
                print " "
                print "Loading pre-trained parameters in the previous session to the model ..."
                print " | --> {} from {}".format(pretrained_model_name, pretrained_model_path)
                with np.load(pretrained_model_path) as f:
                    for k, v in f.iteritems():
                        if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                            continue
                        prev_k=k
                        k=k.replace(pretrained_model_name, train_net.name)
                        tmp_tensor=tf.get_default_graph().get_tensor_by_name(k)
                        sess.run(
                            tf.assign(
                                tmp_tensor,
                                v
                            )
                        )
                        print "assigned {}: {} to {}: {}".format(
                            prev_k, v.shape, k, tmp_tensor.get_shape()
                        )
                print " "
                print "[{}] Start fine-tuning ...\n".format(datetime.now())
            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader=SeqDataLoader(
                    data_dir_mat=self.data_dir_mat,
                    data_dir_pkl=self.data_dir_pkl,
                    upperLimit=8625
                )
                # retrain mouse list
                train_mouse=trainlist[training_step_number-1]
                print('fold_idx : '+str(self.fold_idx))
                print('training mouse list')
                print(train_mouse)
                valid_mouse=validlist
                print('validation mouse list')
                print(valid_mouse)
                # Performance history
                all_train_loss=np.zeros(n_epochs)
                all_train_acc=np.zeros(n_epochs)
                all_train_f1=np.zeros(n_epochs)
                all_valid_loss=np.zeros(n_epochs)
                all_valid_acc=np.zeros(n_epochs)
                all_valid_f1=np.zeros(n_epochs)
                train_losses=[]
                valid_losses=[]
            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # Update parameters and compute loss of training set
                y_true_train=np.array([])
                y_pred_train=np.array([])
                train_loss=0
                train_duration=0
                count=0
                #train_mouse = np.reshape(train_mouse,(-1,40)) # 20190323 LO removed.
                for i, mouseid in enumerate(list(train_mouse)):  # ['name']):
                    print "train:" + str(epoch) + "-" + str(count)
                    print('mouse id : '+str(mouseid))
                    csv_file_id=mouseid  # [i] # 20190323 LO added.
                    print(str(i)+'th training data loading...(DeepSleepNet)')
                    eeg_train, emg_train, y_train=data_loader.load_cv_data([csv_file_id], is_train=True)  # i,is_train=True)
                    #eeg_train, emg_train, y_train = data_loader.load_cv_data(csv_file_id, is_train=True)#i,is_train=True)
                    #print("eeg: ", str(np.array(eeg_train[0]).shape))
                    #print("emg: ", str(np.array(emg_train[0]).shape))
                    #print("y_train shape: ", str(np.array(y_train[0]).shape))
                    true_train, pred_train, loss, duration=self._run_epoch(
                            sess=sess, network=train_net,
                            inputs=(eeg_train[0], emg_train[0]), 
                            targets=y_train[0],
                            train_op=train_op,
                            is_train=True
                    )
                    y_true_train=np.hstack((y_true_train, true_train))
                    y_pred_train=np.hstack((y_pred_train, pred_train))
                    print "loss:" + str(loss)
                    train_losses.append(loss)
                    train_loss=train_loss + loss
                    train_duration=train_duration + duration
                    count += 1
                train_loss=train_loss/count
                n_train_examples=len(y_true_train)
                train_cm=confusion_matrix(y_true_train, y_pred_train)
                train_acc=np.mean(y_true_train == y_pred_train)
                train_f1=f1_score(y_true_train, y_pred_train, average="macro")
                y_true_val=np.array([])
                y_pred_val=np.array([])
                valid_loss=0
                valid_duration=0
                count=0
                # Evaluate the model on the validation set
                for i, mouseid in enumerate(list(valid_mouse)):  # ['name']):
                    print "valid:" + str(count)
                    csv_file_id=mouseid  # [i] # 20190323 LO added.
                    print('loaded csv file name : '+str(csv_file_id))
                    print(str(i)+'th validation data loading...(DeepSleepNet)')
                    eeg_valid, emg_valid, y_valid=data_loader.load_cv_data(
                        [csv_file_id], is_train=False)
                    true_val, pred_val, loss, duration=self._run_epoch(
                            sess=sess, network=valid_net,
                            inputs=(eeg_valid[0], emg_valid[0]), targets=y_valid[0],
                            train_op=tf.no_op(),
                            is_train=False
                    )
                    y_true_val=np.hstack((y_true_val, true_val))
                    y_pred_val=np.hstack((y_pred_val, pred_val))
                    valid_losses.append(loss)
                    valid_loss += loss
                    valid_duration += duration
                    count += 1
                valid_loss=valid_loss/count
                n_valid_examples=len(y_true_val)
                valid_cm=confusion_matrix(y_true_val, y_pred_val)
                valid_acc=np.mean(y_true_val == y_pred_val)
                valid_f1=f1_score(y_true_val, y_pred_val, average="macro")
                all_train_loss[epoch]=train_loss
                all_train_acc[epoch]=train_acc
                all_train_f1[epoch]=train_f1
                all_valid_loss[epoch]=valid_loss
                all_valid_acc[epoch]=valid_acc
                all_valid_f1[epoch]=valid_f1
                # Report performance
                self.print_performance(
                    sess, output_dir, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss, train_acc, train_f1,
                    valid_duration, valid_loss, valid_acc, valid_f1
                )
                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_epoch{}.npz".format(epoch)),
                    train_loss=all_train_loss, valid_loss=all_valid_loss,
                    train_acc=all_train_acc, valid_acc=all_valid_acc,
                    train_f1=all_train_f1, valid_f1=all_valid_f1,
                    y_true_val=np.asarray(y_true_val),
                    y_pred_val=np.asarray(y_pred_val)
                )
                #Save loss file
                output_train_losses=np.array(train_losses)
                #np.savetxt(self.output_dir+"train_loss_pretrain_%s.csv" %
                np.savetxt(output_dir+"train_loss_pretrain_%s.csv" %
                           training_step_number, output_train_losses, delimiter=",")
                output_valid_losses=np.array(valid_losses)
                #np.savetxt(self.output_dir+"valid_loss_pretrain_%s.csv" %
                np.savetxt(output_dir+"valid_loss_pretrain_%s.csv" %
                           training_step_number, output_valid_losses, delimiter=",")
                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(
                        sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir, 16)
                    self.plot_filters(
                        sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)", output_dir, 16)
                # epoch++, Save checkpoint
                sess.run(tf.assign(global_step, epoch+1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time=time.time()
                    save_path=os.path.join(
                        output_dir, "model_epoch{}.ckpt".format(epoch)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration=time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)
                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time=time.time()
                    save_dict={}
                    for v in tf.all_variables():
                        save_dict[v.name]=sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_epoch{}.npz".format(epoch)),
                        **save_dict
                    )
                    duration=time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)
        print "Finish fine-tuning"
        return os.path.join(output_dir, "params_epoch{}.npz".format(epoch))


def finetune_restart(model_path, model_path2, n_epochs, training_step_number):
    trainer = DeepSleepNetRetrainer(
        data_dir=FLAGS.data_dir_mat,
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=FLAGS.fold_idx,
        batch_size=10,
        input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=10,
        interval_save_model=1,
        interval_print_cm=1
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path,
        finetuned_model_path=model_path2,
        n_epochs=n_epochs,
        training_step_number=training_step_number,
        resume=FLAGS.resume
    )
    return finetuned_model_path


def main(argv=None):
    # Output dir
    #training_step_number = int(train_file_list[-1])
    output_dir = FLAGS.output_dir # os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))
    #if not FLAGS.resume:
    #    if tf.gfile.Exists(output_dir):
    #        tf.gfile.DeleteRecursively(output_dir)
    #    tf.gfile.MakeDirs(output_dir)
    #for training_step_number in range(20):
    if training_step_number==1:
        pretrained_model_path = pretrain(
            n_epochs=FLAGS.pretrain_epochs
        )
        print(pretrained_model_path)
        finetuned_model_path = finetune(
            model_path=pretrained_model_path,
            n_epochs=FLAGS.finetune_epochs
        )
        print(finetuned_model_path)
        #training_step_number = int(train_file_list[-1])
        #retraining_model_path = retraining(trained_model_file=os.path.split(pretrained_model_path)[0]+"/model_epoch%d.ckpt-%d"%((training_step_number*10)-1, training_step_number*10), trained_model_params=pretrained_model_path, n_epochs=FLAGS.finetune_epochs, training_step_number=training_step_number, resume=FLAGS.resume)
        #print(retraining_model_path)
    if training_step_number>1:
        pretrained_model_path = pretrain_restart(
            model_path=FLAGS.model_params, # finetuned_model_path,
            n_epochs=FLAGS.pretrain_epochs,
            training_step_number=training_step_number
        )
        print(pretrained_model_path)
        finetuned_model_path = finetune_restart(
            model_path=pretrained_model_path,
            model_path2=FLAGS.model_params, # finetuned_model_path,
            n_epochs=FLAGS.finetune_epochs,
            training_step_number=training_step_number
        )
        print(finetuned_model_path)
        #retraining_model_path = retraining(trained_model_file=FLAGS.model_file, trained_model_params=FLAGS.model_params, n_epochs=FLAGS.finetune_epochs, training_step_number=training_step_number, resume=FLAGS.resume)
        #print(retraining_model_path)

if __name__ == "__main__":
    tf.app.run()
