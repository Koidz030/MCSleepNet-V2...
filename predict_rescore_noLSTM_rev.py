#! /usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import os
import time
import csv

import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime

from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from deepsleep.data_loader import NonSeqDataLoader
from deepsleep.model import DeepFeatureNet
from deepsleep.nn import *
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)
from deepsleep.utils import iterate_minibatches


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir_mat', '/data2/data/',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('data_dir_pkl', '/data1/mouse_4313/',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('model_dir', '/home/ota/cross_1/fold1/deepfeaturenet/model_epoch9.ckpt-10',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', '/data2/mcsleepnet_incrementalLearning/output_20210201featurenet',
                           """Directory where to save outputs.""")
tf.app.flags.DEFINE_string('file_list', '/data1/mouse_4313/retrainEXP_valid_datalist.csv',
                            """csv file of test mice list""")


def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
    reg_loss_value = sess.run(reg_loss)

    # Print performance
    print (
        "duration={:.3f} sec, n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
        "f1={:.3f}".format(
            duration, n_examples, loss, reg_loss_value, acc, f1
        )
    )
    print cm
    print " "


def custom_run_epoch(
    sess, 
    network, 
    inputs, 
    targets, 
    train_op, 
    is_train, 
    output_dir, 
    subject_idx
):
    start_time = time.time()
    y = []
    y_true = []
    W_prob = []
    NR_prob = []
    R_prob = []
    total_loss, n_batches = 0.0, 0

    # Store prediction and actual stages of each patient
    for eeg, emg, y_batch in iterate_minibatches(inputs[0],
                                                 inputs[1],
                                                 targets,
                                                 network.batch_size,
                                                 shuffle=False):
        feed_dict = {
            network.input_eeg_var: eeg,
            network.input_emg_var: emg,
            network.target_var: y_batch
        }

        _, loss_value, y_pred, logits = sess.run(
            [train_op, network.loss_op, network.pred_op, network.prob],
            feed_dict=feed_dict
        )

        total_loss += loss_value
        n_batches += 1

        # Check the loss value
        #assert not np.isnan(loss_value), \
            #    "Model diverged with loss = NaN"

        y.append(y_pred)
        y_true.append(y_batch)
        #W_prob.append(logits[:,0])
        #NR_prob.append(logits[:,1])
        #R_prob.append(logits[:,2])

    # Save memory cells and predictions
    save_dict = {
        "y_true": y_true,
        "y_pred": y,
    }
    print subject_idx
    save_path = os.path.join(
        output_dir,
        "output_subject_{}.npz".format(subject_idx)
    )
    np.savez(save_path, **save_dict)
    print "Saved outputs to {}".format(save_path)

    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)

    return total_y_true, total_y_pred, total_loss, duration


def predict(
    data_dir_mat,
    data_dir_pkl,
    model_dir, 
    output_dir,
    filelist
):
    # Ground truth and predictions
    y_true = []
    y_pred = []

    # The model will be built into the default Graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build the network
        valid_net = DeepFeatureNet(
            batch_size=1, 
            input_dims=EPOCH_SEC_LEN*SAMPLING_RATE,
            n_classes=NUM_CLASSES, 
            is_train=False, 
            reuse_params=False, 
            use_dropout=True
        )

        # Initialize parameters
        valid_net.init_ops()
        #mouse = np.array(pd.read_csv(filelist,header=None,index_col=None,dtype=int)).reshape(-1)
        with open(filelist, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                mouse=row
            mouse = np.array(mouse).astype(np.int32)
        print mouse
        data_loader=NonSeqDataLoader(data_dir_mat=data_dir_mat, data_dir_pkl=data_dir_pkl)
        for subject_idx, subject_name in enumerate(mouse):
            fold_idx = 1
            checkpoint_path = os.path.join(
                model_dir
            )

            # Restore the trained model
            #vars_all = tf.all_variables()
            #print(vars_all)
            #vars_for_featurenet = tf.get_collection(tf.GraphKeys.VARIABLES, scope='deepfeaturenet')
            #print(vars_for_featurenet)
            #vars_to_pred = list(set(vars_all) - set(vars_for_featurenet))
            saver = tf.train.Saver()#vars_to_pred)
            #sess.run(tf.initialize_all_variables()) # 2021/2/1 LO added. 
            saver.restore(sess, checkpoint_path) #tf.train.latest_checkpoint(checkpoint_path))
            print "Model restored from: {}\n".format(checkpoint_path) #tf.train.latest_checkpoint(checkpoint_path))

            # Load testing data
            eeg,emg, y = data_loader.load_cv_data(
                files = [subject_name],
                is_train=False
            )

            # Loop each epoch
            print "[{}] Predicting ...\n".format(datetime.now())

            # Evaluate the model on the subject data
            y_true_, y_pred_, loss, duration = \
                custom_run_epoch(
                    sess=sess, network=valid_net,
                    inputs=(eeg,emg), targets=y,
                    train_op=tf.no_op(),
                    is_train=False,
                    output_dir=output_dir,
                    subject_idx=subject_name
                )
            n_examples = len(y_true_)
            cm_ = confusion_matrix(y_true_, y_pred_)
            acc_ = np.mean(y_true_ == y_pred_)
            mf1_ = f1_score(y_true_, y_pred_, average="macro")

            # Report performance
            print_performance(
                sess, valid_net.name,
                n_examples, duration, loss, 
                cm_, acc_, mf1_
            )

            y_true.extend(y_true_)
            y_pred.extend(y_pred_)
        
    # Overall performance
    print "[{}] Overall prediction performance\n".format(datetime.now())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_examples = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print (
        "n={}, acc={:.3f}, f1={:.3f}".format(
            n_examples, acc, mf1
        )
    )
    print cm


def main(argv=None):
    # # Makes the random numbers predictable
    # np.random.seed(0)
    # tf.set_random_seed(0)

    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    n_subjects = 4
    n_subjects_per_fold = 1
    predict(
        data_dir_mat=FLAGS.data_dir_mat,
        data_dir_pkl=FLAGS.data_dir_pkl,
        model_dir=FLAGS.model_dir,
        output_dir=FLAGS.output_dir,
        filelist=FLAGS.file_list
    )


if __name__ == "__main__":
    tf.app.run()
