import itertools
import os
import re
import time

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score

from deepsleep.data_loader import NonSeqDataLoader, SeqDataLoader
from deepsleep.model import DeepFeatureNet, DeepSleepNet
from deepsleep.optimize import adam, adam_clipping_list_lr, adam_clipping
from deepsleep.utils import iterate_minibatches, iterate_batch_seq_minibatches

#df_mouse_all = pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv")
clearDataFlag = 1# For analysis of clear data, please set this flag 1. For analysis of noisy data, please set it 0.
if clearDataFlag:
    df_mouse_all = pd.read_csv("/data2/clearData20190330/train_mouse_all.csv")
else:
    df_mouse_all = pd.read_csv("/data2/clearData20190330/train_mouse_all_noisy.csv")

class Trainer(object):

    def __init__(
        self,
        interval_plot_filter =1,
        interval_save_model=1,
        interval_print_cm=1
    ):
        self.interval_plot_filter = interval_plot_filter
        self.interval_save_model = interval_save_model
        self.interval_print_cm = interval_print_cm

    def print_performance(self, sess, output_dir, network_name,
                           n_train_examples, n_valid_examples,
                           train_cm, valid_cm, epoch, n_epochs,
                           train_duration, train_loss, train_acc, train_f1,
                           valid_duration, valid_loss, valid_acc, valid_f1):
        # Get regularization loss
        train_reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
        train_reg_loss_value = sess.run(train_reg_loss)
        valid_reg_loss_value = train_reg_loss_value

        # Print performance
        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print " "
            print "[{}] epoch {}:".format(
                datetime.now(), epoch+1
            )
            print (
                "train ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}".format(
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_acc, train_f1
                )
            )
            print train_cm
            print (
                "valid ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}".format(
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_acc, valid_f1
                )
            )
            print valid_cm
            print " "
        else:
            print (
                "epoch {}: "
                "train ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f} | "
                "valid ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f}".format(
                    epoch+1,
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_acc, train_f1,
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_acc, valid_f1
                )
            )

    def print_network(self, network):
        print "inputs ({}): {}".format(
            network.inputs.name, network.inputs.get_shape()
        )
        print "targets ({}): {}".format(
            network.targets.name, network.targets.get_shape()
        )
        for name, act in network.activations:
            print "{} ({}): {}".format(name, act.name, act.get_shape())
        print " "

    def plot_filters(self, sess, epoch, reg_exp, output_dir, n_viz_filters):
        conv_weight = re.compile(reg_exp)
        for v in tf.trainable_variables():
            value = sess.run(v)
            if conv_weight.match(v.name):
                weights = np.squeeze(value)
                # Only plot conv that has one channel
                if len(weights.shape) > 2:
                    continue
                weights = weights.T
                plt.figure(figsize=(18, 10))
                plt.title(v.name)
                for w_idx in xrange(n_viz_filters):
                    plt.subplot(4, 4, w_idx+1)
                    plt.plot(weights[w_idx])
                    plt.axis("tight")
                plt.savefig(os.path.join(
                    output_dir, "{}_{}.png".format(
                        v.name.replace("/", "_").replace(":0", ""),
                        epoch+1
                    )
                ))
                plt.close("all")


class DeepFeatureNetTrainer(Trainer):

    def __init__(
        self,
        data_dir,
        output_dir,
        n_folds,
        fold_idx,
        batch_size,
        input_dims,
        n_classes,
        interval_plot_filter=1,
        interval_save_model=1,
        interval_print_cm=1
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes

    def _run_epoch(self, sess, network, inputs, targets, train_op, is_train):
        start_time = time.time()
        y = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        is_shuffle = True if is_train else False
        for eeg_batch, emg_batch, y_batch in iterate_minibatches(inputs[0],
                                                    inputs[1],
                                                    targets,
                                                    self.batch_size,
                                                    shuffle=is_shuffle):
            feed_dict = {
                network.input_eeg_var: eeg_batch,
                network.input_emg_var: emg_batch,
                network.target_var: y_batch
            }

            # # MONITORING
            # if n_batches == 0:
            #     print "BEFORE UPDATE [is_train={}]".format(is_train)
            #     for n, v in network.monitor_vars[:2]:
            #         val = sess.run(v, feed_dict=feed_dict)
            #         val = np.transpose(val, axes=(3, 0, 1, 2)).reshape((64, -1))
            #         mean_val = np.mean(val, axis=1)
            #         var_val = np.var(val, axis=1)
            #         print "{}: {}\nmean_shape={}, mean_val={}\nvar_shape={}, var_val={}".format(
            #             n, val.shape, mean_val.shape, mean_val[:5], var_val.shape, var_val[:5]
            #         )

            _, loss_value, y_pred = sess.run(
                [train_op, network.loss_op, network.pred_op],
                feed_dict=feed_dict
            )

            # # MONITORING
            # if n_batches == 0:
            #     print "AFTER UPDATE [is_train={}]".format(is_train)
            #     for n, v in network.monitor_vars[:2]:
            #         val = sess.run(v, feed_dict=feed_dict)
            #         val = np.transpose(val, axes=(3, 0, 1, 2)).reshape((64, -1))
            #         mean_val = np.mean(val, axis=1)
            #         var_val = np.var(val, axis=1)
            #         print "{}: {}\nmean_shape={}, mean_val={}\nvar_shape={}, var_val={}".format(
            #             n, val.shape, mean_val.shape, mean_val[:5], var_val.shape, var_val[:5]
            #         )

            total_loss += loss_value
            n_batches += 1
            y.append(y_pred)
            y_true.append(y_batch)

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss, duration

    def train(self, n_epochs, resume):
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
            train_net = DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=True,
                reuse_params=False,
                use_dropout=True
            )
            valid_net = DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=True,
                use_dropout=True
            )

            # Initialize parameters
            train_net.init_ops()
            valid_net.init_ops()

            print "Network (layers={})".format(len(train_net.activations))
            print "inputs_eeg ({}): {}".format(
                train_net.input_eeg_var.name, train_net.input_eeg_var.get_shape()
            )
            print "inputs_emg ({}): {}".format(
                train_net.input_emg_var.name, train_net.input_emg_var.get_shape()
            )
            print "targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            )
            for name, act in train_net.activations:
                print "{} ({}): {}".format(name, act.name, act.get_shape())
            print " "

            # Define optimization operations
            train_op, grads_and_vars_op = adam(
                loss=train_net.loss_op,
                lr=1e-4,
                train_vars=tf.trainable_variables()
            )

            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

            # print "Trainable Variables:"
            # for v in tf.trainable_variables():
            #     print v.name, v.get_shape()
            # print " "

            # print "All Variables:"
            # for v in tf.global_variables():
            #     print v.name, v.get_shape()
            # print " "

            # Create a saver
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

            # Initialize variables in the graph
            sess.run(tf.initialize_all_variables())

            # Add the graph structure into the Tensorboard writer
            train_summary_wrt = tf.train.SummaryWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )

            # Resume the training if applicable
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                        print "Model restored"
                        print "[{}] Resume pre-training ...\n".format(datetime.now())
                    else:
                        print "[{}] Start pre-training ...\n".format(datetime.now())
            else:
                print "[{}] Start pre-training ...\n".format(datetime.now())

            print('fold_idx : '+str(self.fold_idx))
            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader = NonSeqDataLoader(
                    data_dir=self.data_dir,
                    #n_folds=self.n_folds,
                    #fold_idx=self.fold_idx
                ) # 20190322 LO added n_folds and fold_id
                #train_mouse = np.array(pd.read_csv("/data2/data/train_mouse_cross1.csv",header=None,index_col=None,dtype=int))
                #train_mouse = np.array(range(1,13))#501)) # 20190323 LO for test.
                #train_mouse = np.array(pd.read_csv("/data2/clearData20190330/train_mouse_cross1.csv",header=None,index_col=None,dtype=int))
                #train_mouse = pd.read_csv("/data2/clearData20190330/train_mouse_all.csv")
                train_mouse = df_mouse_all#pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv")
                print('all mouse id list')
                print(train_mouse)
                if clearDataFlag:
                    train_mouse = train_mouse.drop([(2*(self.n_folds-self.fold_idx)), (2*(self.n_folds-self.fold_idx)+1)])
                else:
                    train_mouse = train_mouse.drop([self.fold_idx])
                print('training mouse id list')
                print(train_mouse)
                #valid_mouse = np.array(pd.read_csv("/data2/data/valid_mouse_cross1.csv",header=None,index_col=None,dtype=int))[:10]
                #valid_mouse = np.arange(13,15) # 20190323 LO for test.
                #valid_mouse = np.array(pd.read_csv("/data2/clearData20190330/valid_mouse_cross1.csv",header=None,index_col=None,dtype=int))[:10]
                #valid_mouse = pd.read_csv("/data2/clearData20190330/train_mouse_all.csv")
                valid_mouse = df_mouse_all#pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv")
                if clearDataFlag:
                    valid_mouse = valid_mouse[(2*(self.n_folds-self.fold_idx)):(2*(self.n_folds-self.fold_idx)+1)]
                else:
                    valid_mouse = valid_mouse['name'][self.fold_idx]
                print('validation mouse id list')
                print(valid_mouse)
                # Performance history
                all_train_loss = np.zeros(n_epochs)
                all_train_acc = np.zeros(n_epochs)
                all_train_f1 = np.zeros(n_epochs)
                all_valid_loss = np.zeros(n_epochs)
                all_valid_acc = np.zeros(n_epochs)
                all_valid_f1 = np.zeros(n_epochs)
                train_losses = []
                valid_losses = []

            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # # MONITORING
                # print "BEFORE TRAINING"
                # monitor_vars = [
                #     "deepfeaturenet/l1_conv/bn/moving_mean:0",
                #     "deepfeaturenet/l1_conv/bn/moving_variance:0"
                # ]
                # for n in monitor_vars:
                #     v = tf.get_default_graph().get_tensor_by_name(n)
                #     val = sess.run(v)
                #     print "{}: {}, {}".format(n, val.shape, val[:5])


                # Update parameters and compute loss of training set
                y_true_train = np.array([])
                y_pred_train = np.array([])
                train_loss = 0
                train_duration = 0
                count  = 0
                #train_mouse = np.reshape(train_mouse,(-1,40))
                for i,mouseid in enumerate(train_mouse['name']):
                    print "train:" + str(epoch) + "-" +str(count)
                    csv_file_no = mouseid#[i]
                    print('mouse id : '+str(mouseid))
                    print(str(i)+'th training data loading...(DeepFeatureNet)')
                    eeg_train, emg_train, y_train = data_loader.load_cv_data(csv_file_no, is_train=True)#i,is_train=True)
                    count += 1
                    y_true, y_pred, loss, duration = \
                        self._run_epoch(
                            sess=sess, network=train_net,
                            inputs=(eeg_train,emg_train), targets=y_train,
                            train_op=train_op,
                            is_train=True
                        )
                    y_true_train = np.hstack((y_true_train,y_true))
                    y_pred_train = np.hstack((y_pred_train,y_pred))
                    print "loss:" + str(loss)
                    train_loss += loss
                    train_losses.append(loss)
                    train_duration += duration
                train_loss = train_loss/count

                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train)
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

                # # MONITORING
                # print "AFTER TRAINING and BEFORE VALID"
                # for n in monitor_vars:
                #     v = tf.get_default_graph().get_tensor_by_name(n)
                #     val = sess.run(v)
                #     print "{}: {}, {}".format(n, val.shape, val[:5])


                # Evaluate the model on the validation set
                y_true_val = np.array([])
                y_pred_val = np.array([])
                valid_loss = 0
                valid_duration = 0
                count = 0
                #valid_mouse = np.reshape(valid_mouse,(-1,10)) # 20190323 LO removed. 
                if clearDataFlag:
                    for i,mouseid in enumerate(valid_mouse['name']):
                        print "valid"
                        count += 1
                        csv_file_id = mouseid#[i] # 20190323 LO added. 
                        print(str(i)+'th validation data loading ...(DeepFeatureNet)')
                        eeg_valid, emg_valid, y_valid = data_loader.load_cv_data(csv_file_id, is_train=False)#i,is_train=False) # 20190323 LO revised.
                        #eeg_valid, emg_valid, y_valid = data_loader.load_subject_data5(data_dir=self.data_dir, mouse=mouse)
                        true_val, pred_val, loss, duration = \
                            self._run_epoch(
                                sess=sess, network=valid_net,
                                inputs=(eeg_valid,emg_valid), targets=y_valid,
                                train_op=tf.no_op(),
                                is_train=False
                            )
                        y_true_val = np.hstack((y_true_val,true_val))
                        y_pred_val = np.hstack((y_pred_val,pred_val))
                        valid_loss += loss
                        valid_losses.append(loss)
                        valid_duration += duration
                else:
                    for i,mouseid in enumerate([valid_mouse]):#['name']):
                        print "valid"
                        count += 1
                        csv_file_id = mouseid#[i] # 20190323 LO added. 
                        print(str(i)+'th validation data loading ...(DeepFeatureNet)')
                        eeg_valid, emg_valid, y_valid = data_loader.load_cv_data(csv_file_id, is_train=False)#i,is_train=False) # 20190323 LO revised.
                        #eeg_valid, emg_valid, y_valid = data_loader.load_subject_data5(data_dir=self.data_dir, mouse=mouse)
                        true_val, pred_val, loss, duration = \
                            self._run_epoch(
                                sess=sess, network=valid_net,
                                inputs=(eeg_valid,emg_valid), targets=y_valid,
                                train_op=tf.no_op(),
                                is_train=False
                            )
                        y_true_val = np.hstack((y_true_val,true_val))
                        y_pred_val = np.hstack((y_pred_val,pred_val))
                        valid_loss += loss
                        valid_losses.append(loss)
                        valid_duration += duration

                valid_loss = valid_loss/count
    

                n_valid_examples = len(y_true_val)
                valid_cm = confusion_matrix(y_true_val, y_pred_val)
                valid_acc = np.mean(y_true_val == y_pred_val)
                valid_f1 = f1_score(y_true_val, y_pred_val, average="macro")

                # db.train_log(args={
                #     "n_folds": self.n_folds,
                #     "fold_idx": self.fold_idx,
                #     "epoch": epoch,
                #     "train_step": "pretraining",
                #     "datetime": datetime.utcnow(),
                #     "model": train_net.name,
                #     "n_train_examples": n_train_examples,
                #     "n_valid_examples": n_valid_examples,
                #     "train_loss": train_loss,
                #     "train_acc": train_acc,
                #     "train_f1": train_f1,
                #     "train_duration": train_duration,
                #     "valid_loss": valid_loss,
                #     "valid_acc": valid_acc,
                #     "valid_f1": valid_f1,
                #     "valid_duration": valid_duration,
                # })

                all_train_loss[epoch] = train_loss
                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_valid_loss[epoch] = valid_loss
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1

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
                output_train_losses = np.array(train_losses)
                np.savetxt(self.output_dir+"train_loss_pretrain_%s.csv"%self.fold_idx, output_train_losses, delimiter=",")

                output_valid_losses = np.array(valid_losses)
                np.savetxt(self.output_dir+"valid_loss_pretrain_%s.csv"%self.fold_idx, output_valid_losses, delimiter=",")

                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir, 16)
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)", output_dir, 16)

                # epoch++, Save checkpoint
                sess.run(tf.assign(global_step, epoch+1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_epoch{}.ckpt".format(epoch)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)

                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_dict = {}
                    for v in tf.all_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_epoch{}.npz".format(epoch)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)

        print "Finish pre-training"
        return os.path.join(output_dir, "params_epoch{}.npz".format(epoch))


class DeepSleepNetTrainer(Trainer):

    def __init__(
        self,
        data_dir,
        output_dir,
        n_folds,
        fold_idx,
        batch_size,
        input_dims,
        n_classes,
        seq_length,
        n_rnn_layers,
        return_last,
        interval_plot_filter=1,
        interval_save_model=1,
        interval_print_cm=1
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last

    def _run_epoch(self, sess, network, inputs, targets, train_op, is_train):
        start_time = time.time()
        y = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        for sub_idx, each_data in enumerate(itertools.izip(inputs[0], inputs[1], targets)):
            each_eeg, each_emg, each_y = each_data

            # # Initialize state of LSTM - Unidirectional LSTM
            # state = sess.run(network.initial_state)

            # Initialize state of LSTM - Bidirectional LSTM
            fw_state = sess.run(network.fw_initial_state)
            bw_state = sess.run(network.bw_initial_state)

            for eeg_batch, emg_batch, y_batch in iterate_batch_seq_minibatches(eeg=each_eeg,
                                                                  emg=each_emg,
                                                                  targets=each_y,
                                                                  batch_size=self.batch_size,
                                                                  seq_length=self.seq_length):
                feed_dict = {
                    network.input_eeg_var: eeg_batch,
                    network.input_emg_var: emg_batch,
                    network.target_var: y_batch
                }

                # Unidirectional LSTM
                # for i, (c, h) in enumerate(network.initial_state):
                #     feed_dict[c] = state[i].c
                #     feed_dict[h] = state[i].h

                # _, loss_value, y_pred, state = sess.run(
                #     [train_op, network.loss_op, network.pred_op, network.final_state],
                #     feed_dict=feed_dict
                # )

                for i, (c, h) in enumerate(network.fw_initial_state):
                    feed_dict[c] = fw_state[i].c
                    feed_dict[h] = fw_state[i].h

                for i, (c, h) in enumerate(network.bw_initial_state):
                    feed_dict[c] = bw_state[i].c
                    feed_dict[h] = bw_state[i].h

                _, loss_value, y_pred, fw_state, bw_state = sess.run(
                    [train_op, network.loss_op, network.pred_op, network.fw_final_state, network.bw_final_state],
                    feed_dict=feed_dict
                )

                total_loss += loss_value
                n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)

                # Check the loss value
                assert not np.isnan(loss_value), \
                    "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)
        print "total_y_true"
        print total_y_true.shape

        return total_y_true, total_y_pred, total_loss, duration

    def finetune(self, pretrained_model_path, n_epochs, resume):
        pretrained_model_name = "deepfeaturenet"

        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
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

            print "Network (layers={})".format(len(train_net.activations))
            print "inputs_eeg ({}): {}".format(
                train_net.input_eeg_var.name, train_net.input_eeg_var.get_shape()
            )
            print "inputs_emg ({}): {}".format(
                train_net.input_emg_var.name, train_net.input_emg_var.get_shape()
            )
            print "targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            )
            for name, act in train_net.activations:
                print "{} ({}): {}".format(name, act.name, act.get_shape())
            print " "

            
            # Get list of all pretrained parameters
            with np.load(pretrained_model_path) as f:
                pretrain_params = f.keys()

            # Remove the network-name-prefix
            for i in range(len(pretrain_params)):
                pretrain_params[i] = pretrain_params[i].replace(pretrained_model_name, "network")

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
            
            """
            # Define optimization operations
            train_op, grads_and_vars_op = adam_clipping(
                loss=train_net.loss_op,
                lr=1e-4,
                train_vars=tf.trainable_variables()
            )
            """
            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

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
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

            # Initialize variables in the graph
            sess.run(tf.initialize_all_variables())

            # Add the graph structure into the Tensorboard writer
            train_summary_wrt = tf.train.SummaryWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )

            # Resume the training if applicable
            load_pretrain = False
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                        print "Model restored"
                        print "[{}] Resume fine-tuning ...\n".format(datetime.now())
                    else:
                        load_pretrain = True
            else:
                load_pretrain = True

            if load_pretrain:
                # Load pre-trained model
                print "Loading pre-trained parameters to the model ..."
                print " | --> {} from {}".format(pretrained_model_name, pretrained_model_path)
                with np.load(pretrained_model_path) as f:
                    for k, v in f.iteritems():
                        if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                            continue
                        prev_k = k
                        k = k.replace(pretrained_model_name, train_net.name)
                        tmp_tensor = tf.get_default_graph().get_tensor_by_name(k)
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
                data_loader = SeqDataLoader(
                    data_dir=self.data_dir,
                )
                #train_mouse = np.array(pd.read_csv("/data2/data/train_mouse_cross1.csv",header=None,index_col=None,dtype=int))
                #train_mouse = np.array(pd.read_csv("/data2/testData/train_mouse_cross1.csv",header=None,index_col=None,dtype=int)) # 20190323 LO removed. 
                #train_mouse = np.array(range(1,13))#501)) # 2030323 LO inserted. 
                #train_mouse = np.array(pd.read_csv("/data2/clearData20190330/train_mouse_cross1.csv",header=None,index_col=None,dtype=int)) # 20190323 LO removed. 
                #train_mouse = pd.read_csv("/data2/clearData20190330/train_mouse_cross_all.csv",header=1,index_col=None) # 20190323 LO removed. 
                #train_mouse = pd.read_csv("/data2/clearData20190330/train_mouse_all.csv") # 20190323 LO removed. 
                train_mouse = df_mouse_all#pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv") # 20190323 LO removed. 
                print('fold_idx : '+str(self.fold_idx))
                if clearDataFlag:
                    train_mouse = train_mouse.drop([(2*(self.n_folds-self.fold_idx)), ((2*(self.n_folds-self.fold_idx))+1)])
                else:
                    train_mouse = train_mouse.drop([self.fold_idx])
                print('training mouse list')
                print(train_mouse)
                #valid_mouse = np.array(pd.read_csv("/data2/testData/valid_mouse_cross1.csv",header=None,index_col=None,dtype=int))[:10] # 20190323 LO removed. 
                #valid_mouse = np.arange(13,15) # 20190323 LO inserted.
                #valid_mouse = np.array(pd.read_csv("/data2/clearData20190330/valid_mouse_cross1.csv",header=None,index_col=None,dtype=int))[:10] # 20190323 LO removed. 
                #valid_mouse = pd.read_csv("/data2/clearData20190330/valid_mouse_cross_all.csv",header=1,index_col=None) # 20190323 LO removed. 
                valid_mouse = df_mouse_all#pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv") # 20190323 LO removed. 
                if clearDataFlag:
                    #valid_mouse = pd.read_csv("/data2/clearData20190330/train_mouse_all.csv") # 20190323 LO removed. 
                    valid_mouse = valid_mouse[(2*(self.n_folds-self.fold_idx)):((2*(self.n_folds-self.fold_idx))+1)]
                else:
                    #valid_mouse = df_mouse_all#pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv") # 20190323 LO removed. 
                    valid_mouse = valid_mouse['name'][self.fold_idx]
                print('validation mouse list')
                print(valid_mouse)
                # Performance history
                all_train_loss = np.zeros(n_epochs)
                all_train_acc = np.zeros(n_epochs)
                all_train_f1 = np.zeros(n_epochs)
                all_valid_loss = np.zeros(n_epochs)
                all_valid_acc = np.zeros(n_epochs)
                all_valid_f1 = np.zeros(n_epochs)
                train_losses = []
                valid_losses = []

            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # Update parameters and compute loss of training set
                y_true_train = np.array([])
                y_pred_train = np.array([])
                train_loss = 0
                train_duration = 0
                count = 0

                #train_mouse = np.reshape(train_mouse,(-1,40)) # 20190323 LO removed. 
                for i,mouseid in enumerate(train_mouse['name']):
                    print "train:" + str(epoch) + "-" + str(count)
                    print('mouse id : '+str(mouseid))
                    csv_file_id = mouseid#[i] # 20190323 LO added. 
                    print(str(i)+'th training data loading...(DeepSleepNet)')
                    eeg_train, emg_train, y_train = data_loader.load_cv_data(csv_file_id, is_train=True)#i,is_train=True)
                    true_train, pred_train, loss, duration = \
                        self._run_epoch(
                        sess=sess, network=train_net,
                        inputs=(eeg_train,emg_train), targets=y_train,
                        train_op=train_op,
                        is_train=True
                    )
                    y_true_train = np.hstack((y_true_train,true_train))
                    y_pred_train = np.hstack((y_pred_train,pred_train))
                    print "loss:" + str(loss)
                    train_losses.append(loss)
                    train_loss = train_loss + loss
                    train_duration = train_duration + duration
                    count += 1
                train_loss = train_loss/count

                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train)
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

                y_true_val = np.array([])
                y_pred_val = np.array([])
                valid_loss = 0
                valid_duration = 0
                count = 0
                
                # Evaluate the model on the validation set
                #valid_mouse = np.reshape(valid_mouse,(-1,10)) # 20190323 LO removed. 
                for i,mouseid in enumerate([valid_mouse]):#['name']):
                    print "valid:" + str(count)
                    csv_file_id = mouseid#[i] # 20190323 LO added. 
                    print('loaded csv file name : '+str(csv_file_id))
                    print(str(i)+'th validation data loading...(DeepSleepNet)')
                    if clearDataFlag:
                        eeg_valid, emg_valid, y_valid = data_loader.load_cv_data(csv_file_id, is_train=False)#i, is_train=False) # 20190323 LO revised. 
                        #true_val, pred_val, loss, duration = \
                        #self._run_epoch(
                        #    sess=sess, network=valid_net,
                        #    inputs=(eeg_valid,emg_valid), targets=y_valid,
                        #    train_op=tf.no_op(),
                        #    is_train=False
                        #)
                    else:
                        eeg_valid, emg_valid, y_valid = data_loader.load_subject_data6(data_dir=self.data_dir, mouse=[csv_file_id])#i, is_train=False) # 20190323 LO revised. 
                    true_val, pred_val, loss, duration = \
                    self._run_epoch(
                        sess=sess, network=valid_net,
                        inputs=(eeg_valid,emg_valid), targets=y_valid,
                        train_op=tf.no_op(),
                        is_train=False
                    )
                    y_true_val = np.hstack((y_true_val,true_val))
                    y_pred_val = np.hstack((y_pred_val,pred_val))
                    valid_losses.append(loss)
                    valid_loss += loss
                    valid_duration += duration
                    count += 1
                valid_loss = valid_loss/count

                n_valid_examples = len(y_true_val)
                valid_cm = confusion_matrix(y_true_val, y_pred_val)
                valid_acc = np.mean(y_true_val == y_pred_val)
                valid_f1 = f1_score(y_true_val, y_pred_val, average="macro")

                all_train_loss[epoch] = train_loss
                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_valid_loss[epoch] = valid_loss
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1

                # db.train_log(args={
                #     "n_folds": self.n_folds,
                #     "fold_idx": self.fold_idx,
                #     "epoch": epoch,
                #     "train_step": "finetuning",
                #     "datetime": datetime.utcnow(),
                #     "model": train_net.name,
                #     "n_train_examples": n_train_examples,
                #     "n_valid_examples": n_valid_examples,
                #     "train_loss": train_loss,
                #     "train_acc": train_acc,
                #     "train_f1": train_f1,
                #     "train_duration": train_duration,
                #     "valid_loss": valid_loss,
                #     "valid_acc": valid_acc,
                #     "valid_f1": valid_f1,
                #     "valid_duration": valid_duration,
                # })

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
                output_train_losses = np.array(train_losses)
                np.savetxt(self.output_dir+"train_loss_finetune_%s.csv"%self.n_folds, output_train_losses, delimiter=",")

                output_valid_losses = np.array(valid_losses)
                np.savetxt(self.output_dir+"valid_loss_finetune_%s.csv"%self.n_folds, output_valid_losses, delimiter=",")

                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir, 16)
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)", output_dir, 16)

                # Save checkpoint
                sess.run(tf.assign(global_step, epoch+1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_epoch{}.ckpt".format(epoch)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)

                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_dict = {}
                    for v in tf.all_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_epoch{}.npz".format(epoch)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)

        print "Finish fine-tuning"
        return os.path.join(output_dir, "params_fold{}.npz".format(self.fold_idx))
