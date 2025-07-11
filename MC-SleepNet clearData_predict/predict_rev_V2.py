#! /usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from datetime import datetime as dt

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from deepsleep.data_loader import SeqDataLoader
from deepsleep.model import DeepSleepNet
from deepsleep.nn import *
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)
from deepsleep.utils import iterate_batch_seq_minibatches

import sys
#print('python code name : '+sys.argv[0])
#print('data saved directory : '+sys.argv[1])
#print('output directory : '+sys.argv[2])
#print('number of fold of cross validation : '+sys.argv[3])
#print('number of epochs for pretraining : '+sys.argv[4])
#print('number of epochs for fine-tuning : '+sys.argv[5])
#print('fold number id of cross-validation : '+sys.argv[4])#6])
#print('resume : '+sys.argv[5])#7])
dtnow = str(dt.now())
print(dtnow)

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('data_dir', '/data2/clearData',
#                           """Directory where to load training data.""")
#tf.app.flags.DEFINE_string('model_dir', '/data2/clearData/output/cross_1/fold1/deepsleepnet',
#                           """Directory where to load trained models.""")
#tf.app.flags.DEFINE_string('output_dir', '/data2/clearData/output/cross_1/fold1/deepsleepnet/predict_result',
#                           """Directory where to save outputs.""")
tf.app.flags.DEFINE_string('data_dir', "/data2/clearData20190330", # sys.argv[1],
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('model_dir', "/home/juan/model_epoch19.ckpt-20", #"/home/ota/cross_1/fold1/deepsleepnet/model_epoch19.ckpt-20", #  sys.argv[2],
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', "/home/juan/predict", #sys.argv[2],
                           """Directory where to save outputs.""")
tf.app.flags.DEFINE_string('n_fold', 1, # Not cross validation, but prediction by original MC-SleepNet model (large dataset)  sys.argv[3],
                           """Number of fold of cross validation.""")

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
    print (cm)
    print (" ")


def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.
    Args:
        input_seq: Sequence of seq_len tensors of dimension (batch_size, n_features)
                   or nested tuples of tensors.
        lengths:   A `Tensor` of dimension batch_size, containing lengths for each
                   sequence in the batch. If "None" is specified, simply reverses
                   the list.
    Returns:
        time-reversed sequence
    """
    if lengths is None:
        return list(reversed(input_seq))

    flat_input_seq = tuple(nest.flatten(input_) for input_ in input_seq)

    flat_results = [[] for _ in range(len(input_seq))]
    for sequence in zip(*flat_input_seq):
        input_shape = tensor_shape.unknown_shape(
                ndims=sequence[0].get_shape().ndims)
        for input_ in sequence:
            input_shape.merge_with(input_.get_shape())
            input_.set_shape(input_shape)

        # Join into (time, batch_size, depth)
        s_joined = array_ops.pack(sequence)

        # TODO(schuster, ebrevdo): Remove cast when reverse_sequence takes int32
        if lengths is not None:
            lengths = math_ops.to_int64(lengths)

        # Reverse along dimension 0
        s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
        # Split again into list
        result = array_ops.unpack(s_reversed)
        for r, flat_result in zip(result, flat_results):
            r.set_shape(input_shape)
            flat_result.append(r)

    results = [nest.pack_sequence_as(structure=input_, flat_sequence=flat_result)
               for input_, flat_result in zip(input_seq, flat_results)]
    return results


def custom_rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.
    The simplest form of RNN network generated is:
    ```python
        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
            output, state = cell(input_, state)
            outputs.append(output)
        return (outputs, state)
    ```
    However, a few other options are available:
    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.
    The dynamic calculation performed is, at time `t` for batch row `b`,
    ```python
        (output, state)(b, t) =
            (t >= sequence_length(b))
                ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
                : cell(input(b, t), state(b, t - 1))
    ```
    Args:
        cell: An instance of RNNCell.
        inputs: A length T list of inputs, each a `Tensor` of shape
            `[batch_size, input_size]`, or a nested tuple of such elements.
        initial_state: (optional) An initial state for the RNN.
            If `cell.state_size` is an integer, this must be
            a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
            If `cell.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state and expected output.
            Required if initial_state is not provided or RNN state has a heterogeneous
            dtype.
        sequence_length: Specifies the length of each sequence in inputs.
            An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
        scope: VariableScope for the created subgraph; defaults to "RNN".
    Returns:
        A pair (outputs, state) where:
        - outputs is a length T list of outputs (one for each input), or a nested
            tuple of such elements.
        - state is the final state
    Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If `inputs` is `None` or an empty list, or if the input depth
            (column size) cannot be inferred from inputs via shape inference.
    """

    if not isinstance(cell, rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    states = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Obtain the first sequence of the input
        first_input = inputs
        while nest.is_sequence(first_input):
            first_input = first_input[0]

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = nest.flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "Input size (dimension %d of inputs) must be accessible via "
                            "shape inference, but saw value None." % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # Prepare variables
            sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size")
            def _create_zero_output(output_size):
                # convert int to TensorShape if necessary
                size = _state_size_with_prefix(output_size, prefix=[batch_size])
                output = array_ops.zeros(
                    array_ops.pack(size), _infer_state_dtype(dtype, state))
                shape = _state_size_with_prefix(
                    output_size, prefix=[fixed_batch_size.value])
                output.set_shape(tensor_shape.TensorShape(shape))
                return output

            output_size = cell.output_size
            flat_output_size = nest.flatten(output_size)
            flat_zero_output = tuple(
                _create_zero_output(size) for size in flat_output_size)
            zero_output = nest.pack_sequence_as(structure=output_size,
                                                flat_sequence=flat_zero_output)

            sequence_length = math_ops.to_int32(sequence_length)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = _rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()

            outputs.append(output)
            states.append(state)

        return (outputs, state, states)


def custom_bidirectional_rnn(cell_fw, cell_bw, inputs,
                             initial_state_fw=None, initial_state_bw=None,
                             dtype=None, sequence_length=None, scope=None):
    """Creates a bidirectional recurrent neural network.
    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.
    Args:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: A length T list of inputs, each a tensor of shape
            [batch_size, input_size], or a nested tuple of such elements.
        initial_state_fw: (optional) An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape
            `[batch_size, cell_fw.state_size]`.
            If `cell_fw.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
        initial_state_bw: (optional) Same as for `initial_state_fw`, but using
            the corresponding properties of `cell_bw`.
        dtype: (optional) The data type for the initial state.  Required if
            either of the initial states are not provided.
        sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
            containing the actual lengths for each of the sequences.
        scope: VariableScope for the created subgraph; defaults to "BiRNN"
    Returns:
        A tuple (outputs, output_state_fw, output_state_bw) where:
            outputs is a length `T` list of outputs (one for each input), which
                are depth-concatenated forward and backward outputs.
            output_state_fw is the final state of the forward rnn.
            output_state_bw is the final state of the backward rnn.
    Raises:
        TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
        ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell_fw, rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    with vs.variable_scope(scope or "BiRNN"):
        # Forward direction
        with vs.variable_scope("FW") as fw_scope:
            output_fw, output_state_fw, fw_states = custom_rnn(
                cell_fw, inputs, initial_state_fw, dtype,
                sequence_length, scope=fw_scope
            )

        # Backward direction
        with vs.variable_scope("BW") as bw_scope:
            reversed_inputs = _reverse_seq(inputs, sequence_length)
            tmp, output_state_bw, tmp_states = custom_rnn(
                cell_bw, reversed_inputs, initial_state_bw,
                dtype, sequence_length, scope=bw_scope
            )

    output_bw = _reverse_seq(tmp, sequence_length)
    bw_states = _reverse_seq(tmp_states, sequence_length)

    # Concat each of the forward/backward outputs
    flat_output_fw = nest.flatten(output_fw)
    flat_output_bw = nest.flatten(output_bw)

    flat_outputs = tuple(array_ops.concat(1, [fw, bw])
                        for fw, bw in zip(flat_output_fw, flat_output_bw))

    outputs = nest.pack_sequence_as(structure=output_fw,
                                    flat_sequence=flat_outputs)

    return (outputs, output_state_fw, output_state_bw, fw_states, bw_states)


class CustomDeepSleepNet(DeepSleepNet):

    def __init__(
        self,
        batch_size,
        input_dims,
        n_classes,
        seq_length,
        n_rnn_layers,
        return_last,
        is_train,
        reuse_params,
        use_dropout_feature,
        use_dropout_sequence,
        name="deepsleepnet"
    ):
        super(DeepSleepNet, self).__init__(
            batch_size=batch_size,
            input_dims=input_dims,
            n_classes=n_classes,
            is_train=is_train,
            reuse_params=reuse_params,
            use_dropout=use_dropout_feature,
            name=name
        )

        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last

        self.use_dropout_sequence = use_dropout_sequence

    def build_model(self, input_eeg_var, input_emg_var):
        # Create a network with superclass method
        network = super(DeepSleepNet, self).build_model(
            input_eeg_var=self.input_eeg_var,
            input_emg_var=self.input_emg_var
        )

        # Residual (or shortcut) connection
        output_conns = []

        # Fully-connected to select some part of the output to add with the output from bi-directional LSTM
        name = "l{}_fc".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output_tmp = fc(name="fc", input_var=network, n_hiddens=1024, bias=None, wd=0)
            output_tmp = batch_norm_new(name="bn", input_var=output_tmp, is_train=self.is_train)
            output_tmp = tf.nn.relu(output_tmp, name="relu")
        self.activations.append((name, output_tmp))
        self.layer_idx += 1
        output_conns.append(output_tmp)

        ######################################################################

        # Reshape the input from (batch_size * seq_length, input_dim) to
        # (batch_size, seq_length, input_dim)
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1].value
        seq_input = tf.reshape(network,
                               shape=[-1, self.seq_length, input_dim],
                               name=name)
        assert self.batch_size == seq_input.get_shape()[0].value
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        # Bidirectional LSTM network
        name = "l{}_bi_lstm".format(self.layer_idx)
        hidden_size = 512   # will output 1024 (512 forward, 512 backward)
        with tf.variable_scope(name) as scope:
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size,
                                                   use_peepholes=True,
                                                   state_is_tuple=True)
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size,
                                                   use_peepholes=True,
                                                   state_is_tuple=True)
            if self.use_dropout_sequence:
                keep_prob = 0.5 if self.is_train else 1.0
                fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    fw_lstm_cell,
                    output_keep_prob=keep_prob
                )
                bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    bw_lstm_cell,
                    output_keep_prob=keep_prob
                )

            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * self.n_rnn_layers,
                                                  state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * self.n_rnn_layers,
                                                  state_is_tuple=True)

            # Initial state of RNN
            self.fw_initial_state = fw_cell.zero_state(self.batch_size, tf.float32)
            self.bw_initial_state = bw_cell.zero_state(self.batch_size, tf.float32)

            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unpack(seq_input, axis=1)
            outputs, fw_state, bw_state, fw_states, bw_states = custom_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if self.return_last:
                network = outputs[-1]
            else:
                network = tf.reshape(tf.concat(1, outputs), [-1, hidden_size*2],
                                     name=name)
            self.activations.append((name, network))
            self.layer_idx +=1

            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

            self.fw_states = fw_states
            self.bw_states = bw_states

        # Append output
        output_conns.append(network)

        ######################################################################

        # Add
        name = "l{}_add".format(self.layer_idx)
        network = tf.add_n(output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout_sequence:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return network


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

    # Ground truth and predictions
    y_pred = []

    start_time = time.time()
    y = []
    y_true = []
    W_prob = []
    NR_prob = []
    R_prob = []
    all_fw_memory_cells = []
    all_bw_memory_cells = []
    total_loss, n_batches = 0.0, 0
    for sub_f_idx, each_data in enumerate(itertools.izip(inputs[0],inputs[1], targets)):
        each_x_1,each_x_2, each_y = each_data

        # # Initialize state of LSTM - Unidirectional LSTM
        # state = sess.run(network.initial_state)

        # Initialize state of LSTM - Bidirectional LSTM
        fw_state = sess.run(network.fw_initial_state)
        bw_state = sess.run(network.bw_initial_state)

        # Prepare storage for memory cells
        n_all_data = len(each_x_1)
        extra = n_all_data % network.seq_length
        n_data = n_all_data - extra
        cell_size = 512
        fw_memory_cells = np.zeros((n_data, network.n_rnn_layers, cell_size))
        bw_memory_cells = np.zeros((n_data, network.n_rnn_layers, cell_size))
        seq_idx = 0

        # Store prediction and actual stages of each patient
        each_y_true = []
        each_y_pred = []
        each_w_prob = []
        each_nr_prob = []
        each_r_prob = []
        all_logits = [] # 29/06/21 J added

        for eeg, emg,  y_batch in iterate_batch_seq_minibatches(eeg=each_x_1,
                                                                emg=each_x_2,
                                                              targets=each_y,
                                                              batch_size=network.batch_size,
                                                              seq_length=network.seq_length):
            feed_dict = {
                network.input_eeg_var: eeg,
                network.input_emg_var: emg,
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

            _, loss_value, y_pred, fw_state, bw_state, logits = sess.run(
                [train_op, network.loss_op, network.pred_op, network.fw_final_state, network.bw_final_state, network.logits], # network.prob],
                feed_dict=feed_dict
            )# 29/06/21 J added logits output for each 25 epoch sequence.

            ######### 29/06/21 J added to save logit output. ############
            print("shape of logits: "+str(logits.shape))
            if len(all_logits)==0:
                all_logits = logits
            else:
                all_logits = np.concatenate([all_logits, logits], axis=0) # 2021/6/4 LO added.
            ######### 29/06/21 J added to save logit output. ############

            # Extract memory cells
            fw_states = sess.run(network.fw_states, feed_dict=feed_dict)
            bw_states = sess.run(network.bw_states, feed_dict=feed_dict)
            offset_idx = seq_idx * network.seq_length
            for s_idx in range(network.seq_length):
                for r_idx in range(network.n_rnn_layers):
                    fw_memory_cells[offset_idx + s_idx][r_idx] = np.squeeze(fw_states[s_idx][r_idx].c)
                    bw_memory_cells[offset_idx + s_idx][r_idx] = np.squeeze(bw_states[s_idx][r_idx].c)
            seq_idx += 1
            each_y_true.extend(y_batch)
            each_y_pred.extend(y_pred)
            each_w_prob.extend(logits[:,0])
            each_nr_prob.extend(logits[:,1])
            each_r_prob.extend(logits[:,2])

            total_loss += loss_value
            n_batches += 1

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        all_fw_memory_cells.append(fw_memory_cells)
        all_bw_memory_cells.append(bw_memory_cells)
        y.append(each_y_pred)
        y_true.append(each_y_true)
        W_prob.append(each_w_prob)
        NR_prob.append(each_nr_prob)
        R_prob.append(each_r_prob)

    # Save memory cells and predictions
    save_dict = {
        #"fw_memory_cells": fw_memory_cells,
        #"bw_memory_cells": bw_memory_cells,
        "y_true": np.array(y_true).astype(np.int8),
        "y_pred": np.array(y).astype(np.int8),
        "W_prob": np.array(W_prob).astype(np.float16),
        "NR_prob": np.array(NR_prob).astype(np.float16),
        "R_prob": np.array(R_prob).astype(np.float16)
    }
    print subject_idx
    save_path = os.path.join(
        output_dir,
        "output_subject_{}.npz".format(subject_idx)
    )
    np.savez(save_path, **save_dict)
    print "Saved outputs to {}".format(save_path)

    ######### 29/06/21 J added to save prediction and logit output. ############
    y_pred = pd.DataFrame(y).values.flatten()

    y_pred[y_pred==0] = 65
    y_pred[y_pred==1] = 49
    y_pred[y_pred==2] = 82
    pd.DataFrame(y_pred,index=range(1,len(y_pred)+1)).to_csv(output_dir+"/output_subject_%s.csv"%(subject_idx),header=False,index=True)
    pd.DataFrame(all_logits, index=range(1,all_logits.shape[0]+1)).to_csv(output_dir+"/output_subject_logits%s.csv"%(subject_idx), header=False, index=True) # W, NR, R logit output for each epochs.
    ######### 29/06/21 J added to save prediction and logit output. ############

    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)
    print duration
    return total_y_true, total_y_pred, total_loss, duration


def predict(
    data_dir,
    model_dir,
    output_dir,
    n_subjects,
    n_subjects_per_fold
):
    # Ground truth and predictions
    y_true = []
    y_pred = []

    #mouse = np.array(pd.read_csv("/data2/clearData20190330/train_mouse_all_noisy.csv")).reshape(-1) # 20190514 LO for analysis clear data. (14 mice)
    #mouse = np.array(pd.read_csv("/data2/clearData20190330/Y0mouselist.csv")).reshape(-1) # 20190807 LO for analysis of noisy data. (4 mice)
    mouse = np.array(pd.read_csv("/data2/clearData20190330/train_mouse_all_2.csv")).reshape(-1) # Juan-san's experiment is for all of 15 mice
    #mouse = np.array(pd.read_csv("/data2/clearData20190330/train_mouse_all.csv")).reshape(-1) # original experiment is for 14 mice clear signal (B6J1~B6J15, except for B6J8)
    print mouse
    for fold_idx in range(n_subjects//n_subjects_per_fold):
        # The model will be built into the default Graph
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build the network
            valid_net = CustomDeepSleepNet(
                batch_size=1,
                input_dims=EPOCH_SEC_LEN*250,
                n_classes=NUM_CLASSES,
                seq_length=25,
                n_rnn_layers=2,
                return_last=False,
                is_train=False,
                reuse_params=False,
                use_dropout_feature=True,
                use_dropout_sequence=True
            )

            # Initialize parameters
            valid_net.init_ops()
            # Restore the trained model
            # 20190519 LO for clear data. 
            checkpoint_path = "/home/ota/cross_1/fold1/deepsleepnet/model_epoch19.ckpt-20"
                #model_dir+'cross_'+str(fold_idx+1)+'/fold'+str(fold_idx+1)+'/deepsleepnet/'#model_epoch199.ckpt-200'
            #)
            # 20190807 LO for noisy data. 
            #checkpoint_path = os.path.join(
            #    model_dir+'cross_'+str(fold_idx)+'/fold'+str(fold_idx)+'/deepsleepnet/'#model_epoch199.ckpt-200'
            #)
            print(checkpoint_path)
            print('path exists or not : ', os.path.exists(checkpoint_path))
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path) # tf.train.latest_checkpoint(checkpoint_path))
            print "Model restored from: {}\n".format(checkpoint_path) # tf.train.latest_checkpoint(checkpoint_path))


            #mouse = np.array(pd.read_csv("/data2/data/train_mouse_cross2.csv",header=None,index_col=None,dtype=int)).reshape(-1)
            #mouse = np.arange(13,15)

            data_loader = SeqDataLoader(
                data_dir=data_dir,
            )
            #for subject_idx, subject_name in enumerate(mouse):
            for num, mouseid in enumerate(mouse[(fold_idx*n_subjects_per_fold):((fold_idx*n_subjects_per_fold)+n_subjects_per_fold)]):
                #fold_idx = 1

                # Load testing data
                #eeg,emg, y = data_loader.load_subject_data(
                #    data_dir=data_dir,
                #    subject_idx=subject_idx,
                #    mouse = mouse
                #)
                #eeg,emg, y = data_loader.load_subject_data(
                #    data_dir=data_dir,
                #    subject_idx=mouseid#(fold_idx*n_subjects_per_fold)+num,
                #    mouse = mouse#[(fold_idx*n_subjects_per_fold):((fold_idx*n_subjects_per_fold)+n_subjects_per_fold)]
                #)
                eeg,emg, y = data_loader.load_subject_data(
                    data_dir=data_dir,
                    subject_idx=num,
                    mouse = mouse[(fold_idx*n_subjects_per_fold):((fold_idx*n_subjects_per_fold)+n_subjects_per_fold)]
                )

                # Loop each epoch
                print "[{}] Predicting ...\n".format(datetime.now())

                # Evaluate the model on the subject data
                #y_true_, y_pred_, loss, duration = \
                #    custom_run_epoch(
                #        sess=sess, network=valid_net,
                #        inputs=(eeg,emg), targets=y,
                #        train_op=tf.no_op(),
                #        is_train=False,
                #        output_dir=output_dir,
                #        subject_idx=subject_name
                #    )
                # for clear data
                #if not os.path.exists(output_dir+'cross_'+str(fold_idx+1)+'/fold'+str(fold_idx+1)+'/deepsleepnet/predict_result'):
                #    os.makedirs(os.path.join(output_dir+'cross_'+str(fold_idx+1)+'/fold'+str(fold_idx+1)+'/deepsleepnet/predict_result'))
                # for noisy data
                #if not os.path.exists(output_dir+'cross_'+str(fold_idx)+'/fold'+str(fold_idx)+'/deepsleepnet/predict_result_17275'):
                #    os.makedirs(os.path.join(output_dir+'cross_'+str(fold_idx)+'/fold'+str(fold_idx)+'/deepsleepnet/predict_result_17275'))
                # for clear data
                y_true_, y_pred_, loss, duration = \
                    custom_run_epoch(
                        sess=sess, network=valid_net,
                        inputs=(eeg,emg), targets=y,
                        train_op=tf.no_op(),
                        is_train=False,
                        output_dir=output_dir,#+'cross_'+str(fold_idx+1)+'/fold'+str(fold_idx+1)+'/deepsleepnet/predict_result',
                        subject_idx=mouse[(fold_idx*n_subjects_per_fold)+num]#mouseid
                    )
                # for noisy data
                #y_true_, y_pred_, loss, duration = \
                #    custom_run_epoch(
                #        sess=sess, network=valid_net,
                #        inputs=(eeg,emg), targets=y,
                #        train_op=tf.no_op(),
                #        is_train=False,
                #        output_dir=output_dir+'cross_'+str(fold_idx)+'/fold'+str(fold_idx)+'/deepsleepnet/predict_result_17275',
                #        subject_idx=mouse[(fold_idx*n_subjects_per_fold)+num]#mouseid
                #    )
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
            del saver
            del sess

    # Overall performance
    print "[{}] Overall prediction performance\n".format(datetime.now())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_examples = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    cm = np.array(cm).astype(dtype=np.float32)
    acc = np.trace(cm)/np.sum(np.sum(cm))
    mf1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    print (
        "n={}, acc={:.3f}, f1={:.3f}, kappa={:.3f}".format(
            n_examples, acc, mf1, kappa
        )
    )
    print cm
    dtnow = str(dt.now())
    #np.savez_compressed(output_dir+'result_overall_sleep_17275epoch_'+str(dtnow[0:19].replace(' ', '_').replace(':', ''))+'.npz', y_true=y_true, y_pred=y_pred, acc=acc, macro_f1=mf1, kappa=kappa)


def main(argv=None):
    # Makes the random numbers predictable
    np.random.seed(0)
    tf.set_random_seed(0)

    # Output dir
    #if not os.path.exists(FLAGS.output_dir):
    #    os.makedirs(FLAGS.output_dir)
    
    n_subjects = 15 # for clear data. (B6JX)  29/06/21 J added for prediction of 1 mouse
    #n_subjects = 14# for clear data. (B6JX)
    #n_subjects = 4# for noisy data. (Y0-------)
    #n_subjects_per_fold = n_subjects//int(FLAGS.n_fold)#2#1
    n_subjects_per_fold = 1 # all mice were called one-by-one # 2021/6/29 LO revised for clear mice prediction by large MC-SleepNet model
    #n_subjects_per_fold = 2# for clear data. 7-fold cross validation
    #n_subjects_per_fold = 1# for noisy data. 4-fold cross validation
    predict(
        data_dir=FLAGS.data_dir,
        model_dir=FLAGS.model_dir,
        output_dir=FLAGS.output_dir,
        n_subjects=n_subjects,
        n_subjects_per_fold=n_subjects_per_fold
    )


if __name__ == "__main__":
    tf.app.run()
