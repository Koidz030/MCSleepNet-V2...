#! /usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB

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

import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('model_dir', './mc_sleep_tf_0.11.0/model',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', './mc_sleep_tf_0.11.0/predict',
                           """Directory where to save outputs.""")

NBmodelPath = "./mc_sleep_tf_0.11.0/model/outputNB_params_13.pkl"
Model = ""

"""
def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\t"))
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
"""

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


def softmax_(logits, epsilon):
    return np.true_divide(np.exp(logits), np.repeat((np.sum(np.exp(logits), axis=1)+epsilon).reshape(-1,1), 3, axis=1))


def custom_run_epoch(
    sess,
    network,
    featurenet, # 2021/4/7 LO added
    NBmodel,
    inputs,
    train_op,
    is_train,
    output_dir
):
    #Ground truth and predictions
    y_pred_LSTM = []
    y_pred_NB = []
    
    #start_time = time.time()
    y = []

    each_x_1 = inputs[0]
    each_x_2 = inputs[1]
    # # Initialize state of LSTM - Unidirectional LSTM
    # state = sess.run(network.initial_state)

    # Initialize state of LSTM - Bidirectional LSTM
    fw_state = sess.run(network.fw_initial_state)
    bw_state = sess.run(network.bw_initial_state)

    # Prepare storage for memory cells
    n_all_data = len(each_x_1)
    extra = n_all_data % network.seq_length
    #n_data = n_all_data - extra
    #cell_size = 512
    #seq_idx = 0
    
    # Store prediction and actual stages of each patient
    each_y_pred_LSTM = [] # for check (This is the original MC-SleepNet output)
    each_y_pred_NB = [] # for check (This is the NBmodel output)
    each_y_pred_prob_LSTM = [] # predicted sleep stage probability of LSTM model.
    each_y_pred_prob_NB = [] # predicted sleep stage probability of NBmodel. 
    
    for eeg, emg in iterate_batch_seq_minibatches(eeg=each_x_1,
                                                  emg=each_x_2,
                                                  batch_size=network.batch_size,
                                                  seq_length=network.seq_length):
        feed_dict = {
            network.input_eeg_var: eeg,
            network.input_emg_var: emg,
        }
        for i, (c, h) in enumerate(network.fw_initial_state):
            feed_dict[c] = fw_state[i].c
            feed_dict[h] = fw_state[i].h
        for i, (c, h) in enumerate(network.bw_initial_state):
            feed_dict[c] = bw_state[i].c
            feed_dict[h] = bw_state[i].h
        _, y_pred_LSTM, fw_state, bw_state, logits = sess.run(
            [train_op, network.pred_op,network.fw_final_state, network.bw_final_state, network.logits],
            feed_dict=feed_dict
        )
        #print("logits: ", str(logits))
        #print("length of logits: ", str(len(logits)))
        each_y_pred_LSTM.extend(y_pred_LSTM)
        y_pred_prob_LSTM = softmax_(logits, 1e-10)
        if len(each_y_pred_prob_LSTM)==0:
            each_y_pred_prob_LSTM = y_pred_prob_LSTM
        else:
            each_y_pred_prob_LSTM = np.concatenate([each_y_pred_prob_LSTM, y_pred_prob_LSTM], axis=0)
        
        #adding GaussianNB
        x_train=np.array([
            sess.run(featurenet.features,
                    feed_dict={
                        featurenet.input_eeg_var:eeg[j,:,:,:].reshape(1, 5000, 1, 1),
                        featurenet.input_emg_var:emg[j,:,:,:].reshape(1, 5000, 1, 1)
                     }
            )
            for j in range(eeg.shape[0])
        ])
        x_train=x_train.reshape(x_train.shape[0],-1)
        y_pred_NB=NBmodel.predict_proba(x_train)
        y_pred_prob_NB=NBmodel.predict_proba(x_train)
        each_y_pred_NB.extend(y_pred_NB) # sleep stage label sequence (0:W, 1:NR, 2:R)
        if len(each_y_pred_prob_NB)==0:
            each_y_pred_prob_NB = y_pred_prob_NB
        else:
            each_y_pred_prob_NB=np.concatenate([each_y_pred_prob_NB, y_pred_prob_NB], axis=0)

    if extra != 0:
        feed_dict = {
            network.input_eeg_var: each_x_1[-25:],
            network.input_emg_var: each_x_2[-25:]
            }
        _, y_pred_LSTM, logits = sess.run(
            [train_op, network.pred_op, network.logits],
            feed_dict = feed_dict
        )
        y_pred_LSTM = y_pred_LSTM[-extra:]
        each_y_pred_LSTM.extend(y_pred_LSTM)
        y_pred_prob_LSTM = softmax_(logits[-extra:], 1e-10)
        each_y_pred_prob_LSTM = np.concatenate([each_y_pred_prob_LSTM, y_pred_prob_LSTM], axis=0) # 2021/4/9 LO added.
        # 2021/4/9 LO revised for prediction of the extra data (last rest data).
        x_train = np.array([
            sess.run(
                featurenet.features,
                feed_dict = {
                    featurenet.input_eeg_var:eeg[j,:,:,:].reshape(1, 5000, 1, 1),
                    featurenet.input_emg_var:emg[j,:,:,:].reshape(1, 5000, 1, 1)
                }
            )
            for j in range(-extra,0)
        ])
        x_train=x_train.reshape(x_train.shape[0],-1)
        y_pred_NB = NBmodel.predict(x_train)
        y_pred_prob_NB = NBmodel.predict_proba(x_train)
        # 2021/4/9 LO revised part ends.
        each_y_pred_NB.extend(y_pred_NB)
        each_y_pred_prob_NB = np.concatenate([each_y_pred_prob_NB, y_pred_prob_NB], axis=0)

    y_pred_consol=[define_stage(prob1, prob2) for prob1, prob2 in zip(each_y_pred_prob_LSTM, each_y_pred_prob_NB)] #J modified 09/04/21
    y.append(y_pred_consol)
    y_pred_consol = pd.DataFrame(y).values.flatten() #J added 09/04/21
    y_pred_consol[y_pred_consol==0] = 65
    y_pred_consol[y_pred_consol==1] = 49
    y_pred_consol[y_pred_consol==2] = 82
    pd.DataFrame(y_pred_consol,index=range(1,len(y_pred_consol)+1)).to_csv("predict/output_subject.csv",header=False,index=True)


# Consolidate the outputs
def define_stage(prob1, prob2):
    # Find max value of an average between two terms
    #average=0.25*((3*prob1)+prob2)
    average=0.2*((4*prob1)+prob2)
    argmax_average=np.argmax(average)
    return argmax_average


def predict(
    data_dir,
    model_dir,
    output_dir,
):
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

        mouse = "target"
        checkpoint_path = os.path.join(
                        model_dir,
                        "deepsleepnet",
                        Model
                        )


        # Definition of Gaussian Naive Bayes model
        NBmodel=GaussianNB()
        with open(NBmodelPath, 'rb') as f:
            NBmodel=pickle.load(f, encoding='latin1')
        # Definition of DeepFeatureNet (MC-SleepNet feature extraction part)
        from train_NB import customDeepFeatureNet
        featurenet=customDeepFeatureNet(
            batch_size=1,
            input_dims=20*250,
            n_classes=3,
            is_train=False,
            reuse_params=True, # 2021/4/7 LO added.
            use_dropout=True
        )
        featurenet.init_ops()

        for v in tf.get_collection(tf.GraphKeys.VARIABLES):
            print(v.name)
        # Restore the trained model
        print(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print ("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))
        

        # Load testing data
        eeg,emg = SeqDataLoader.load_subject_data(
            data_dir=data_dir,
            mouse = mouse
        )
        # Loop each epoch
        print ("[{}] Predicting ...\n".format(datetime.now()))
        custom_run_epoch(
            sess=sess, network=valid_net,
            featurenet=featurenet, NBmodel=NBmodel, # 2021/4/7 LO added.
            inputs=(eeg[0],emg[0]),
            train_op=tf.no_op(),
            is_train=False,
            output_dir=output_dir
        )


def main(argv=None):
    # # Makes the random numbers predictable
    # np.random.seed(0)
    # tf.set_random_seed(0)
    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    predict(
        data_dir=FLAGS.data_dir,
        model_dir=FLAGS.model_dir,
        output_dir=FLAGS.output_dir,
    )


if __name__ == "__main__":
    tf.app.run()
