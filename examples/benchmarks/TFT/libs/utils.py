# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.

    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude

    Returns:
      List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for tensorflow.

    Standard quantile loss as defined in the "Training Procedure" section of
    the main TFT paper

    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
      Tensor for quantile loss.
    """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            'Illegal quantile value={}! Values should be between 0 and 1.'.format(
                quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
        1. - quantile) * tf.maximum(-prediction_underflow, 0.)

    return tf.reduce_sum(input_tensor=q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    """Computes normalised quantile loss for numpy arrays.

    Uses the q-Risk metric as defined in the "Training Procedure" section of the
    main TFT paper.

    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
      Float for normalised quantile loss.
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
        + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

    quantile_loss = weighted_errors.mean()
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

    Args:
      directory: Folder path to create.
    """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device='gpu', gpu_id=0):
    """Creates tensorflow config for graphs to run on CPU or GPU.

    Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
    GPU machines.

    Args:
      tf_device: 'cpu' or 'gpu'
      gpu_id: GPU ID to use if relevant

    Returns:
      Tensorflow config.
    """

    if tf_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
        tf_config = tf.compat.v1.ConfigProto(
            log_device_placement=False, device_count={'GPU': 0})

    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print('Selecting GPU ID={}'.format(gpu_id))

        tf_config = tf.compat.v1.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def set_tf_device(tf_device='gpu', gpu_id=0):
    """Configure TensorFlow 2 to use CPU or a specific GPU with growth enabled."""

    if tf_device == 'cpu':
        # 关闭所有 GPU
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU only.")
    else:
        # 选择第 gpu_id 张 GPU
        print("TF version:", tf.__version__)
        print("Built with CUDA:", tf.test.is_built_with_cuda())
        print("GPU available:", tf.config.list_physical_devices('GPU'))
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 只让 TensorFlow 看见特定 GPU
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')

                # 设置显存自适应增长
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

                print(f"Using GPU ID = {gpu_id}")

            except RuntimeError as e:
                print(e)
        else:
            print("No GPU detected, using CPU.")


def save(tf_session, model_folder, cp_name, scope=None):
    """Saves Tensorflow graph to checkpoint.

    Saves all trainiable variables under a given variable scope to checkpoint.

    Args:
      tf_session: Session containing graph
      model_folder: Folder to save models
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope containing variables to save
    """
    # Save model
    if scope is None:
        saver = tf.compat.v1.train.Saver()
    else:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session,
                           os.path.join(model_folder, '{0}.ckpt'.format(cp_name)))
    print('Model saved to: {0}'.format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """Loads Tensorflow graph from checkpoint.

    Args:
      tf_session: Session to load graph into
      model_folder: Folder containing serialised model
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope to use.
      verbose: Whether to print additional debugging information.
    """
    # Load model proper
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print('Loading model from {0}'.format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set(
        [v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.compat.v1.train.Saver()
    else:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node])

    if verbose:
        print('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
        print('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
        print('All {0}'.format(','.join(all_vars)))

    print('Done.')


def print_weights_in_checkpoint(model_folder, cp_name):
    """Prints all weights in Tensorflow checkpoint.

    Args:
      model_folder: Folder containing checkpoint
      cp_name: Name of checkpoint

    Returns:

    """
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print_tensors_in_checkpoint_file(
        file_name=load_path,
        tensor_name='',
        all_tensors=True,
        all_tensor_names=True)


def get_record_base_dir(file_path, run_path):
    pos = file_path.find(run_path)
    return file_path[0:pos]


# PROVIDER_URI = "./qlib_data/cn_data"
# MLRUNS_PATH = "mlruns_tft_sh600000"
# MARKET = "sh600000"
# BENCHMARK = "SH000300"

PROVIDER_URI = "./qlib_data_btc"
MLRUNS_PATH = "mlruns_tft_btc"
MARKET = "all"
BENCHMARK = "SH000300"

# TRAIN_START_TIME = "2008-01-01"
# TRAIN_END_TIME = "2014-12-31" # 7
# VALID_START_TIME = "2015-01-01"
# VALID_END_TIME = "2016-12-31" # 2
# TEST_START_TIME = "2017-01-01"
# TEST_END_TIME = "2020-08-01" # 4

# TRAIN_START_TIME = "2005-01-01"
# TRAIN_END_TIME = "2014-12-31"  # 10
# VALID_START_TIME = "2015-01-01"
# VALID_END_TIME = "2016-12-31"  # 2
# TEST_START_TIME = "2017-01-01"
# TEST_END_TIME = "2020-08-01"  # 4

TRAIN_START_TIME = "2017-08-17 04:00:00"
TRAIN_END_TIME = "2021-12-31 23:00:00"  # 5
VALID_START_TIME = "2022-01-01 00:00:00"
VALID_END_TIME = "2023-12-31 23:00:00"  # 2
TEST_START_TIME = "2024-01-01 00:00:00"
TEST_END_TIME = "2025-11-30 23:00:00"  # 2

# DATA_HANDLER_CONFIG = {
#     "start_time": TRAIN_START_TIME,
#     "end_time": TEST_END_TIME,
#     "fit_start_time": TRAIN_START_TIME,
#     "fit_end_time": TRAIN_END_TIME,
#     "instruments": MARKET,
#     "freq": "day",
#     "learn_processors": [{
#             "class": "DropnaLabel"
#     }]
# }

# DATA_HANDLER_CONFIG = {
#     "start_time": TRAIN_START_TIME,
#     "end_time": TEST_END_TIME,
#     "fit_start_time": TRAIN_START_TIME,
#     "fit_end_time": TRAIN_END_TIME,
#     "instruments": MARKET,
#     "freq": "day",
# }

DATA_HANDLER_CONFIG = {
    "start_time": TRAIN_START_TIME,
    "end_time": TEST_END_TIME,
    "fit_start_time": TRAIN_START_TIME,
    "fit_end_time": TRAIN_END_TIME,
    "instruments": MARKET,
    "freq": "60min",
    "learn_processors": [{
            "class": "DropnaLabel"
    }]
}

TASK_CONFIG = {
    "model": {
        "class": "TFTModel",
        "module_path": "tft",
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                 "class": "Alpha158",
                 "module_path": "qlib.contrib.data.handler",
                 "kwargs": DATA_HANDLER_CONFIG,
            },
            "segments": {
                "train": (TRAIN_START_TIME, TRAIN_END_TIME),
                "valid": (VALID_START_TIME, VALID_END_TIME),
                "test": (TEST_START_TIME, TEST_END_TIME),
            }
        },
    },
}
