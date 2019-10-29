# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for training and evaluation of representation learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os

from absl import flags

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.contrib.tpu import RunConfig, TPUConfig, TPUEstimator
from tensorflow.contrib.training import checkpoints_iterator

import datasets
import utils
import semi_supervised

FLAGS = flags.FLAGS

# Number of iterations (=training steps) per TPU training loop. Use >100 for
# good speed. This is the minimum number of steps between checkpoints.
TPU_ITERATIONS_PER_LOOP = 200


def train_and_eval():
  """Trains a network on (self) supervised data."""
  checkpoint_dir = FLAGS.get_flag_value("checkpoint", FLAGS.workdir)
  tf.gfile.MakeDirs(checkpoint_dir)

  if FLAGS.tpu_name:
    cluster = TPUClusterResolver(tpu=[FLAGS.tpu_name])
  else:
    cluster = None

  # tf.logging.info("master: %s", master)
  config = RunConfig(
      model_dir=checkpoint_dir,
      tf_random_seed=FLAGS.random_seed,
      cluster=cluster,
      keep_checkpoint_max=None,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=TPUConfig(iterations_per_loop=TPU_ITERATIONS_PER_LOOP))

  # Optionally resume from a stored checkpoint.
  if FLAGS.path_to_initial_ckpt:
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAGS.path_to_initial_ckpt,
        # The square bracket is important for loading all the
        # variables from GLOBAL_VARIABLES collection.
        # See https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings  # pylint: disable=line-too-long
        # section vars_to_warm_start for more details.
        vars_to_warm_start=[FLAGS.vars_to_restore]
    )
  else:
    warm_start_from = None

  # The global batch-sizes are passed to the TPU estimator, and it will pass
  # along the local batch size in the model_fn's `params` argument dict.
  estimator = TPUEstimator(
      model_fn=semi_supervised.get_model(FLAGS.task),
      model_dir=checkpoint_dir,
      config=config,
      use_tpu=FLAGS.tpu_name is not None,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.get_flag_value("eval_batch_size", FLAGS.batch_size),
      warm_start_from=warm_start_from
  )

  if FLAGS.run_eval:
    data_fn = functools.partial(
        datasets.get_data,
        split_name=FLAGS.val_split,
        preprocessing=FLAGS.get_flag_value("preprocessing_eval",
                                           FLAGS.preprocessing),
        is_training=False,
        shuffle=False,
        num_epochs=1,
        drop_remainder=True)

    # Contrary to what the documentation claims, the `train` and the
    # `evaluate` functions NEED to have `max_steps` and/or `steps` set and
    # cannot make use of the iterator's end-of-input exception, so we need
    # to do some math for that here.
    num_samples = datasets.get_count(FLAGS.val_split)
    num_steps = num_samples // FLAGS.get_flag_value("eval_batch_size",
                                                    FLAGS.batch_size)
    tf.logging.info("val_steps: %d", num_steps)

    for checkpoint in checkpoints_iterator(
        estimator.model_dir, timeout=FLAGS.eval_timeout_mins * 60):

      result_dict_val = estimator.evaluate(
          checkpoint_path=checkpoint, input_fn=data_fn, steps=num_steps)

      hub_exporter = hub.LatestModuleExporter("hub", serving_input_fn)
      hub_exporter.export(
          estimator,
          os.path.join(checkpoint_dir, "export/hub"),
          checkpoint)
      # This is here instead of using the above `checkpoints_iterator`'s
      # `timeout_fn` param, because that would wait forever on failed
      # trainers which will never create this file.
      if tf.gfile.Exists(os.path.join(FLAGS.workdir, "TRAINING_IS_DONE")):
        break

    # Evaluates the latest checkpoint on validation set.
    result_dict_val = estimator.evaluate(input_fn=data_fn, steps=num_steps)
    tf.logging.info(result_dict_val)

    # Optionally evaluates the latest checkpoint on test set.
    if FLAGS.test_split:
      data_fn = functools.partial(
          datasets.get_data,
          split_name=FLAGS.test_split,
          preprocessing=FLAGS.get_flag_value("preprocessing_eval",
                                             FLAGS.preprocessing),
          is_training=False,
          shuffle=False,
          num_epochs=1,
          drop_remainder=True)
      num_samples = datasets.get_count(FLAGS.test_split)
      num_steps = num_samples // FLAGS.get_flag_value("eval_batch_size",
                                                      FLAGS.batch_size)
      result_dict_test = estimator.evaluate(input_fn=data_fn, steps=num_steps)
      tf.logging.info(result_dict_test)
    return result_dict_val

  else:
    train_data_fn = functools.partial(
        datasets.get_data,
        split_name=FLAGS.train_split,
        preprocessing=FLAGS.preprocessing,
        is_training=True,
        num_epochs=None,  # read data indefenitely for training
        drop_remainder=True)

    # We compute the number of steps and make use of Estimator's max_steps
    # arguments instead of relying on the Dataset's iterator to run out after
    # a number of epochs so that we can use "fractional" epochs, which are
    # used by regression tests. (And because TPUEstimator needs it anyways.)
    num_samples = datasets.get_count(FLAGS.train_split)
    if FLAGS.num_supervised_examples:
      num_samples = FLAGS.num_supervised_examples
    # Depending on whether we drop the last batch each epoch or only at the
    # ver end, this should be ordered differently for rounding.
    updates_per_epoch = num_samples // FLAGS.batch_size
    epochs = utils.str2intlist(FLAGS.schedule, strict_int=False)[-1]
    num_steps = int(math.ceil(epochs * updates_per_epoch))
    tf.logging.info("train_steps: %d", num_steps)

    return estimator.train(
        train_data_fn,
        max_steps=num_steps)


def serving_input_fn():  # pylint: disable=missing-docstring
  """A serving input fn."""
  input_shape = utils.str2intlist(FLAGS.serving_input_shape)
  image_features = {
      FLAGS.serving_input_key:
          tf.placeholder(dtype=tf.float32, shape=input_shape)}
  return tf.estimator.export.ServingInputReceiver(
      features=image_features, receiver_tensors=image_features)
