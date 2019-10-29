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

"""Util functions for representation learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import collections
import csv
import os
import re

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib.layers import max_pool2d, avg_pool2d, l2_regularizer
from tensorflow.contrib.tpu.python.tpu.tpu_function import get_tpu_context

import tpu_ops

TEST_FAIL_MAGIC = "QUALITY_FAILED"
TEST_PASS_MAGIC = "QUALITY_PASSED"


def linear(inputs, num_outputs, name, reuse=tf.AUTO_REUSE, weight_decay="flag"):
  """A linear layer on the inputs."""
  if weight_decay == "flag":
    weight_decay = flags.FLAGS.weight_decay

  kernel_regularizer = l2_regularizer(scale=weight_decay)
  logits = tf.layers.conv2d(
      inputs,
      filters=num_outputs,
      kernel_size=1,
      kernel_regularizer=kernel_regularizer,
      name=name,
      reuse=reuse)

  return tf.squeeze(logits, [1, 2])


def top_k_accuracy(k, labels, logits):
  """Builds a tf.metric for the top-k accuracy between labels and logits."""
  in_top_k = tf.nn.in_top_k(predictions=logits, targets=labels, k=k)
  return tf.metrics.mean(tf.cast(in_top_k, tf.float32))


def into_batch_dim(x, keep_last_dims=-3):
  """Turns (B,M,...,H,W,C) into (BM...,H,W,C) if `keep_last_dims` is -3."""
  last_dims = x.get_shape().as_list()[keep_last_dims:]
  return tf.reshape(x, shape=[-1] + last_dims)


def split_batch_dim(x, split_dims):
  """Turns (BMN,H,...) into (B,M,N,H,...) if `split_dims` is [-1, M, N]."""
  last_dims = x.get_shape().as_list()[1:]
  return tf.reshape(x, list(split_dims) + last_dims)


def repeat(x, times):
  """Exactly like np.repeat."""
  return tf.reshape(tf.tile(tf.expand_dims(x, -1), [1, times]), [-1])


def get_representation_dict(tensor_dict):
  rep_dict = {}
  for name, tensor in tensor_dict.items():
    rep_dict["representation_" + name] = tensor
  return rep_dict


def assert_not_in_graph(tensor_name, graph=None):
  # Put get_default_graph() to the function instead of the parameter. It cannot
  # be called if the graph is not initialized.
  if graph is None:
    graph = tf.get_default_graph()
  tensor_names = [
      tensor.name for tensor in graph.as_graph_def().node
  ]

  assert tensor_name not in tensor_names, "%s already exists." % tensor_name


def name_tensor(tensor, tensor_name):
  assert_not_in_graph(tensor_name)
  return tf.identity(tensor, name=tensor_name)


def import_graph(checkpoint_dir):
  """Imports the tf graph from latest checkpoint in checkpoint_dir."""
  checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  tf.train.import_meta_graph(checkpoint + ".meta", clear_devices=True)
  return tf.get_default_graph()


def check_quality(score, output_dir, min_value=None, max_value=None):
  """Checks the metric score, outputs magic file accordingly.

  Args:
     score: a float value represents evaluation metric value.
     output_dir: a string output directory for the magic file.
     min_value: a float value for the min of metric value.
     max_value: a float value for the max of metric value.

  Returns:
    Name of the magic-file that was created (i.e. result of test.)
  """
  assert min_value or max_value, "min_value and max_value are not set"
  if min_value and max_value:
    assert min_value <= max_value
  message = ""
  if min_value and score < min_value:
    message += "too low: %.2f < %.2f " % (score, min_value)
  if max_value and score > max_value:
    message += "too high: %.2f > %.2f " % (score, max_value)
  magic_file = TEST_FAIL_MAGIC if message else TEST_PASS_MAGIC

  with tf.gfile.Open(os.path.join(output_dir, magic_file), "w") as f:
    f.write(message)

  return magic_file


def append_multiple_rows_to_csv(dictionaries, csv_path):
  """Writes multiples rows to csv file from a list of dictionaries.

  Args:
    dictionaries: a list of dictionaries, mapping from csv header to value.
    csv_path: path to the result csv file.
  """

  # By default csv file was saved as %rs=6.3 in cns. It is finalized and
  # cannot be appended. We set %r=3 replication explicitly.
  # CNS file replication and encoding:
  csv_path = csv_path + "%r=3"

  keys = set([])
  for d in dictionaries:
    keys.update(d.keys())

  if not tf.gfile.Exists(csv_path):
    with tf.gfile.Open(csv_path, "w") as f:
      writer = csv.DictWriter(f, sorted(keys))
      writer.writeheader()
      f.flush()

  with tf.gfile.Open(csv_path, "a") as f:
    writer = csv.DictWriter(f, sorted(keys))
    writer.writerows(dictionaries)
    f.flush()


def concat_dicts(dict_list):
  """Given a list of dicts merges them into a single dict.

  This function takes a list of dictionaries as an input and then merges all
  these dictionaries into a single dictionary by concatenating the values
  (along the first axis) that correspond to the same key.

  Args:
    dict_list: list of dictionaries

  Returns:
    d: merged dictionary
  """
  d = collections.defaultdict(list)
  for e in dict_list:
    for k, v in e.items():
      d[k].append(v)
  for k in d:
    d[k] = tf.concat(d[k], axis=0)
  return d


def str2intlist(s, repeats_if_single=None, strict_int=True):
  """Parse a config's "1,2,3"-style string into a list of ints.

  Also handles it gracefully if `s` is already an integer, or is already a list
  of integer-convertible strings or integers.

  Args:
    s: The string to be parsed, or possibly already an (list of) int(s).
    repeats_if_single: If s is already an int or is a single element list,
                       repeat it this many times to create the list.
    strict_int: if True, fail when numbers are not integers.
      But if this is False, also attempt to convert to floats!

  Returns:
    A list of integers based on `s`.
  """
  def to_int_or_float(s):
    if strict_int:
      return int(s)
    else:
      try:
        return int(s)
      except ValueError:
        return float(s)

  if isinstance(s, int):
    result = [s]
  elif isinstance(s, (list, tuple)):
    result = [to_int_or_float(i) for i in s]
  else:
    result = [to_int_or_float(i.strip()) if i != "None" else None
              for i in s.split(",")]
  if repeats_if_single is not None and len(result) == 1:
    result *= repeats_if_single
  return result


def tf_apply_to_image_or_images(fn, image_or_images, **map_kw):
  """Applies a function to a single image or each image in a batch of them.

  Args:
    fn: the function to apply, receives an image, returns an image.
    image_or_images: Either a single image, or a batch of images.
    **map_kw: Arguments passed through to tf.map_fn if called.

  Returns:
    The result of applying the function to the image or batch of images.

  Raises:
    ValueError: if the input is not of rank 3 or 4.
  """
  static_rank = len(image_or_images.get_shape().as_list())
  if static_rank == 3:  # A single image: HWC
    return fn(image_or_images)
  elif static_rank == 4:  # A batch of images: BHWC
    return tf.map_fn(fn, image_or_images, **map_kw)
  elif static_rank > 4:  # A batch of images: ...HWC
    input_shape = tf.shape(image_or_images)
    h, w, c = image_or_images.get_shape().as_list()[-3:]
    image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
    image_or_images = tf.map_fn(fn, image_or_images, **map_kw)
    return tf.reshape(image_or_images, input_shape)
  else:
    raise ValueError("Unsupported image rank: %d" % static_rank)


def tf_apply_with_probability(p, fn, x):
  """Apply function `fn` to input `x` randomly `p` percent of the time."""
  return tf.cond(
      tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), p),
      lambda: fn(x),
      lambda: x)


def expand_glob(glob_patterns):
  checkpoints = []
  for pattern in glob_patterns:
    checkpoints.extend(tf.gfile.Glob(pattern))
  assert checkpoints, "There are no checkpoints in " + str(glob_patterns)
  return checkpoints


def get_latest_hub_per_task(hub_module_paths):
  """Get latest hub module for each task.

  The hub module path should match format ".*/hub/[0-9]*/module/.*".
  Example usage:
  get_latest_hub_per_task(expand_glob(["/cns/el-d/home/dune/representation/"
                                       "xzhai/1899361/*/export/hub/*/module/"]))
  returns 4 latest hub module from 4 tasks respectivley.

  Args:
    hub_module_paths: a list of hub module paths.

  Returns:
    A list of latest hub modules for each task.

  """
  task_to_path = {}
  for path in hub_module_paths:
    task_name, module_name = path.split("/hub/")
    timestamp = int(re.findall(r"([0-9]*)/module", module_name)[0])
    current_path = task_to_path.get(task_name, "0/module")
    current_timestamp = int(re.findall(r"([0-9]*)/module", current_path)[0])
    if current_timestamp < timestamp:
      task_to_path[task_name] = path
  return sorted(task_to_path.values())


def get_schedule_from_config(schedule, steps_per_epoch):
  """Get the appropriate learning rate schedule from the config.

  Args:
    config: ConfigDict to get the schedule from.
    steps_per_epoch: Number of steps in each epoch (integer).
      Needed to convert epochs-based schedule to steps-based.
  Returns:
    A list of integers representing the learning rate schedule (in steps).

  Raises:
    ValueError if both or neither of config.schedule or config.schedule_steps
    are given in the ConfigDict.
  """
  if schedule is None:
    raise ValueError(
        "You must specify exactly one of config.schedule or "
        "config.schedule_steps.")
  elif schedule is not None:
    schedule = str2intlist(schedule, strict_int=False)
    schedule = [epoch * steps_per_epoch for epoch in schedule]

  if sorted(schedule) != schedule:
    raise ValueError("Invalid schedule {!r}".format(schedule))

  return schedule
