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

"""Common utilities for the labeling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

import preprocess as preprocess_lib


FLAGS = flags.FLAGS

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def into_batch_dim(x, keep_last_dims=-3):
  """Turns (B,M,...,H,W,C) into (BM...,H,W,C) if `keep_last_dims` is -3."""
  last_dims = x.get_shape().as_list()[keep_last_dims:]
  return tf.reshape(x, shape=[-1] + last_dims)


def split_batch_dim(x, split_dims):
  """Turns (BMN,H,...) into (B,M,N,H,...) if `split_dims` is [-1, M, N]."""
  last_dims = x.get_shape().as_list()[1:]
  logging.info("split_dims: %s", split_dims)
  logging.info("last_dims: %s", last_dims)
  return tf.reshape(x, list(split_dims) + last_dims)


def make_transform(mod, model_signature, representation_name):
  """Function to transfrom the input to a dict of results."""

  def transform_batch(input_tuple):
    """Transform a batch of input batch to a dictionary of embeddings."""
    num_crops = input_tuple["image_normalized"].get_shape().as_list()[1]
    num_rotations = input_tuple["image_normalized"].get_shape().as_list()[2]
    images = into_batch_dim(input_tuple["image_normalized"])
    outputs = mod(inputs=images, signature=model_signature, as_dict=True)
    outputs = outputs[representation_name]
    outputs = split_batch_dim(outputs, [-1, num_crops, num_rotations])
    logging.info("outputs after split batch dim: %s", outputs)
    logits = outputs[:, 0, 0, :]

    logits_avg_croprot = tf.reduce_mean(outputs, axis=[1, 2])
    return {
        "logits": logits,
        "outputs": outputs,
        "logits_avg_croprot": logits_avg_croprot,
        "filename": input_tuple["filename"],
        "original_label": input_tuple["original_label"],
        "jpeg_str": input_tuple["jpeg_str"],
    }

  return transform_batch


def parse_imagenet(image_data,
                   image_key="image",
                   file_name_key="file_name",
                   label_key="label"):
  """Parses imagenet record to input of sklearn pipeline."""
  value = tf.parse_single_example(
      image_data,
      features={
          image_key: tf.FixedLenFeature([], tf.string),
          file_name_key: tf.FixedLenFeature([], tf.string),
          label_key: tf.FixedLenFeature([], tf.int64)
      })
  jpeg_str = value[image_key]
  image = tf.image.decode_jpeg(jpeg_str, channels=3)
  data = {
      "image": image,
      "filename": value[file_name_key],
      "label": value[label_key]
  }
  data = preprocess_lib.get_resize_small(256)(data)
  data = preprocess_lib.get_multi_crop((224, 224))(data)
  data = preprocess_lib.get_rotate_preprocess(create_labels=False)(data)
  image_normalized = preprocess_lib.get_value_range_preprocess(-1,
                                                               1)(data)["image"]

  return {
      "image_normalized": image_normalized,
      "filename": data["filename"],
      "original_label": data["label"],
      "jpeg_str": jpeg_str,
  }


def write_imagenet_labels(writer,
                          results,
                          target_label,
                          label_offset=0,
                          image_key="image",
                          file_name_key="file_name",
                          label_key="label",
                          original_label_key="original_label",
                          label_flag_key="label_flag",
                          logits_key="logits"):
  """Writes sample from next node to writer with the provided label."""
  filenames = results["filenames"]
  jpeg_str = results["jpeg_str"]

  if target_label == "predicted_labels":
    # ImageNet class label offset.
    labels = results["predicted_labels"] + label_offset
  elif target_label == "original_labels":
    original_labels = results["original_labels"]
    flags = results["flags"]
    labels = []
    for i in range(len(flags)):
      labels.append(original_labels[i] if flags[i] else -1)
  elif target_label == "logits":
    labels = results["logits"]
  elif target_label == "all":
    predicted_labels = results["predicted_labels"] + label_offset
    predicted_labels_avg_croprot = results[
        "predicted_labels_avg_croprot"] + label_offset

    original_labels = results["original_labels"]
    flags = results["flags"]
    logits = results["logits"]
    logits_avg_croprot = results["logits_avg_croprot"]
  else:
    raise NotImplementedError("Unsupported mode: %s" % target_label)

  for i in range(len(filenames)):
    jpeg = _bytes_feature(jpeg_str[i])
    if target_label == "all":
      predicted_label = _int64_feature(int(predicted_labels[i]))
      original_label = _int64_feature(original_labels[i])
      label_flag = _int64_feature(flags[i])
      logit = _float_feature(logits[i])
      logit_avg_croprot = _float_feature(logits_avg_croprot[i])

      predicted_label_avg_croprot = _int64_feature(
          int(predicted_labels_avg_croprot[i]))

    elif target_label == "logits":
      label_i = _float_feature(labels[i])
    elif (target_label == "predicted_labels" or
          target_label == "original_labels"):
      label_i = _int64_feature(int(labels[i]))

    if target_label == "all":
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  image_key: jpeg,
                  file_name_key: _bytes_feature(filenames[i]),
                  label_key: predicted_label,
                  original_label_key: original_label,
                  label_flag_key: label_flag,
                  logits_key: logit,
                  "logit_avg_croprot": logit_avg_croprot,
                  "predicted_labels_avg_croprot": predicted_label_avg_croprot,
              }))
    else:
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  image_key: jpeg,
                  file_name_key: _bytes_feature(filenames[i]),
                  label_key: label_i,
              }))
    writer.write(example.SerializeToString())


def create_labels(input_tfrecord_path,
                  output_tfrecord_path,
                  dataset_preprocess_fn,
                  embedding_fn,
                  label_fn,
                  write_fn=None,
                  batch_size=64,
                  parallel_calls=1):
  """Creates a new set of labels for a single chunk.

  Args:
    input_tfrecord_path: String with input TF Record file.
    output_tfrecord_path: String with input TF Record file.
    dataset_preprocess_fn: Preprocessing function applied to dataset.
    embedding_fn: Embedding function applied to the dataset tensor.
    label_fn: Label function applied to the (after sess.run).
    write_fn: Function to write TF Record to TF Record writer.
    batch_size: Optional integer with batch_size.
  """
  tf.logging.info("Input: {}\nOutput: {}".format(
      input_tfrecord_path, output_tfrecord_path))
  if write_fn is None:
    write_fn = write_imagenet
  
  if FLAGS.tpu_name:
    cluster = TPUClusterResolver(tpu=[FLAGS.tpu_name])
  else:
    cluster = None
  config = tf.contrib.tpu.RunConfig(cluster=cluster)

  # Load the data in the chunk.
  input_dataset = tf.data.TFRecordDataset(input_tfrecord_path)
  input_dataset = input_dataset.map(
          dataset_preprocess_fn, parallel_calls)
  input_dataset = input_dataset.batch(batch_size)
  next_node = input_dataset.make_one_shot_iterator().get_next()
  embedding = embedding_fn(next_node)
  with tf.Session(cluster.get_master(), config=config.session_config) as sess:
    with tf.python_io.TFRecordWriter(output_tfrecord_path) as writer:
      sess.run(tf.global_variables_initializer())
      while True:
        try:
          embedded = sess.run(embedding)
          results = label_fn(embedded)
          write_fn(writer, results)
        except tf.errors.OutOfRangeError:
          break


def write_imagenet(writer, labels):
  """Writes sample from next node to writer with the provided label."""
  for label in labels:
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": _bytes_feature(""),
                "image/class/label": _int64_feature(int(label)),
            }))
    writer.write(example.SerializeToString())
