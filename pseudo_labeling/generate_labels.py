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

# pylint:disable=line-too-long
r"""Script to generate labels from a trained SSSL model.

blaze run -c opt --config=dmtf_cuda \
learning/brain/research/dune/experimental/representation/tools/pseudo_labeling:generate_labels
-- \
--alsologtostderr \
--input_tf_record=/placer/prod/home/tensor2tensor/datasets/rs=6.3/\
v1/image_imagenet-dev-00000-of-00128 \
--output_tf_record=/tmp/image_imagenet-train-00000-of-01024 \
--hub_module=/cns/tp-d/home/dune/representation/xzhai/sssl-datasets/supervised_10_module/1552643616/module/ \
--labels_file=/cns/tp-d/home/xzhai/semisupervised/imagenet/label_map_count_512000_index_0 \
--target_label=predicted_labels \
--data_format=t2t \
--representation_name=classes

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm

import labeling_utils as utils

FLAGS = flags.FLAGS
flags.DEFINE_string("input_tf_record", None, "Path to input TFRecord file.")
flags.DEFINE_string("output_tf_record", None, "Path to output TFRecord file.")
flags.DEFINE_string("input_glob", None, "Path to input TFRecord files.")
flags.DEFINE_string("output_folder", None, "Dir to output TFRecord file.")
flags.DEFINE_string("hub_module", None, "Path to sssl hub module.")
flags.DEFINE_string("model_signature", "representation",
                    "Signature of sssl hub module representations.")
flags.DEFINE_string("representation_name", "logits_unsupervised_classification",
                    "Representation name of sssl hub module outputs.")
flags.DEFINE_integer("batch_size", 64, "Batch size for embedding computation.")
flags.DEFINE_string(
    "data_format", "tfds",
    "Dataset format, should be set one of {tfds, t2t}. The "
    "data format affects tf record column names in the "
    "input/output files, e.g. tfds use image but t2t use "
    "image/encoded for the encoded jpeg image.")
flags.DEFINE_string("labels_file", None, "Path to SSL label files.")
flags.DEFINE_string(
    "target_label", "predicted_labels",
    "Target labels to generate, should be one of "
    "{predicted_labels, logits, original_labels, all}")
flags.DEFINE_string("tpu_name", None, "Name of the TPU node to use.")
flags.DEFINE_integer("parallel_calls", 1, 
        "Number of parallel threads for loading dataset, "
        "set to 1 for determinized sequence.")


def main(unused_argv):
  # Check if the output file already exists. If this is the case, delete it as
  # this was a previous, partially completed job that was preempted.
  if (FLAGS.input_tf_record and
      FLAGS.output_tf_record) == (FLAGS.input_glob and FLAGS.output_folder):
    raise ValueError(
        "You must specify only one of 'file' or 'glob' in your config.")

  if (FLAGS.input_tf_record and FLAGS.output_tf_record):
    if tf.gfile.Exists(FLAGS.output_tf_record):
      tf.gfile.Remove(FLAGS.output_tf_record)

    # Create the containing folder if necessary.
    folder_path = os.path.dirname(FLAGS.output_tf_record)
    if not tf.gfile.Exists(folder_path):
      tf.gfile.MakeDirs(folder_path)
  else:
    if not tf.gfile.Exists(FLAGS.output_folder):
      tf.gfile.MakeDirs(FLAGS.output_folder)

  labels_list = None
  labels_file = FLAGS.labels_file
  with tf.gfile.Open(labels_file, "r") as f:
    labels_list = json.load(f)
    labels_list = labels_list["values"]

  def label_fn(embeddings):
    """Generate labels from embeddings."""
    predicted_labels = np.argmax(embeddings["logits"], axis=1)
    predicted_labels_avg_croprot = np.argmax(
        embeddings["logits_avg_croprot"], axis=1)
    example_flags = [
        filename in labels_list for filename in embeddings["filename"]
    ]

    return {
        "logits": embeddings["logits"],
        "logits_avg_croprot": embeddings["logits_avg_croprot"],
        "predicted_labels": predicted_labels,
        "predicted_labels_avg_croprot": predicted_labels_avg_croprot,
        "flags": example_flags,
        "filenames": embeddings["filename"],
        "original_labels": embeddings["original_label"],
        "jpeg_str": embeddings["jpeg_str"]
    }

  if FLAGS.data_format == "tfds":
    write_fn = functools.partial(
        utils.write_imagenet_labels, target_label=FLAGS.target_label)
    parser_fn = utils.parse_imagenet
  elif FLAGS.data_format == "t2t":
    write_fn = functools.partial(
        utils.write_imagenet_labels,
        target_label=FLAGS.target_label,
        label_offset=1,
        image_key="image/encoded",
        file_name_key="image/filename",
        label_key="image/class/label",
        original_label_key="image/class/original_label",
        label_flag_key="image/class/label_flag",
        logits_key="image/class/logits")
    parser_fn = functools.partial(
        utils.parse_imagenet,
        image_key="image/encoded",
        file_name_key="image/filename",
        label_key="image/class/label")
  else:
    raise ValueError("Unsupported data source %s" % FLAGS.data_format)
  # Do the actual conversion.
  with tf.Graph().as_default():
    # We need to wrap the transform function to extract the proper tensor.
    mod = hub.Module(FLAGS.hub_module)
    transform_fn = utils.make_transform(mod, FLAGS.model_signature,
                                        FLAGS.representation_name)

    def wrapped_transform_fn(input_tuple):
      return transform_fn(input_tuple)

    def convert(input_file, output_file):
      utils.create_labels(
          input_file,
          output_file,
          parser_fn,
          wrapped_transform_fn,
          label_fn,
          write_fn=write_fn,
          batch_size=FLAGS.batch_size,
          parallel_calls=FLAGS.parallel_calls)

    if (FLAGS.input_tf_record and FLAGS.output_tf_record):
      convert(FLAGS.input_tf_record, FLAGS.output_tf_record)
    else:
      for filename in tqdm.tqdm(tf.gfile.Glob(FLAGS.input_glob)):
        output_filename = os.path.join(FLAGS.output_folder,
                                       os.path.basename(filename))
        convert(filename, output_filename)


if __name__ == "__main__":
  app.run(main)
