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

"""Train only on the labelled images in the semi-supervised setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import datasets
import semi_supervised.utils as ss_utils
import trainer
import utils


def model_fn(data, mode):
  """Produces a loss for the rotation task with semi-supervision.

  Args:
    data: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  # In this mode (called once at the end of training), we create the tf.Hub
  # module in order to export the model, and use that to do one last prediction.
  if mode == tf.estimator.ModeKeys.PREDICT:
    # This defines a function called by the hub module to create the model.
    def model_building_fn(img, is_training):
      # This is an example of calling `apply_model_semi` with only one of the
      # inputs provided. The outputs will simply use the given names:
      end_points = ss_utils.apply_model_semi(img, None, is_training, outputs={
          'classes': datasets.get_auxiliary_num_classes(),
      })
      return end_points, end_points['classes']
    return trainer.make_estimator(
        mode, predict_fn=model_building_fn, predict_input=data['image'])

  # In all other cases, we are in train/eval mode.
  # Note that here we only use data[1], i.e. the part with labels.

  # Forward them both through the model. The scope is needed for tf.Hub export.
  with tf.variable_scope('module'):
    # Here, we pass both inputs to `apply_model_semi`, and so we now get
    # outputs corresponding to each in `end_points` as "rotations_unsup" and
    # similar, which we will use below.
    end_points = ss_utils.apply_model_semi(
        None, data[1]['image'],
        is_training=mode == tf.estimator.ModeKeys.TRAIN,
        outputs={'classes': datasets.get_auxiliary_num_classes()})

  # Compute the classification loss on supervised images.

  logits_class = end_points['classes']
  labels_class = data[1]['label']
  loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_class, logits=logits_class)
  loss = tf.reduce_mean(loss_class)

  # Define metrics.

  eval_metrics = (
      lambda labels_class, logits_class: {  # pylint: disable=g-long-lambda
          'top1 accuracy': utils.top_k_accuracy(1, labels_class, logits_class),
          'top5 accuracy': utils.top_k_accuracy(5, labels_class, logits_class),
      }, [labels_class, logits_class])

  return trainer.make_estimator(mode, loss, eval_metrics)
