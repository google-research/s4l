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

"""Produces ratations for input images.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import datasets
import semi_supervised.utils as ss_utils
import trainer
import utils

FLAGS = tf.flags.FLAGS


def model_fn(data, mode):
  """Produces a loss for the rotation task with semi-supervision.

  Args:
    data: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  num_angles = 4

  # In this mode (called once at the end of training), we create the tf.Hub
  # module in order to export the model, and use that to do one last prediction.
  if mode == tf.estimator.ModeKeys.PREDICT:
    # This defines a function called by the hub module to create the model.
    def model_building_fn(img, is_training):
      # This is an example of calling `apply_model_semi` with only one of the
      # inputs provided. The outputs will simply use the given names:
      end_points = ss_utils.apply_model_semi(img, None, is_training, outputs={
          'rotations': num_angles,
          'classes': datasets.get_auxiliary_num_classes(),
      })
      return end_points, end_points['classes']
    return trainer.make_estimator(
        mode, predict_fn=model_building_fn,
        predict_input=data['image'],
        polyak_averaging=FLAGS.get_flag_value('polyak_averaging', False))

  # In all other cases, we are in train/eval mode.

  images_unsup = None
  if FLAGS.rot_loss_unsup:
    # Potentially flatten the rotation "R" dimension (B,R,H,W,C) into the batch
    # "B" dimension so we get (BR,H,W,C)
    images_unsup = data[0]['image']
    images_unsup = utils.into_batch_dim(images_unsup)

  images_sup = data[1]['image']
  images_sup = utils.into_batch_dim(images_sup)
  labels_class = data[1]['copy_label']

  # Forward them both through the model. The scope is needed for tf.Hub export.
  with tf.variable_scope('module'):
    # Here, we pass both inputs to `apply_model_semi`, and so we now get
    # outputs corresponding to each in `end_points` as "rotations_unsup" and
    # similar, which we will use below.
    end_points = ss_utils.apply_model_semi(
        images_unsup, images_sup,
        is_training=mode == tf.estimator.ModeKeys.TRAIN,
        outputs={
            'rotations': num_angles,
            'classes': datasets.get_auxiliary_num_classes(),
        })

  # Compute the rotation self-supervision loss.
  # =====

  losses_rot = []

  # Compute the rotation loss on the unsupervised images.
  if FLAGS.rot_loss_unsup:
    labels_rot_unsup = tf.reshape(data[0]['label'], [-1])
    loss_rot_unsup = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=end_points['rotations_unsup'], labels=labels_rot_unsup)
    losses_rot.append(tf.reduce_mean(loss_rot_unsup))

  # And on the supervised images too.
  if FLAGS.rot_loss_sup:
    labels_rot_sup = tf.reshape(data[1]['label'], [-1])
    loss_rot_sup = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=end_points['rotations_sup'], labels=labels_rot_sup)
    losses_rot.append(tf.reduce_mean(loss_rot_sup))

  loss_rot = tf.reduce_mean(losses_rot) if losses_rot else 0.0

  # Compute the classification loss on supervised images.
  # =====
  logits_class = end_points['classes_sup']

  # Replicate the supervised label for each rotated version.
  labels_class_repeat = tf.tile(labels_class[:, None], [1, num_angles])
  labels_class_repeat = tf.reshape(labels_class_repeat, [-1])

  loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_class_repeat, logits=logits_class)
  loss_class = tf.reduce_mean(loss_class)

  # Combine losses and define metrics.
  # =====
  w = FLAGS.rot_loss_weight
  loss = loss_class + w * loss_rot

  # At eval time, we compute accuracy of both the unrotated image,
  # and the average prediction across all four rotations
  logits_class = utils.split_batch_dim(logits_class, [-1, num_angles])
  logits_class_orig = logits_class[:, 0]
  logits_class_avg = tf.reduce_mean(logits_class, axis=1)

  eval_metrics = (
      lambda labels_class, logits_class_orig, logits_class_avg: {  # pylint: disable=g-long-lambda
          'classification/unrotated top1 accuracy':
              utils.top_k_accuracy(1, labels_class, logits_class_orig),
          'classification/unrotated top5 accuracy':
              utils.top_k_accuracy(5, labels_class, logits_class_orig),
          'classification/rot_avg top1 accuracy':
              utils.top_k_accuracy(1, labels_class, logits_class_avg),
          'classification/rot_avg top5 accuracy':
              utils.top_k_accuracy(5, labels_class, logits_class_avg)
      }, [labels_class, logits_class_orig, logits_class_avg])

  return trainer.make_estimator(
      mode, loss, eval_metrics,
      polyak_averaging=FLAGS.get_flag_value('polyak_averaging', False))
