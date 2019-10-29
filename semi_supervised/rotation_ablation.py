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

import google3.learning.brain.research.dune.experimental.representation.datasets as datasets
import google3.learning.brain.research.dune.experimental.representation.semi_supervised.utils as ss_utils
import google3.learning.brain.research.dune.experimental.representation.trainer as trainer
import google3.learning.brain.research.dune.experimental.representation.utils as utils

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
    return trainer.make_estimator(mode, predict_fn=model_building_fn,
                                  predict_input=data['image'])

  # In all other cases, we are in train/eval mode.

  unsup_has_rot_loss = FLAGS.config.get('unsup_has_rot_loss', True)
  if unsup_has_rot_loss:
    # Potentially flatten the rotation "R" dimension (B,R,H,W,C) into the batch
    # "B" dimension so we get (BR,H,W,C)
    images_unsup = data[0]['image']
    images_unsup = utils.into_batch_dim(images_unsup)
  else:
    images_unsup = None

  # For the supervised branch, it gets complicated, because in a few cases we
  # only need to send un-rotated images through the network, and we'd like to do
  # that in these cases for 1) performance reasons and 2) BN stats reasons.
  images_sup = data[1]['image']
  labels_class = data[1]['copy_label']

  # Just a shortcut.
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  # Decide which images to send through the net, and send them.
  # =====

  # In two cases we only need the un-rotated supervised images.
  if ((FLAGS.config.sup_class_loss == 'single' and is_training
       and not FLAGS.config.sup_has_rot_loss)
      or
      (FLAGS.config.sup_class_eval == 'single' and not is_training)):
    images_sup = images_sup[:, 0]
  else:
    # Move rotation into batch dimension.
    images_sup = utils.into_batch_dim(images_sup)

  # Forward them both through the model. The scope is needed for tf.Hub export.
  with tf.variable_scope('module'):
    # Here, we pass both inputs to `apply_model_semi`, and so we now get
    # outputs corresponding to each in `end_points` as "rotations_unsup" and
    # similar, which we will use below.
    end_points = ss_utils.apply_model_semi(
        images_unsup, images_sup, is_training=is_training,
        outputs={
            'rotations': num_angles,
            'classes': datasets.get_auxiliary_num_classes(),
        })

  # Compute the rotation self-supervision loss.
  # =====
  losses_rot = []

  # Optionally enable the unsupervised data batch for the rotation loss.
  if unsup_has_rot_loss:
    # Compute the rotation loss on the unsupervised images.
    labels_rot_unsup = tf.reshape(data[0]['label'], [-1])
    loss_rot_unsup = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=end_points['rotations_unsup'], labels=labels_rot_unsup)
    losses_rot.append(tf.reduce_mean(loss_rot_unsup))

  # And optionally on the supervised images too.
  if is_training and FLAGS.config.sup_has_rot_loss:
    loss_rot_sup = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=end_points['rotations_sup'],
        labels=tf.reshape(data[1]['label'], [-1]))
    losses_rot.append(tf.reduce_mean(loss_rot_sup))

  loss_rot = tf.reduce_mean(losses_rot) if losses_rot else 0.0

  # Compute the classification loss on supervised images.
  # =====

  logits_class = end_points['classes_sup']

  if FLAGS.config.sup_class_loss == 'single' and FLAGS.config.sup_has_rot_loss and is_training:
    logits_class = utils.split_batch_dim(logits_class, [-1, num_angles])[:, 0]

  # In only one case, we replicate the supervised label for each rotated
  # version, otherwise, we keep them as is (1 label for num_angle images).
  if is_training and FLAGS.config.sup_class_loss == 'all':
    labels_class = tf.tile(labels_class[:, None], [1, num_angles])
    labels_class = tf.reshape(labels_class, [-1])

  # Possibly average the logits or probabilities of the rotated versions.
  if ((FLAGS.config.sup_class_loss == 'avg' and is_training) or
      (FLAGS.config.sup_class_eval == 'avg' and not is_training)):
    if FLAGS.config.sup_avg_probs:
      logits_class = tf.nn.softmax(logits_class)
    logits_class = utils.split_batch_dim(logits_class, [-1, num_angles])
    logits_class = tf.reduce_mean(logits_class, axis=1)
    if FLAGS.config.sup_avg_probs:
      logits_class = tf.log(logits_class)

  loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_class, logits=logits_class)
  loss_class = tf.reduce_mean(loss_class)

  # Combine losses and define metrics.
  # =====

  # Combine the two losses as a weighted average.
  wc = FLAGS.config.get('sup_weight', 0.5)
  assert 0.0 <= wc <= 1.0, 'Loss weight should be in [0, 1] range.'

  loss = (1.0 - wc) * loss_rot + wc * loss_class

  eval_metrics = (
      lambda labels_class, logits_class: {  # pylint: disable=g-long-lambda
          'classification top1 accuracy':
              utils.top_k_accuracy(1, labels_class, logits_class),
          'classification top5 accuracy':
              utils.top_k_accuracy(5, labels_class, logits_class)
      },
      [
          labels_class, logits_class
      ])

  return trainer.make_estimator(mode, loss, eval_metrics)
