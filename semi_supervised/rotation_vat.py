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

"""Jointly learns orientation of rotated images and their class, + VAT loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

import datasets
from semi_supervised import utils as ss_utils
from semi_supervised import vat_utils
import trainer
import utils
import tpu_ops

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
      }, normalization_fn=tpu_ops.cross_replica_batch_norm)
      return end_points, end_points['classes']
    return trainer.make_estimator(
        mode, predict_fn=model_building_fn,
        predict_input=data['image'],
        polyak_averaging=FLAGS.get_flag_value('polyak_averaging', False))

  # In all other cases, we are in train/eval mode.

  # Potentially flatten the rotation "R" dimension (B,R,H,W,C) into the batch
  # "B" dimension so we get (BR,H,W,C)
  images_unsup = data[0]['image']
  images_unsup = utils.into_batch_dim(images_unsup)

  # For the supervised branch, we also apply rotation on them.
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
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        outputs={
            'rotations': num_angles,
            'classes': datasets.get_auxiliary_num_classes(),
        }, normalization_fn=tpu_ops.cross_replica_batch_norm)

  # Compute virtual adversarial perturbation
  # =====

  def classification_net_fn(x):  # pylint: disable=missing-docstring
    with tf.variable_scope('module', reuse=True):
      end_points_x = ss_utils.apply_model_semi(
          x, None,
          is_training=(mode == tf.estimator.ModeKeys.TRAIN),
          outputs={'classes': datasets.get_auxiliary_num_classes()},
          # Don't update batch norm stats as we're running this on perturbed
          # (corrupted) inputs. Setting decay=1 is what does the trick.
          normalization_fn=functools.partial(
              tpu_ops.cross_replica_batch_norm, decay=1.0))
      return end_points_x['classes']

  vat_eps = FLAGS.get_flag_value('vat_eps', 1.0)
  vat_num_power_method_iters = FLAGS.get_flag_value('vat_num_power_method_iters', 1)
  vat_perturbation = vat_utils.virtual_adversarial_perturbation_direction(
      images_unsup,
      end_points['classes_unsup'],
      net=classification_net_fn,
      num_power_method_iters=vat_num_power_method_iters,
  ) * vat_eps

  loss_vat = tf.reduce_mean(vat_utils.kl_divergence_from_logits(
      classification_net_fn(images_unsup + vat_perturbation),
      tf.stop_gradient(end_points['classes_unsup'])))

  # Compute the rotation self-supervision loss.
  # =====

  # Compute the rotation loss on the unsupervised images.
  labels_rot_unsup = tf.reshape(data[0]['label'], [-1])
  loss_rot_unsup = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=end_points['rotations_unsup'], labels=labels_rot_unsup)
  loss_rot = tf.reduce_mean(loss_rot_unsup)

  # And on the supervised images too.
  labels_rot_sup = tf.reshape(data[1]['label'], [-1])
  loss_rot_sup = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=end_points['rotations_sup'], labels=labels_rot_sup)
  loss_rot_sup = tf.reduce_mean(loss_rot_sup)

  loss_rot = 0.5*loss_rot + 0.5*loss_rot_sup

  # Compute the classification loss on supervised images.
  # =====
  logits_class = end_points['classes_sup']

  # Replicate the supervised label for each rotated version.
  labels_class_repeat = tf.tile(labels_class[:, None], [1, num_angles])
  labels_class_repeat = tf.reshape(labels_class_repeat, [-1])

  loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_class_repeat, logits=logits_class)
  loss_class = tf.reduce_mean(loss_class)

  # Compute the EntMin regularization loss.
  # =====
  logits_unsup = end_points['classes_unsup']
  conditional_ent = -tf.reduce_sum(
      tf.nn.log_softmax(logits_unsup) * tf.nn.softmax(logits_unsup), axis=-1)
  loss_entmin = tf.reduce_mean(conditional_ent)

  # Combine losses and define metrics.
  # =====

  # Combine the two losses as a weighted average.
  wc = FLAGS.get_flag_value('sup_weight', 0.3)
  assert 0.0 <= wc <= 1.0, 'Loss weight should be in [0, 1] range.'
  wv = FLAGS.get_flag_value('vat_weight', 0.3)
  assert 0.0 <= wv <= 1.0, 'Loss weight should be in [0, 1] range.'

  # Combine VAT, classification and rotation loss as a weighted average, then
  # add weighted conditional entropy loss.
  loss = ((1.0 - wc - wv) * loss_rot + wc * loss_class + wv * loss_vat +
          FLAGS.entmin_factor * loss_entmin)

  train_scalar_summaries = {
      'vat_eps': vat_eps,
      'vat_weight': wv,
      'vat_num_power_method_iters': vat_num_power_method_iters,
      'loss_class': loss_class,
      'loss_class_weighted': wc * loss_class,
      'class_weight': wc,
      'loss_vat': loss_vat,
      'loss_vat_weighted': wv * loss_vat,
      'rot_weight': 1.0 - wc - wv,
      'loss_rot': loss_rot,
      'loss_rot_weighted': (1.0 - wc - wv) * loss_rot,
      'loss_entmin': loss_entmin,
      'loss_entmin_weighted': FLAGS.entmin_factor * loss_entmin
  }

  # For evaluation, we want to see the result of using only the un-rotated, and
  # also the average of four rotated class-predictions.
  logits_class = utils.split_batch_dim(logits_class, [-1, num_angles])
  logits_class_orig = logits_class[:, 0]
  logits_class_avg = tf.reduce_mean(logits_class, axis=1)

  eval_metrics = (
      lambda labels_rot_unsup, logits_rot_unsup, labels_class, logits_class_orig, logits_class_avg: {  # pylint: disable=g-long-lambda,line-too-long
          'rotation top1 accuracy':
              utils.top_k_accuracy(1, labels_rot_unsup, logits_rot_unsup),
          'classification/unrotated top1 accuracy':
              utils.top_k_accuracy(1, labels_class, logits_class_orig),
          'classification/unrotated top5 accuracy':
              utils.top_k_accuracy(5, labels_class, logits_class_orig),
          'classification/rot_avg top1 accuracy':
              utils.top_k_accuracy(1, labels_class, logits_class_avg),
          'classification/rot_avg top5 accuracy':
              utils.top_k_accuracy(5, labels_class, logits_class_avg),
      },
      [
          labels_rot_unsup, end_points['rotations_unsup'],
          labels_class, logits_class_orig, logits_class_avg,
      ])

  return trainer.make_estimator(
      mode, loss, eval_metrics, train_scalar_summaries=train_scalar_summaries,
      polyak_averaging=FLAGS.get_flag_value('polyak_averaging', False))
