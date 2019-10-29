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

"""Apply supervised loss + VAT + EntMin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

import datasets

import datasets
import semi_supervised.utils as ss_utils
import trainer
import utils
from semi_supervised import vat_utils

FLAGS = tf.flags.FLAGS


def model_fn(data, mode):
  """Produces a loss for the VAT semi-supervised task.

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
    return trainer.make_estimator(mode, predict_fn=model_building_fn,
                                  predict_input=data['image'])

  # In all other cases, we are in train/eval mode.

  images_unsup = data[0]['image']
  images_sup = data[1]['image']

  # Forward them both through the model. The scope is needed for tf.Hub export.
  with tf.variable_scope('module'):
    # Here, we pass both inputs to `apply_model_semi`, and so we now get
    # outputs corresponding to each in `end_points` as "classes_unsup" and
    # "classes_sup", which we can use below.
    end_points = ss_utils.apply_model_semi(
        images_unsup, images_sup,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        outputs={'classes': datasets.get_auxiliary_num_classes()})

  # Compute virtual adversarial perturbation
  def classification_net_fn(x):  # pylint: disable=missing-docstring
    with tf.variable_scope('module', reuse=True):
      end_points_x = ss_utils.apply_model_semi(
          x, None,
          is_training=(mode == tf.estimator.ModeKeys.TRAIN),
          outputs={'classes': datasets.get_auxiliary_num_classes()},
          # Don't update batch norm stats as we're running this on perturbed
          # (corrupted) inputs. Setting momentum = 1 is what does the trick.
          normalization_fn=functools.partial(
              tf.layers.batch_normalization, momentum=1.0))
      return end_points_x['classes']

  ## Compute VAT perturbation
  vat_eps = FLAGS.get_flag_value('vat_eps', 1.0)
  vat_num_power_method_iters = FLAGS.get_flag_value('vat_num_power_method_iters', 1)

  if FLAGS.apply_vat_to_labeled:
    images_vat_baseline = tf.concat(
        (images_sup, images_unsup), axis=0)
    predictions_vat_baseline = tf.concat(
        (end_points['classes_sup'], end_points['classes_unsup']), axis=0)
  else:
    images_vat_baseline = images_unsup
    predictions_vat_baseline = end_points['classes_unsup']

  vat_perturbation = vat_utils.virtual_adversarial_perturbation_direction(
      images_vat_baseline,
      predictions_vat_baseline,
      net=classification_net_fn,
      num_power_method_iters=vat_num_power_method_iters,
  ) * vat_eps

  loss_vat = tf.reduce_mean(vat_utils.kl_divergence_from_logits(
      classification_net_fn(images_vat_baseline + vat_perturbation),
      tf.stop_gradient(predictions_vat_baseline)))

  ## Compute the classification loss on supervised, clean images.
  labels_class = data[1]['label']
  logits_class = end_points['classes_sup']

  loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_class, logits=logits_class)
  loss_class = tf.reduce_mean(loss_class)

  ## Compute conditional entropy loss
  conditional_ent = -tf.reduce_sum(
      tf.nn.log_softmax(predictions_vat_baseline) *
      tf.nn.softmax(predictions_vat_baseline), axis=-1)
  loss_entmin = tf.reduce_mean(conditional_ent)

  # Combine VAT and classification loss as a weighted average, then add
  # weighted conditional entropy loss.
  loss = (loss_class +
          FLAGS.vat_weight * loss_vat +
          FLAGS.entmin_factor * loss_entmin)

  train_scalar_summaries = {
      'vat_eps': vat_eps,
      'vat_weight': FLAGS.vat_weight,
      'entmin_weight': FLAGS.entmin_factor,
      'vat_num_power_method_iters': vat_num_power_method_iters,
      'loss_class': loss_class,
      'loss_vat': loss_vat,
      'loss_vat_weighted': FLAGS.vat_weight * loss_vat,
      'loss_entmin': loss_entmin,
      'loss_entmin_weighted': FLAGS.entmin_factor * loss_entmin
  }

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

  return trainer.make_estimator(
      mode, loss, eval_metrics, train_scalar_summaries=train_scalar_summaries)
