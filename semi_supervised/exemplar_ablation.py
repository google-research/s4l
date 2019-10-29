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

from google3.learning.brain.research.dune.experimental.representation import utils
import google3.learning.brain.research.dune.experimental.representation.datasets as datasets
import google3.learning.brain.research.dune.experimental.representation.semi_supervised.utils as ss_utils
import google3.learning.brain.research.dune.experimental.representation.trainer as trainer

from google3.photos.vision.human_sensing.tensorflow.layers.batch_hard_triplet_loss import batch_hard


FLAGS = tf.flags.FLAGS


def model_fn(data, mode):
  """Produces a loss for the exemplar task.

  Args:
    data: Dict of inputs ("image" being the image)
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """

  # In this mode (called once at the end of training), we create the tf.Hub
  # module in order to export the model, and use that to do one last prediction.
  if mode == tf.estimator.ModeKeys.PREDICT:
    def model_building_fn(img, is_training):
      end_points = ss_utils.apply_model_semi(img, None, is_training, outputs={
          'embeddings': FLAGS.config.triplet_embed_dim,
          'classes': datasets.get_auxiliary_num_classes(),
      })
      return end_points, end_points['classes']
    return trainer.make_estimator(mode, predict_fn=model_building_fn,
                                  predict_input=data['image'])

  # In all other cases, we are in train/eval mode.
  images_unsup = data[0]['image']
  images_sup = data[1]['image']

  # There is one special case, typically in eval mode, when we don't want to use
  # multiple exemplars, but a single one. In that case, add the fake length-1
  # exemplar dimension to the input so that everything still works.
  # i.e. turn BHWC into B1HWC
  if images_unsup.shape.ndims == 4:
    images_unsup = images_unsup[:, None, ...]
  if images_sup.shape.ndims == 4:
    images_sup = images_sup[:, None, ...]

  # Find out the number of exemplars that have been created per image, which
  # may be different for sup/unsup, and use that for creating the labels.
  ninstances_unsup, nexemplars_unsup = images_unsup.shape[:2]
  ninstances_sup, nexemplars_sup = images_sup.shape[:2]

  # Then, fold the exemplars into the batch.
  images_unsup = utils.into_batch_dim(images_unsup)
  images_sup = utils.into_batch_dim(images_sup)

  # We're not doing exemplar on the unsupervised data, skip it!
  if FLAGS.config.triplet_on == 'sup':
    images_unsup = None

  # Forward them both through the model. The scope is needed for tf.Hub export.
  with tf.variable_scope('module'):
    # Here, we pass both inputs to `apply_model_semi`, and so we now get
    # outputs corresponding to each in `end_points` as "classes_unsup" and
    # similar, which we will use below.
    end_points = ss_utils.apply_model_semi(
        images_unsup, images_sup,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        outputs={
            'embeddings': FLAGS.config.triplet_embed_dim,
            'classes': datasets.get_auxiliary_num_classes(),
        })

  # Labelled classification loss
  # =====

  # Compute the supervision loss for each exemplar of the supervised branch.
  labels_class = utils.repeat(data[1]['label'], nexemplars_sup)
  logits_class = end_points['classes_sup']
  losses_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_class, logits=logits_class)
  loss_class = tf.reduce_mean(losses_class)

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metrics = (
        lambda labels_class, logits_class, losses_class: {  # pylint: disable=g-long-lambda
            'classification/top1 accuracy':
                utils.top_k_accuracy(1, labels_class, logits_class),
            'classification/top5 accuracy':
                utils.top_k_accuracy(5, labels_class, logits_class),
            'classification/loss': tf.metrics.mean(losses_class),
        }, [labels_class, logits_class, losses_class])

    return trainer.make_estimator(mode, loss_class, eval_metrics)

  # Exemplar triplet loss
  # =====
  summaries = {}

  def do_triplet(embeddings, labels, summaries_prefix='triplet/'):
    """Computes one of many variants of the triplet loss."""
    if FLAGS.config.triplet_normalize == 'batch':
      embeddings = tf.nn.l2_normalize(embeddings, axis=0)
    elif FLAGS.config.triplet_normalize == 'embedding':
      embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    # For exemplar, batch-hard should be a lot better,
    # because there is no label-noise by design!
    if FLAGS.config.triplet_variant == 'semihard':
      return tf.contrib.losses.metric_learning.triplet_semihard_loss(
          labels, embeddings, margin=FLAGS.config.triplet_margin)
    elif FLAGS.config.triplet_variant == 'batch_hard':
      losses_tri, _, _, pos_dists, neg_dists, top1_accu = \
          batch_hard(embeddings, labels, margin=FLAGS.config.triplet_margin,
                     soft=FLAGS.config.triplet_soft_margin, norm='euclidean',
                     sample_pos=FLAGS.config.triplet_sample,
                     sample_neg=FLAGS.config.triplet_sample,
                     return_details=True)
      loss_ex = tf.reduce_mean(losses_tri)
      summaries[summaries_prefix + 'loss'] = loss_ex
      summaries[summaries_prefix + 'top1'] = tf.reduce_mean(top1_accu)
      summaries[summaries_prefix + 'pos_dists'] = tf.reduce_mean(pos_dists)
      summaries[summaries_prefix + 'neg_dists'] = tf.reduce_mean(neg_dists)
      return loss_ex

  # Generate the labels for the exemplar loss as [0 0 0 1 1 1 ...]
  labels_ex_unsup = utils.repeat(tf.range(ninstances_unsup), nexemplars_unsup)
  labels_ex_sup = utils.repeat(tf.range(ninstances_sup), nexemplars_sup)

  # We have the option to apply the triplet loss to the unlabelled and the
  # labelled parts of the batch separately, or combine them into one big batch.
  # Separately could make sense because the same image could appear in both.
  if FLAGS.config.triplet_on == 'combined':
    # But if we want to use them together, make sure they don't overlap!
    # [0 0 1 1 ... 0 0 1 1 ...] is bad, we need [0 0 1 1 ... 7 7 8 8 ...]
    labels_ex = tf.concat([labels_ex_unsup,
                           labels_ex_sup + ninstances_unsup], axis=0)
    loss_ex = do_triplet(end_points['embeddings'], labels_ex)
  else:
    loss_ex_unsup = do_triplet(end_points['embeddings_unsup'], labels_ex_unsup,
                               summaries_prefix='triplet/unsup/')
    loss_ex_sup = do_triplet(end_points['embeddings_sup'], labels_ex_sup,
                             summaries_prefix='triplet/sup/')
    loss_ex = {
        'both': 0.5 * loss_ex_unsup + 0.5 * loss_ex_sup,
        'unsup': loss_ex_unsup,
        'sup': loss_ex_sup,
    }[FLAGS.config.triplet_on]

  # Combine the two losses as a weighted average.
  loss = loss_class + FLAGS.config.triplet_loss_weight * loss_ex

  return trainer.make_estimator(mode, loss, train_scalar_summaries=summaries)
