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

import utils
import datasets
import semi_supervised.utils as ss_utils
import trainer


FLAGS = tf.flags.FLAGS


def _stable_norm(tensor, norm_type, axis, keep_dims=False):
  """Stable (for backprop) implementation of a few cases of tf.norm.

  An implementation of a few common cases of `tf.norm` used in the batch-hard
  loss function, but adding a hard cut to the euclidean distance computation
  because d/dx sqrt(x) is inf at 0.

  Args:
    tensor: a tensor to be normalized.
    norm_type: a string, one of "euclidean", "squared_euclidean", "manhattan"
      describing which norm is to be computed.
    axis: an integer indicating which axis to take the norm across.
    keep_dims: a boolean indicating whether to keep (True) the dimension across
      which the norm is computed as singleton, or drop it (False).

  Returns:
    A tensor containing the norms of `tensor` along `axis`.

  Raises:
    ValueError: on invalid `norm_type` argument.
  """
  if 'euclidean' in norm_type:
    norm = tf.reduce_sum(tensor*tf.conj(tensor), axis=axis, keep_dims=keep_dims)
    if 'squared' not in norm_type:
      norm = tf.sqrt(tf.maximum(norm, 1e-12))
    return norm
  elif norm_type == 'manhattan':
    return tf.reduce_sum(tf.abs(tensor), axis=axis, keep_dims=keep_dims)
  else:
    raise ValueError('Unknown norm type {}'.format(norm_type))


def cdist(tensor, axis, normaxis=-1, norm='euclidean'):
  """Compute all-to-all distance matrix symbolically.

  Args:
    tensor: a (typically 2D) tensor for which to compute all pairwise distances.
    axis: integer specifying the axis along which all pairs should be taken.
      That means the resulting distance matrix will have shape DxD, with D
      being equal to `tensor.shape[axis]`. Any axis before that is kept as-is.
    normaxis: integer specifying along which axis the norm should be taken for
      distance computation.
    norm: string, one of "euclidean", "squared_euclidean", "manhattan",
      indicating which distance to be computed.

  Returns:
    A square tensor containing all pairwise distances.

  Raises:
    ValueError: on invalid `norm` argument.
  """
  diff = tf.expand_dims(tensor, axis) - tf.expand_dims(tensor, axis+1)
  return _stable_norm(diff, norm, axis=normaxis)


def _batched_extract_indices(batch, indices):
  """This is a weird way of doing `batch[arange(len(batch)), indices]`."""
  upper = tf.shape(indices, out_type=indices.dtype)[0]
  iota = tf.range(upper, dtype=indices.dtype)
  return tf.gather_nd(batch, tf.stack([iota, indices], axis=1))


def batch_hard(batch, identities, margin=0.1, soft=False, norm='euclidean',
               sample_pos=True, sample_neg=True, return_details=False):
  """Compute the "batch hard" triplet loss from arxiv.org/abs/1703.07737.

  Args:
    batch: a 2D tensor of shape (`batch`, `features`) for which to compute the
      batch-hard loss, all samples being along the first dimension.
    identities: a 1D tensor of length `batch` of whatever type (typically
      string or int) which contains the identities of the samples in the batch.
    margin: a float indicating the margin to be used in the loss. 0.0 for none.
    soft: a boolean indicating whether to use the soft-margin formulation
      (True) or the regular (hinge) margin formulation (False).
    norm: string, one of "euclidean", "squared_euclidean", "manhattan",
      indicating which distance to be used.
    sample_pos: boolean, if False, the hardest positive for each sample is used,
      if True, the positive to be used is randomly sampled from all positives,
      according to their hardness.
    sample_neg: boolean, like `sample_pos` but for the negatives.
    return_details: boolean, if True, many more tensors are returned, which can
      be useful not only for logging but also for further use in a model.
      The extra returned tensors are, in that order:
        2D tensor of selected positives,
        2D tensor of selected negatives,
        flattened 1D tensor of *all* positive distances,
        flattened 1D tensor of *all* negative distances,
        on-the-fly within-batch top-1 retrieval accuracy.

  Returns:
    1D tensor of length `batch` with the loss for each sample in the batch.
    If `return_details` is True, see that argument's description.

  Raises:
    ValueError: on invalid `norm` argument.
  """
  dists = cdist(batch, axis=0, norm=norm)

  # Compute masks of which entries in dists belong to the same identity,
  # and to a different identity
  same_identity_mask = tf.equal(tf.expand_dims(identities, 0),
                                tf.expand_dims(identities, 1))
  other_identity_mask = tf.logical_not(same_identity_mask)

  # Remove the diagonal (myself) from `same_identity`.
  inverted_eye = tf.logical_not(tf.eye(tf.shape(batch)[0], dtype=tf.bool))
  # inverted_eye = tf.cast(1.0 - tf.diag(tf.ones_like(dists[0])), tf.bool)
  same_identity_mask = tf.logical_and(same_identity_mask, inverted_eye)
  infs = tf.ones_like(dists)*tf.constant(float('inf'))

  # The hardest positives are those positives which are furthest away.
  dists_positives = tf.where(same_identity_mask, dists, -infs)
  if sample_pos:
    indices = tf.multinomial(dists_positives, 1)[:, 0]
    positives = _batched_extract_indices(dists, indices)
  else:
    positives = tf.reduce_max(dists_positives, axis=1)

  # The hardest negatives are the closest ones, i.e. with the smallest distance.
  dists_negatives = tf.where(other_identity_mask, dists, infs)
  if sample_neg:
    indices = tf.multinomial(-dists_negatives, 1)[:, 0]
    negatives = _batched_extract_indices(dists, indices)
  else:
    negatives = tf.reduce_min(dists_negatives, axis=1)

  diff = (positives + margin) - negatives
  if soft:
    diff = tf.nn.softplus(diff)
  else:
    diff = tf.nn.relu(diff)

  # No more details wanted, we can stop here.
  if not return_details:
    return diff

  # These can be useful to summarize
  positive_dists = tf.boolean_mask(dists, same_identity_mask)
  negative_dists = tf.boolean_mask(dists, other_identity_mask)

  # For cheap on-the-fly top-1 evaluation.
  _, topk_idx = tf.nn.top_k(-dists, k=2)
  top1_idx = topk_idx[:, 1]  # topk[0] will always be self.
  top1_is_same = _batched_extract_indices(same_identity_mask, top1_idx)
  top1_accuracy = tf.reduce_mean(tf.cast(top1_is_same, tf.float32))

  return (diff, positives, negatives, positive_dists, negative_dists,
          top1_accuracy)


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
          'embeddings': FLAGS.triplet_embed_dim,
          'classes': datasets.get_auxiliary_num_classes(),
      })
      return end_points, end_points['classes']
    return trainer.make_estimator(mode, predict_fn=model_building_fn,
                                  predict_input=data['image'])

  # In all other cases, we are in train/eval mode.
  images_unsup = data[0]['image']
  images_sup = data[1]['image']

  # There is one special case, typically in eval mode, when we don't want to use
  # multiple examples, but a single one. In that case, add the fake length-1
  # example dimension to the input so that everything still works.
  # i.e. turn BHWC into B1HWC
  if images_unsup.shape.ndims == 4:
    images_unsup = images_unsup[:, None, ...]
  if images_sup.shape.ndims == 4:
    images_sup = images_sup[:, None, ...]

  # Find out the number of examples that have been created per image, which
  # may be different for sup/unsup, and use that for creating the labels.
  ninstances_unsup, nexamples_unsup = images_unsup.shape[:2]
  ninstances_sup, nexamples_sup = images_sup.shape[:2]

  # Then, fold the examples into the batch.
  images_unsup = utils.into_batch_dim(images_unsup)
  images_sup = utils.into_batch_dim(images_sup)

  # If we're not doing exemplar on the unsupervised data, skip it!
  if not FLAGS.triplet_loss_unsup:
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
            'embeddings': FLAGS.triplet_embed_dim,
            'classes': datasets.get_auxiliary_num_classes(),
        })

  # Labelled classification loss
  # =====

  # Compute the supervision loss for each example of the supervised branch.
  labels_class = utils.repeat(data[1]['label'], nexamples_sup)
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
  losses_ex = []

  def do_triplet(embeddings, nexamples, ninstances):
    """Applies the triplet loss to the given embeddings."""
    # Empirically, normalizing the embeddings is more robust.
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    # Generate the labels as [0 0 0 1 1 1 ...]
    labels = utils.repeat(tf.range(ninstances), nexamples)

    # Apply batch-hard loss with a soft-margin.
    losses_tri = batch_hard(embeddings, labels, margin=0.0, soft=True,
                            sample_pos=False, sample_neg=False)
    return tf.reduce_mean(losses_tri)

  # Compute exemplar triplet loss on the unsupervised images
  if FLAGS.triplet_loss_unsup:
    loss_ex_unsup = do_triplet(
        end_points['embeddings_unsup'], ninstances_unsup, nexamples_unsup)
    losses_ex.append(tf.reduce_mean(loss_ex_unsup))

  # Compute exemplar triplet loss on the supervised images.
  if FLAGS.triplet_loss_sup:
    loss_ex_sup = do_triplet(
        end_points['embeddings_sup'], ninstances_sup, nexamples_sup)
    losses_ex.append(tf.reduce_mean(loss_ex_sup))

  loss_ex = tf.reduce_mean(losses_ex) if losses_ex else 0.0

  # Combine the two losses as a weighted average.
  loss = loss_class + FLAGS.triplet_loss_weight * loss_ex

  return trainer.make_estimator(mode, loss)
