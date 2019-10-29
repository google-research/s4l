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

"""Base trainer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from tensorflow.contrib.tpu import CrossShardOptimizer, TPUEstimatorSpec
import tensorflow.contrib.summary as contrib_summary

import datasets
import utils

FLAGS = flags.FLAGS


def get_lr(global_step, base_lr,  # pylint: disable=missing-docstring
           decay_steps, lr_decay_factor, warmup_steps):

  warmup_lr = 0.0
  if warmup_steps > 0:
    warmup_lr = (tf.cast(global_step, tf.float32) * (base_lr / warmup_steps))

  if decay_steps:
    normal_lr = tf.train.piecewise_constant(
        global_step,
        [s for s in decay_steps],
        [base_lr * (lr_decay_factor ** i) for i in range(len(decay_steps) + 1)]
    )
  else:
    normal_lr = base_lr

  lr = tf.cond(
      tf.less(global_step, tf.cast(warmup_steps, dtype=tf.dtypes.int64)),
      lambda: warmup_lr, lambda: normal_lr)

  return lr


# TODO(akolesnikov): add more logging
class Trainer(object):
  """Base trainer class."""

  def __init__(self,
               update_batchnorm_params=True):
    self.update_batchnorm_params = update_batchnorm_params

    num_samples = datasets.get_count(FLAGS.train_split)
    if FLAGS.num_supervised_examples:
      num_samples = FLAGS.num_supervised_examples
    steps_per_epoch = num_samples // FLAGS.batch_size
    self.steps_per_epoch = steps_per_epoch

    global_step = tf.train.get_or_create_global_step()
    self.global_step_inc = tf.assign_add(global_step, 1)

    # lr_scale_batch_size defines a canonical batch size that is coupled with
    # the initial learning rate. If actual batch size is not the same as
    # canonical than learning rate is linearly scaled. This is very convinient
    # as this allows to vary batch size without recomputing learning rate.
    lr_factor = 1.0
    if FLAGS.lr_scale_batch_size:
      lr_factor = FLAGS.batch_size / float(FLAGS.lr_scale_batch_size)

    # We actually also accept fractional epochs.
    schedule_in_steps = utils.get_schedule_from_config(
        FLAGS.schedule, steps_per_epoch)
    warmup, decays = schedule_in_steps[0], schedule_in_steps[1:-1]

    self.lr = get_lr(
        global_step,
        base_lr=FLAGS.lr * lr_factor,
        decay_steps=decays,
        lr_decay_factor=FLAGS.lr_decay_factor,
        warmup_steps=warmup)

    # TODO(marvinritter): Re-enable summaries with support for TPU training.
    # tf.summary.scalar('learning_rate', self.lr)

  def get_train_op(self, loss,  # pylint: disable=missing-docstring
                   var_list=None,
                   add_reg_loss=True,
                   use_tpu=False):

    if add_reg_loss:
      l2_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
      loss += l2_loss

    optimizer = FLAGS.optimizer
    if optimizer == 'sgd':
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                             momentum=0.9)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    else:
      raise ValueError('Unknown optimizer: %s' % optimizer)

    if use_tpu:
      # Wrap optimizer in CrossShardOptimizer which takes care of
      # synchronizing the weight updates between TPU cores.
      optimizer = CrossShardOptimizer(optimizer)

    opt_step = optimizer.minimize(loss, var_list=var_list,
                                  colocate_gradients_with_ops=True)

    if self.update_batchnorm_params:
      opt_step = tf.group([opt_step] +
                          tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    opt_step = tf.group([opt_step, self.global_step_inc])

    return opt_step


def eval_ema_scope():
  """Return a context in which variables are replaced with their EMA shadows.

  Example usage:

  with eval_ema_scope():
    output = model.make_predictions()

  Note: If you use this context, then there is no guarantee about
  which variables (plain or EMA) you'll see if you access them before
  the context enters or after it exits. Basically, enter this context,
  make a prediction, and then throw your model away.

  Note: If you construct tensors outside of this context, and want to
  re-use them inside such that they use the EMA shadow variables, use
  `tf.identity`, as such:

  loss = ...
  with eval_ema_scope():
    output = tf.identity(loss)
  """
  # `decay` shouldn't matter during eval
  ema = tf.train.ExponentialMovingAverage(decay=0.0, name='EMA')

  # Construct the EMA shadow variables, so that we can load them
  # from a checkpoint.
  ema.apply(tf.trainable_variables())

  # Copy EMA "shadow parameters" into their corresponding vanilla
  # parameters so that making predictions use the EMA parameters.
  assign_ema_vars = [tf.assign(var, ema.average(var))
                     for var in tf.trainable_variables()]

  return tf.control_dependencies(assign_ema_vars)


def make_estimator(mode, loss=None, eval_metrics=None, predictions=None,
                   predict_fn=None, predict_input=None,
                   train_scalar_summaries=None,
                   polyak_averaging=False):
  """Returns an EstimatorSpec (maybe TPU) for all modes."""

  # Always use TPUEstimator, even when not using TPU, then it's (almost) no-op.
  spec_type = TPUEstimatorSpec

  if mode == tf.estimator.ModeKeys.PREDICT:
    # For backwards-compatibility, still accept `predictions`.
    if predictions is None:
      # What we do here is create the hub module and use its predictions.
      assert predict_fn is not None, 'Need to pass `predict_fn` arg.'
      assert predict_input is not None, 'Need to pass `predict_input` arg.'
      tf_hub_module = make_hub_predictor(predict_fn)
      predictions = tf_hub_module(predict_input)

    if polyak_averaging:
      with eval_ema_scope():
        # Use `tf.identity` to ensure that the dependencies are executed first.
        # (Otherwise, since loss is constructed outside of this function, the
        # `eval_ema_scope` scope would do nothing)
        predictions = tf.identity(predictions)

    return spec_type(mode=mode, predictions=predictions)

  if mode == tf.estimator.ModeKeys.EVAL:
    if polyak_averaging:
      with eval_ema_scope():
        # Use `tf.identity` to ensure that the dependencies are executed first.
        # (Otherwise, since loss is constructed outside of this function, the
        # `eval_ema_scope()` scope would do nothing)
        loss = tf.identity(loss)

        # `eval_metrics` is an ordered pair of a lambda, and a list of tensors
        # that are evaluated and fed into the lambda. Do "surgery" to wrap only
        # the tensors into `tf.identity` (see comment above)
        eval_metrics = (
            eval_metrics[0],
            [tf.identity(x) for x in eval_metrics[1]],
        )

    return spec_type(mode=mode, loss=loss, eval_metrics=eval_metrics)

  if mode == tf.estimator.ModeKeys.TRAIN:
    assert loss is not None, 'Need to pass `loss` arg.'
    trainer = Trainer(update_batchnorm_params=True)

    if polyak_averaging:
      # Set EMA half-life to one epoch
      ema_decay = 0.5**(1.0/trainer.steps_per_epoch)
      ema = tf.train.ExponentialMovingAverage(
          ema_decay, zero_debias=True, name='EMA')

    if FLAGS.use_summaries:
      # Need to reshape with a fake batch for summaries on TPU host.
      # Also need to explicitly note which tensors are used, and pass
      # them in explicitly.
      summary_names = ['lr', 'loss']
      summary_reshaped_tensors = [tf.reshape(trainer.lr, [1]),
                                  tf.reshape(loss, [1])]

      if train_scalar_summaries is not None:
        for name, summary_tensor in train_scalar_summaries.items():
          summary_names.append(name)
          summary_reshaped_tensors.append(tf.reshape(summary_tensor, [1]))

      def host_call_fn(gs, *summary_tensors):
        gs = gs[0]
        with contrib_summary.create_file_writer(
            FLAGS.workdir).as_default():
          with contrib_summary.always_record_summaries():
            for name, reshaped_tensor in zip(summary_names, summary_tensors):
              contrib_summary.scalar(
                  name, tf.reduce_mean(reshaped_tensor), step=gs)
            return contrib_summary.all_summary_ops()

      gs_t = tf.reshape(tf.train.get_global_step(), [1])
      host_call = (host_call_fn, [gs_t] + summary_reshaped_tensors)
    else:
      host_call = None

    train_op = trainer.get_train_op(loss, use_tpu=FLAGS.tpu_name is not None)
    if polyak_averaging:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(tf.trainable_variables())

    return spec_type(
        mode=mode, loss=loss, train_op=train_op, host_call=host_call)

  raise ValueError('Unsupported mode %s' % mode)


def make_hub_predictor(model_fn):
  """Creates a tf.hub module for export (in PREDICT mode).

  Args:
    model_fn: This function is called with the placeholder inputs and
      is_training as arguments and should call the model, returning both the
      end_points collection and the tensor that should become the hub module's
      default prediction (for the default signature).

  Returns:
    The tf.hub module.
  """

  # This defines a function called by the hub module to create the model's
  # graph in a new/empty tf.Graph, hence it creates the placeholder etc.
  def create_model_fn(is_training):  # pylint: disable=missing-docstring
    input_shape = utils.str2intlist(FLAGS.serving_input_shape)
    img = tf.placeholder(shape=input_shape, dtype=tf.float32)

    # This is an example of calling `apply_model_semi` with only one of the
    # inputs provided. The outputs will simply use the given names:
    end_points, predictions = model_fn(img, is_training)

    # Register both the class output and all endpoints to the hub module.
    hub.add_signature(inputs={'image': img}, outputs=predictions)
    hub.add_signature(inputs={'image': img}, outputs=end_points,
                      name='representation')

  tf_hub_module_spec = hub.create_module_spec(
      create_model_fn, [(["is_training"], {
          'is_training': True
      }), (set(), {
          'is_training': False
      })],
      # For some not understood reason, this is necessary when the model uses
      # cross_replica_batch_norm. We verified that moving averages are still
      # being stored in the hub module just fine.
      drop_collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES])
  tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
  hub.register_module_for_export(tf_hub_module, export_name='module')

  return tf_hub_module
