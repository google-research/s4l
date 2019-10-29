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

"""Author: Marvin Ritter, tiny adaptations by Lucas beyer."""


import collections
import contextlib

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu.tpu_function import get_tpu_context
from tensorflow.python.training.moving_averages import assign_moving_average


# Many normalization methods define a different behavior for inference time.
# To build the according graph ops we control the beavior using scope.
# Note: Other frameworks handle is by passing around a boolean variable,
# usually called is_training.
# bn_current_batch: bool, if true use the current batch to compute batch
#   statistics. Otherwise use moving average or accumulated moments.
# sn_update_estimates: bool, if true update the estimate of the singular
#   vector in spectral norm.
NormModes = collections.namedtuple("NormModes",
                                   ["bn_current_batch", "sn_update_estimates"])

# Stack of NormModes. Use the contextmanagers below to set the current norm
# modes and get_norm_modes() to get the most recently set NormModes.
# Note: There is no default NormModes here. If your architecture uses batch
# norm or spectral norm you will have to use the context managers below to
# the NormModes (AbstractGenerater and AbstractDiscriminator does this for you).
_NORM_MODES = []


@contextlib.contextmanager
def norm_modes_for_training():
  """Set NormModes to create a graph for training."""
  _NORM_MODES.append(NormModes(bn_current_batch=True, sn_update_estimates=True))
  # Disallow nested norm modes because there is currently no use case for it.
  assert len(_NORM_MODES) == 1, "_NORM_MODES={}".format(_NORM_MODES)
  yield
  _NORM_MODES.pop()


@contextlib.contextmanager
def norm_modes_for_inference(bn_current_batch=False):
  """Set NormModes to create a graph for inference."""
  _NORM_MODES.append(
      NormModes(bn_current_batch=bn_current_batch, sn_update_estimates=False))
  # Disallow nested norm modes because there is currently no use case for it.
  assert len(_NORM_MODES) == 1, "_NORM_MODES={}".format(_NORM_MODES)
  yield
  _NORM_MODES.pop()


def get_norm_modes():
  """Returns the currently set NormModes."""
  if not _NORM_MODES:
    raise ValueError("No norm modes set.")
  return _NORM_MODES[-1]


def _moving_means_of_moments_for_inference(mean, variance, decay):
  """Use moving averages of moments during inference.

  Args:
    mean: Tensor of shape [num_channels] with the mean of the current batch.
    variance: Tensor of shape [num_channels] with the variance of the current
      batch.
    decay: Decay rate to use for moving averages.

  Returns:
    Tuple of (mean, variance) to use. This can the same as the inputs.
  """
  # Create the moving average variables and add them to the appropriate
  # collections.
  variable_collections = [
      tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
      tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,
  ]
  # Disable partition setting for moving_mean and moving_variance
  # as assign_moving_average op below doesn"t support partitioned variable.
  moving_mean = tf.get_variable(
      "moving_mean",
      shape=mean.shape,
      initializer=tf.zeros_initializer(),
      trainable=False,
      partitioner=None,
      collections=variable_collections)
  moving_variance = tf.get_variable(
      "moving_variance",
      shape=variance.shape,
      initializer=tf.ones_initializer(),
      trainable=False,
      partitioner=None,
      collections=variable_collections)
  if not get_norm_modes().bn_current_batch:
    return moving_mean, moving_variance

  # Update variables for mean and variance during training.
  update_moving_mean = assign_moving_average(
      moving_mean, tf.cast(mean, moving_mean.dtype), decay, zero_debias=False)
  update_moving_variance = assign_moving_average(
      moving_variance,
      tf.cast(variance, moving_variance.dtype),
      decay,
      zero_debias=False)
  with tf.control_dependencies([update_moving_mean, update_moving_variance]):
    mean = tf.identity([mean])
  return mean, variance


def _accumulated_moments_for_inference(mean, variance):
  """Use accumulated statistics for moments during inference.

  After training the user is responsible for filling the accumulators with the
  actual values. See _UpdateBnAccumulators() in eval_gan_lib.py for an example.

  Args:
    mean: Tensor of shape [num_channels] with the mean of the current batch.
    variance: Tensor of shape [num_channels] with the variance of the current
      batch.

  Returns:
    Tuple of (mean, variance) to use. This can the same as the inputs.
  """
  variable_collections = [
      tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,
  ]
  with tf.variable_scope(None, values=[mean, variance], default_name="accu"):
    # Create variables for accumulating batch statistic and use them during
    # inference. The ops for filling the accumulators must be created and run
    # before eval. See docstring above.
    accu_mean = tf.get_variable(
        "accu_mean",
        shape=mean.shape,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=variable_collections)
    accu_variance = tf.get_variable(
        "accu_variance",
        shape=variance.shape,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=variable_collections)
    accu_counter = tf.get_variable(
        "accu_counter",
        shape=[],
        initializer=tf.initializers.constant(1e-12),
        trainable=False,
        collections=variable_collections)
    # TODO(marvinritter): Remove this switch and the functionality below.
    # It adds unnecessary complexity.
    update_accus = tf.get_variable(
        "update_accus",
        shape=[],
        dtype=tf.int32,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=variable_collections)

    mean = tf.identity(mean, "mean")
    variance = tf.identity(variance, "variance")
    if get_norm_modes().bn_current_batch:
      return mean, variance

    # Return the accumulated batch statistics and add current batch statistics
    # to accumulators if update_accus variables equals 1.
    def update_accus_fn():
      return tf.group([
          tf.assign_add(accu_mean, mean),
          tf.assign_add(accu_variance, variance),
          tf.assign_add(accu_counter, 1),
      ])
    dep = tf.cond(
        tf.equal(update_accus, 1),
        update_accus_fn,
        tf.no_op)
    with tf.control_dependencies([dep]):
      return accu_mean / accu_counter, accu_variance / accu_counter


def cross_replica_concat(value, replica_id, num_replicas):
  """Reduce a concatenation of the `value` across TPU replicas.

  Args:
    value: Tensor to concatenate.
    replica_id: Integer tensor that indicates the index of the replica.
    num_replicas: Python integer, total number of replicas.

  Returns:
    Tensor of the same rank as value with first dimension `num_replicas`
    times larger.

  Raises:
    ValueError: If `value` is a scalar.
  """
  if value.shape.ndims < 1:
    raise ValueError("Value must have at least rank 1 but got {}.".format(
        value.shape.ndims))
  if num_replicas <= 1:
    return value
  with tf.name_scope(None, "tpu_cross_replica_concat"):
    # Mask is one hot encoded position of the core_index.
    mask = tf.to_float(tf.equal(tf.range(num_replicas), replica_id))
    # Expand dims with 1's to match rank of value.
    mask = tf.reshape(mask, [num_replicas] + [1] * value.shape.ndims)
    if value.dtype in {tf.bfloat16, tf.float32}:
      result = mask * value
    else:
      result = mask * tf.to_float(value)
    # Thanks to broadcasting now result is set only in the position pointed by
    # replica_id, the rest of the vector is set to 0's.
    # All these steps are basically implementing tf.scatter_nd which is missing
    # in TPU's backend since it doesn't support sparse operations.

    # Merge first 2 dimensions.
    # This is equivalent to (value.shape[0].value * num_replicas).
    # Using [-1] trick to support also scalar input.
    result = tf.reshape(result, [-1] + result.shape.as_list()[2:])
    # Each core set the "results" in position pointed by replica_id. When we now
    # sum across replicas we exchange the information and fill in local 0's with
    # values from other cores.
    result = tf.contrib.tpu.cross_replica_sum(result)
    # Now all the cores see exactly the same data.
    return tf.cast(result, dtype=value.dtype)


def cross_replica_mean(inputs, group_size=None):
  """Calculates the average value of inputs tensor across TPU replicas."""
  num_replicas = get_tpu_context().number_of_shards
  if not group_size:
    group_size = num_replicas
  if group_size == 1:
    return inputs
  if group_size != num_replicas:
    group_assignment = []
    assert num_replicas % group_size == 0
    for g in range(num_replicas // group_size):
      replica_ids = [g * group_size + i for i in range(group_size)]
      group_assignment.append(replica_ids)
  else:
    group_assignment = None
  return tf.contrib.tpu.cross_replica_sum(inputs, group_assignment) / tf.cast(
      group_size, inputs.dtype)


def cross_replica_moments(inputs, axis, parallel=True, group_size=None):
  """Compute mean and variance of the inputs tensor across TPU replicas.

  Args:
    inputs: A tensor with 2 or more dimensions.
    axis: Array of ints. Axes along which to compute mean and variance.
    parallel: Use E[x^2] - (E[x])^2 to compute variance. Then can be done
      in parallel to computing the mean and reducing the communication overhead.
    group_size: Integer, the number of replicas to compute moments arcoss.
      None or 0 will use all replicas (global).

  Returns:
    Two tensors with mean and variance.
  """
  # Compute local mean and then average across replicas.
  mean = tf.math.reduce_mean(inputs, axis=axis)
  mean = cross_replica_mean(mean)
  if parallel:
    # Compute variance using the E[x^2] - (E[x])^2 formula. This is less
    # numerically stable than the E[(x-E[x])^2] formula, but allows the two
    # cross-replica sums to be computed in parallel, saving communication
    # overhead.
    mean_of_squares = tf.reduce_mean(tf.square(inputs), axis=axis)
    mean_of_squares = cross_replica_mean(mean_of_squares, group_size=group_size)
    mean_squared = tf.square(mean)
    variance = mean_of_squares - mean_squared
  else:
    variance = tf.math.reduce_mean(
        tf.math.square(inputs - mean), axis=axis)
    variance = cross_replica_mean(variance, group_size=group_size)
  return mean, variance


def standardize_batch(inputs,
                      decay=0.999,
                      epsilon=1e-3,
                      data_format="NHWC",
                      use_moving_averages=True,
                      use_cross_replica_mean=None):
  """Adds TPU-enabled batch normalization layer.

  This version does not apply trainable scale or offset!
  It normalizes a tensor by mean and variance.

  Details on Batch Normalization can be found in "Batch Normalization:
  Accelerating Deep Network Training by Reducing Internal Covariate Shift",
  Ioffe S. and Szegedy C. 2015 [http://arxiv.org/abs/1502.03167].

  Note #1: This method computes the batch statistic across all TPU replicas,
  thus simulating the true batch norm in the distributed setting. If one wants
  to avoid the cross-replica communication set use_cross_replica_mean=False.

  Note #2: During training this will estimate the mean and variance using the
  current batch. For inference there are two options:
  a) Keep moving averages of the mean and variance during training by
     setting use_moving_averages=True.
  b) Set use_moving_averages=False to create acccumulators that will have to be
     filled with values for mean and variance after training. This can be done
     by doing forward passes and recording the mean/variance vectors.
  In both cases the inference behavior is activated when the current
  `NormModes`, as return by `get_norm_modes()`, sets update_bn_stats=False.

  Note #3: Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.99, 0.9, etc. Lower the `decay` value (trying
  `decay`=0.9) if model experiences reasonably good training performance but
  poor validation and/or test performance.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      values.
    decay: Decay rate to use for moving averages.
    epsilon: Small float added to variance to avoid dividing by zero.
    data_format: Input data format. NHWC or NCHW.
    use_moving_averages: If True keep moving averages of mean and variance that
      are used during inference. Otherwise use accumlators.
    use_cross_replica_mean: If True add operations to do computes batch norm
      statistics across all TPU cores. These ops are not compatible with other
      platforms. The default (None) will only add the operations if running
      on TPU.

  Returns:
    The normalized tensor with the same type and shape as `inputs`.
  """
  if data_format not in {"NCHW", "NHWC"}:
    raise ValueError(
        "Invalid data_format {}. Allowed: NCHW, NHWC.".format(data_format))
  if use_cross_replica_mean is None:
    # Default to global batch norm only on TPUs.
    use_cross_replica_mean = (get_tpu_context().number_of_shards is not None)

  inputs = tf.convert_to_tensor(inputs)
  inputs_dtype = inputs.dtype
  inputs_shape = inputs.get_shape()

  num_channels = inputs.shape[-1].value
  if num_channels is None:
    raise ValueError("`C` dimension must be known but is None")

  inputs_rank = inputs_shape.ndims
  if inputs_rank is None:
    raise ValueError("Inputs %s has undefined rank" % inputs.name)
  elif inputs_rank not in [2, 4]:
    raise ValueError(
        "Inputs %s has unsupported rank."
        " Expected 2 or 4 but got %d" % (inputs.name, inputs_rank))
  # Bring 2-D inputs into 4-D format.
  if inputs_rank == 2:
    new_shape = [-1, 1, 1, num_channels]
    if data_format == "NCHW":
      new_shape = [-1, num_channels, 1, 1]
    inputs = tf.reshape(inputs, new_shape)

  # Execute a distributed batch normalization
  axis = 1 if data_format == "NCHW" else 3
  inputs = tf.cast(inputs, tf.float32)
  reduction_axes = [i for i in range(4) if i != axis]
  if use_cross_replica_mean:
    mean, variance = cross_replica_moments(inputs, reduction_axes)
  else:
    counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
        inputs, reduction_axes, keep_dims=False)
    mean, variance = tf.nn.normalize_moments(
        counts, mean_ss, variance_ss, shift=None)

  if use_moving_averages:
    mean, variance = _moving_means_of_moments_for_inference(
        mean=mean, variance=variance, decay=decay)
  else:
    mean, variance = _accumulated_moments_for_inference(
        mean=mean, variance=variance)

  outputs = tf.nn.batch_normalization(
      inputs,
      mean=mean,
      variance=variance,
      offset=None,
      scale=None,
      variance_epsilon=epsilon)
  outputs = tf.cast(outputs, inputs_dtype)

  # Bring 2-D inputs back into 2-D format.
  if inputs_rank == 2:
    outputs = tf.reshape(outputs, [-1] + inputs_shape[1:].as_list())
  outputs.set_shape(inputs_shape)
  return outputs


def batch_norm(inputs, center=True, scale=True, name=None, **std_kw):
  """Performs the vanilla batch normalization with trainable scaling and offset.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      `NCHW`.
    center: If True, add offset of beta to normalized tensor.
    scale: If True, multiply by gamma. When the next layer is linear  this can
      be disabled since the scaling will be done by the next layer.
    name: Name of the variable scope.
    **std_kw: Arguments forwarded to `standardize_batch`.

  Returns:
    The normalized tensor with the same type and shape as `inputs`.
  """
  with tf.variable_scope(name, values=[inputs], default_name="batch_norm"):
    outputs = standardize_batch(inputs, **std_kw)
    num_channels = inputs.shape[-1].value

    # Allocate parameters for the trainable variables.
    var_collections = [
        tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES
    ]
    if scale:
      gamma = tf.get_variable(
          "gamma", [num_channels],
          collections=var_collections,
          initializer=tf.ones_initializer())
      outputs *= gamma
    if center:
      beta = tf.get_variable(
          "beta", [num_channels],
          collections=var_collections,
          initializer=tf.zeros_initializer())
      outputs += beta
    return outputs


def cross_replica_batch_norm(inputs, training, decay=0.99, **kw):
  """Applies batch norm in a way that accumulates statistics across TPU cores.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      `NCHW`.
    training: Whether or not the layer is in training mode.
    decay: Decay factor for exponential moving averages of batch mean and
      variance used during evaluation.
    **kw: Other arguments forwarded to `batch_norm` and `standardize_batch`.

  Returns:
    The normalized tensor with the same type and shape as `inputs`.
  """
  if training:
    mode = norm_modes_for_training
  else:
    mode = norm_modes_for_inference
  with mode():
    return batch_norm(inputs, decay=decay, **kw)

