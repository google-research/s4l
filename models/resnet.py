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

"""Implements Resnet model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v1 as tf
from tensorflow.contrib.layers import l2_regularizer


def get_shape_as_list(x):
  return x.get_shape().as_list()


def fixed_padding(x, kernel_size):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  x = tf.pad(x, [[0, 0],
                 [pad_beg, pad_end], [pad_beg, pad_end],
                 [0, 0]])
  return x


def batch_norm(x, training):
  return tf.layers.batch_normalization(x, fused=True, training=training)


def identity_norm(x, training):
  del training
  return x


def maybe_group_conv(x, filters, groups, **kw):
  """Does regular conv or (inefficient) group-conv.

  Args:
    x: The input image/feature-map.
    filters: The total number of filters in the convolution, i.e. number of
      channels that the output feature-map will have.
    groups: The number of groups. Setting it to 1 leads to regular convolution,
      setting it to `filters` leads to separable convolution (but inefficient).
    **kw: Any further arguments are passed along to the `conv2d` operation.

  Returns:
    The output feature-map with `filters` channels.
  """
  assert filters % groups == 0, ('Filters ({}) not divisible by '
                                 'groups ({}).'.format(filters, groups))
  assert x.shape.rank == 4, 'Only implemented for 4D inputs.'

  if groups == 1:
    return tf.layers.conv2d(x, filters, **kw)

  outputs = []
  for i, xi in enumerate(tf.split(x, groups, axis=-1)):
    with tf.variable_scope('group_{}'.format(i)):
      outputs.append(tf.layers.conv2d(xi, filters // groups, **kw))
  return tf.concat(outputs, axis=-1)


def resblock_v1(x, filters, training,  # pylint: disable=missing-docstring
                strides=1,
                dilation=1,
                activation_fn=tf.nn.relu,
                normalization_fn=batch_norm,
                kernel_regularizer=None,
                name='unit'):

  with tf.variable_scope(name):

    # Record input tensor, such that it can be used later in as skip-connection
    x_shortcut = x

    # Project input if necessary
    if (strides > 1) or (filters != x.shape[-1]):
      with tf.variable_scope('proj'):
        x_shortcut = tf.layers.conv2d(x_shortcut, filters=filters,
                                      kernel_size=1,
                                      strides=strides,
                                      kernel_regularizer=kernel_regularizer,
                                      use_bias=False,
                                      padding='SAME')
        x_shortcut = normalization_fn(x_shortcut, training=training)

    # First convolution
    with tf.variable_scope('a'):
      x = fixed_padding(x, kernel_size=3  + 2 * (dilation - 1))
      x = tf.layers.conv2d(x, filters=filters,
                           kernel_size=3,
                           dilation_rate=dilation,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='VALID')
      x = normalization_fn(x, training=training)
      x = activation_fn(x)

    # Second convolution
    with tf.variable_scope('b'):
      x = fixed_padding(x, kernel_size=3)
      x = tf.layers.conv2d(x, filters=filters,
                           strides=strides,
                           kernel_size=3,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='VALID')
      x = normalization_fn(x, training=training)

    # Skip connection
    x = x_shortcut + x
    x = activation_fn(x)

  return x


def bottleneck_v1(x, filters, training,  # pylint: disable=missing-docstring
                  strides=1,
                  dilation=1,
                  groups=1,
                  activation_fn=tf.nn.relu,
                  normalization_fn=batch_norm,
                  kernel_regularizer=None,
                  name='unit'):

  with tf.variable_scope(name):

    # Record input tensor, such that it can be used later in as skip-connection
    x_shortcut = x

    # Project input if necessary
    with tf.variable_scope('proj'):
      if (strides > 1) or (filters != x.shape[-1]):
        x_shortcut = tf.layers.conv2d(x_shortcut, filters=filters,
                                      kernel_size=1, strides=strides,
                                      kernel_regularizer=kernel_regularizer,
                                      use_bias=False,
                                      padding='SAME')
        x_shortcut = normalization_fn(x_shortcut, training=training)

    # Note that ResNeXt doubles middle's channel count!
    middle_filters = filters // 4 if groups == 1 else filters // 2

    # First convolution
    with tf.variable_scope('a'):
      # Note, that unlike original Resnet paper we never use stride in the first
      # convolution. Instead, we apply stride in the second convolution. The
      # reason is that the first convolution has kernel of size 1x1, which
      # results in information loss when combined with stride bigger than one.
      x = tf.layers.conv2d(x, filters=middle_filters,
                           kernel_size=1,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='SAME')
      x = normalization_fn(x, training=training)
      x = activation_fn(x)

    # Second convolution
    with tf.variable_scope('b'):
      x = fixed_padding(x, kernel_size=3  + 2 * (dilation - 1))
      x = maybe_group_conv(x, filters=middle_filters, groups=groups,
                           strides=strides,
                           kernel_size=3,
                           dilation_rate=dilation,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='VALID')
      x = normalization_fn(x, training=training)
      x = activation_fn(x)

    # Third convolution
    with tf.variable_scope('c'):
      x = tf.layers.conv2d(x, filters=filters,
                           kernel_size=1,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='SAME')
      x = normalization_fn(x, training=training)

    # Skip connection
    x = x_shortcut + x
    x = activation_fn(x)

  return x


def resblock_v2(x, filters, training,  # pylint: disable=missing-docstring
                strides=1,
                dilation=1,
                activation_fn=tf.nn.relu,
                normalization_fn=batch_norm,
                kernel_regularizer=None,
                no_shortcut=False,
                out_filters=None,
                name='unit'):

  with tf.variable_scope(name):

    # If the number of output filters is not specified, it defaults to the
    # number of input filters.
    out_filters = out_filters or filters

    # Record input tensor, such that it can be used later in as skip-connection
    x_shortcut = x

    x = normalization_fn(x, training=training)
    x = activation_fn(x)

    # Project input if necessary
    with tf.variable_scope('proj'):
      if (strides > 1) or (out_filters != x.shape[-1]):
        x_shortcut = tf.layers.conv2d(x, filters=out_filters, kernel_size=1,
                                      strides=strides,
                                      kernel_regularizer=kernel_regularizer,
                                      use_bias=False,
                                      padding='VALID')

    # First convolution
    with tf.variable_scope('a'):
      x = fixed_padding(x, kernel_size=3 + 2 * (dilation - 1))
      x = tf.layers.conv2d(x, filters=filters,
                           kernel_size=3,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           dilation_rate=dilation,
                           padding='VALID')

      x = normalization_fn(x, training=training)
      x = activation_fn(x)

    # Second convolution
    with tf.variable_scope('b'):
      x = fixed_padding(x, kernel_size=3)
      x = tf.layers.conv2d(x, filters=out_filters,
                           strides=strides,
                           kernel_size=3,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='VALID')

  if no_shortcut:
    return x
  else:
    return x + x_shortcut


def bottleneck_v2(x, filters, training,  # pylint: disable=missing-docstring
                  strides=1,
                  dilation=1,
                  groups=1,
                  activation_fn=tf.nn.relu,
                  normalization_fn=batch_norm,
                  kernel_regularizer=None,
                  no_shortcut=False,
                  out_filters=None,
                  name='unit'):

  with tf.variable_scope(name):

    # If the number of output filters is not specified, it defaults to the
    # number of input filters.
    out_filters = out_filters or filters

    # Record input tensor, such that it can be used later in as skip-connection
    x_shortcut = x

    x = normalization_fn(x, training=training)
    x = activation_fn(x)

    # Project input if necessary
    with tf.variable_scope('proj'):
      if (strides > 1) or (out_filters != x.shape[-1]):
        x_shortcut = tf.layers.conv2d(x, filters=out_filters, kernel_size=1,
                                      strides=strides,
                                      kernel_regularizer=kernel_regularizer,
                                      use_bias=False,
                                      padding='VALID')

    # Note that ResNeXt doubles middle's channel count!
    middle_filters = filters // 4 if groups == 1 else filters // 2

    # First convolution
    with tf.variable_scope('a'):
      # Note, that unlike original Resnet paper we never use stride in the first
      # convolution. Instead, we apply stride in the second convolution. The
      # reason is that the first convolution has kernel of size 1x1, which
      # results in information loss when combined with stride bigger than one.
      x = tf.layers.conv2d(x, filters=middle_filters,
                           kernel_size=1,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='SAME')

    # Second convolution
    with tf.variable_scope('b'):
      x = normalization_fn(x, training=training)
      x = activation_fn(x)
      # Note, that padding depends on the dilation rate.
      x = fixed_padding(x, kernel_size=3 + 2 * (dilation - 1))
      x = maybe_group_conv(x, filters=middle_filters, groups=groups,
                           strides=strides,
                           kernel_size=3,
                           dilation_rate=dilation,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='VALID')

    # Third convolution
    with tf.variable_scope('c'):
      x = normalization_fn(x, training=training)
      x = activation_fn(x)
      x = tf.layers.conv2d(x, filters=out_filters,
                           kernel_size=1,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=False,
                           padding='SAME')

  if no_shortcut:
    return x
  else:
    return x + x_shortcut


def resnet(x,  # pylint: disable=missing-docstring
           is_training,
           num_classes=1000,
           filters_factor=4,
           weight_decay=1e-4,
           include_root_block=True,
           root_conv_size=7, root_conv_stride=2,
           root_pool_size=3, root_pool_stride=2,
           activation_fn=tf.nn.relu,
           last_relu=True,
           normalization_fn=batch_norm,
           strides=(2, 2, 2),
           dilations=(1, 1, 1, 1),
           num_layers=(3, 4, 6, 3),
           global_pool=True,
           unit='bottleneck',
           mode='v2',
           representation_size=None,
           spatial_squeeze=True,
           groups=1):

  unit_kw = {
      'activation_fn': activation_fn,
      'normalization_fn': normalization_fn,
      'training': is_training,
  }

  mult = 1
  if unit == 'bottleneck':
    unit = bottleneck_v2 if mode == 'v2' else bottleneck_v1
    mult = 4
    unit_kw['groups'] = groups
  elif unit == 'resblock':
    unit = resblock_v2 if mode == 'v2' else resblock_v1
    assert groups == 1, 'Groups not supported in "resblock".'
  else:
    raise ValueError('Unknown resnet unit: %s' % unit)

  strides = list(strides)[::-1]
  dilations = list(dilations)[::-1]
  num_layers = list(num_layers)[::-1]

  end_points = {}

  filters = 16 * filters_factor

  kernel_regularizer = l2_regularizer(scale=weight_decay)
  unit_kw['kernel_regularizer'] = kernel_regularizer

  if include_root_block:
    with tf.variable_scope('root_block'):
      x = fixed_padding(x, kernel_size=root_conv_size)
      x = tf.layers.conv2d(x, filters=filters,
                           kernel_size=root_conv_size,
                           strides=root_conv_stride,
                           padding='VALID', use_bias=False,
                           kernel_regularizer=kernel_regularizer)

      if mode == 'v1':
        x = normalization_fn(x, training=is_training)
        x = activation_fn(x)

      x = fixed_padding(x, kernel_size=root_pool_size)
      x = tf.layers.max_pooling2d(x, pool_size=root_pool_size,
                                  strides=root_pool_stride, padding='VALID')
      end_points['after_root'] = x

  with tf.variable_scope('block1'):
    filters *= mult
    unit_kw.update({'dilation': dilations.pop()})
    for i in range(num_layers.pop()):
      x = unit(x, filters, strides=1, name='unit%d' % (i + 1), **unit_kw)
    end_points['block1'] = x

  with tf.variable_scope('block2'):
    filters *= 2
    unit_kw.update({'dilation': dilations.pop()})
    x = unit(x, filters, strides=strides.pop(), name='unit1', **unit_kw)
    for i in range(1, num_layers.pop()):
      x = unit(x, filters, strides=1, name='unit%d' % (i + 1), **unit_kw)
    end_points['block2'] = x

  with tf.variable_scope('block3'):
    filters *= 2
    unit_kw.update({'dilation': dilations.pop()})
    x = unit(x, filters, strides=strides.pop(), name='unit1', **unit_kw)
    for i in range(1, num_layers.pop()):
      x = unit(x, filters, strides=1, name='unit%d' % (i + 1), **unit_kw)
    end_points['block3'] = x

  with tf.variable_scope('block4'):
    filters *= 2
    unit_kw.update({'dilation': dilations.pop()})
    nlayers = num_layers.pop()
    if nlayers > 1:
      x = unit(x, filters, strides=strides.pop(), name='unit1', **unit_kw)
      for i in range(1, nlayers - 1):
        x = unit(x, filters, strides=1, name='unit%d' % (i + 1), **unit_kw)
      # representation_size is the number of dimensions of the final output,
      # right before (and after) the global average pooling. By default, it's
      # simply the number of filters that we got there with the given
      # architecture, but optionally we may want to control it explicitly and
      # independently.
      x = unit(x, representation_size or filters, strides=1,
               name='unit%d' % nlayers, **unit_kw)
    else:
      # But in the case of just one block, do everything in that block.
      x = unit(x, representation_size or filters, strides=strides.pop(),
               name='unit1', **unit_kw)
    end_points['block4'] = x

  if (mode == 'v1') and (not last_relu):
    raise ValueError('last_relu should be set to True in the v1 mode.')

  if mode == 'v2':
    x = normalization_fn(x, training=is_training)
    if last_relu:
      x = activation_fn(x)

  if global_pool:
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    end_points['pre_logits'] = tf.squeeze(x, [1, 2]) if spatial_squeeze else x
  else:
    end_points['pre_logits'] = x

  if num_classes:
    with tf.variable_scope('head'):
      logits = tf.layers.conv2d(x, filters=num_classes,
                                kernel_size=1,
                                kernel_regularizer=kernel_regularizer)
      if global_pool and spatial_squeeze:
        logits = tf.squeeze(logits, [1, 2])
    end_points['logits'] = logits
    return logits, end_points
  else:
    return end_points['pre_logits'], end_points

resnet18 = functools.partial(resnet, num_layers=(2, 2, 2, 2),
                             unit='resblock')
resnet34 = functools.partial(resnet, num_layers=(3, 4, 6, 3),
                             unit='resblock')
resnet50 = functools.partial(resnet, num_layers=(3, 4, 6, 3),
                             unit='bottleneck')
resnet101 = functools.partial(resnet, num_layers=(3, 4, 23, 3),
                              unit='bottleneck')
resnet152 = functools.partial(resnet, num_layers=(3, 8, 36, 3),
                              unit='bottleneck')

# Experimental code ########################################
# "Reversible" resnet ######################################


# Invertible residual block as outlined in https://arxiv.org/abs/1707.04585
def bottleneck_rev(x, training,  # pylint: disable=missing-docstring
                   activation_fn=tf.nn.relu,
                   normalization_fn=batch_norm,
                   dilation=1,
                   kernel_regularizer=None,
                   simple=False,
                   unit='bottleneck',
                   name='unit_rev'):

  if unit == 'bottleneck':
    unit = bottleneck_v2
  elif unit == 'resblock':
    unit = resblock_v2
  else:
    raise ValueError('Unknown resnet unit: %s' % unit)

  x1, x2 = tf.split(x, 2, 3)

  with tf.variable_scope(name):
    y1 = x1 + unit(x2, x2.shape[-1], training,
                   strides=1,
                   activation_fn=activation_fn,
                   normalization_fn=normalization_fn,
                   kernel_regularizer=kernel_regularizer,
                   dilation=dilation,
                   no_shortcut=True,
                   name='unit1')
    if simple:
      y2 = x2
      # The swap of 'y' parts is intentional here as 'simple' block processes
      # only one part of the input. Thus, without this swap only one part will
      # be repeatedly processed. This is a standard practice from e.g. RealNVP
      # or iRevnet papers.
      return tf.concat([y2, y1], axis=3)
    else:
      y2 = x2 + unit(y1, y1.shape[-1], training,
                     strides=1,
                     activation_fn=activation_fn,
                     normalization_fn=normalization_fn,
                     kernel_regularizer=kernel_regularizer,
                     dilation=dilation,
                     no_shortcut=True,
                     name='unit2')

      return tf.concat([y1, y2], axis=3)


# This operation is not strictly speaking invertible. However, realistically,
# it always preserves large amount of information and can be inverted up to
# some error.
def pool_and_double_channels(x, stride):
  if stride > 1:
    x = tf.layers.average_pooling2d(x, pool_size=stride, strides=stride,
                                    padding='SAME')
  return tf.pad(x, [[0, 0], [0, 0], [0, 0],
                    [x.shape[3] // 2, x.shape[3] // 2]])


def revnet(x,  # pylint: disable=missing-docstring
           is_training,
           num_classes=1000,
           filters_factor=4,
           weight_decay=1e-4,
           include_root_block=True,
           root_conv_size=7, root_conv_stride=2,
           root_pool_size=3, root_pool_stride=2,
           strides=(2, 2, 2),
           dilations=(1, 1, 1, 1),
           num_layers=(3, 4, 6, 3),
           global_pool=True,
           activation_fn=tf.nn.relu,
           normalization_fn=batch_norm,
           last_relu=False,
           mode='v2',
           inside_unit='bottleneck',
           representation_size=None,
           regularize_last_proj=False,
           spatial_squeeze=True):

  del mode  # unused parameter, exists for compatibility with resnet function

  # Use simple block. Note that two consecutive simple blocks are equivalent
  # to the normal block
  unit = functools.partial(bottleneck_rev, simple=True)

  mult = 1
  if inside_unit == 'bottleneck':
    mult = 4

  strides = list(strides)[::-1]
  dilations = list(dilations)[::-1]
  num_layers = list(num_layers)[::-1]

  end_points = {}

  filters = 16 * filters_factor

  kernel_regularizer = l2_regularizer(scale=weight_decay)

  # First convolution serves as random projection in order to increase number
  # of channels. It is not possible to skip it.

  with tf.variable_scope('root_block'):
    x = fixed_padding(x, kernel_size=root_conv_size)
    x = tf.layers.conv2d(x, filters=filters * mult,
                         kernel_size=root_conv_size,
                         strides=root_conv_stride,
                         padding='VALID', use_bias=False,
                         kernel_regularizer=None)

    if include_root_block:
      x = fixed_padding(x, kernel_size=root_pool_size)
      x = tf.layers.max_pooling2d(
          x, pool_size=root_pool_size, strides=root_pool_stride,
          padding='VALID')

  end_points['after_root'] = x

  params = {'activation_fn': activation_fn,
            'normalization_fn': normalization_fn,
            'training': is_training,
            'kernel_regularizer': kernel_regularizer,
            'unit': inside_unit
           }

  with tf.variable_scope('block1'):
    params.update({'dilation': dilations.pop()})
    for i in range(num_layers.pop()):
      x = unit(x, name='block%i' % i, **params)
    x = pool_and_double_channels(x, strides.pop())
    end_points['block1'] = x

  with tf.variable_scope('block2'):
    params.update({'dilation': dilations.pop()})
    for i in range(num_layers.pop()):
      x = unit(x, name='block%i' % i, **params)
    x = pool_and_double_channels(x, strides.pop())
    end_points['block2'] = x

  with tf.variable_scope('block3'):
    params.update({'dilation': dilations.pop()})
    for i in range(num_layers.pop()):
      x = unit(x, name='block%i' % i, **params)
    x = pool_and_double_channels(x, strides.pop())
    end_points['block3'] = x

  with tf.variable_scope('block4'):
    params.update({'dilation': dilations.pop()})
    for i in range(num_layers.pop()):
      x = unit(x, name='block%i' % i, **params)
    end_points['block4'] = x

  if representation_size:
    with tf.variable_scope('resize'):
      x = activation_fn(normalization_fn(x, training=is_training))
      kern_reg = kernel_regularizer if regularize_last_proj else None
      x = tf.layers.conv2d(x, filters=representation_size, use_bias=False,
                           kernel_size=1, kernel_regularizer=kern_reg)

  with tf.variable_scope('head'):
    x = normalization_fn(x, training=is_training)

    if last_relu:
      x = activation_fn(x)

    if global_pool:
      x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
      end_points['pre_logits'] = tf.squeeze(x, [1, 2]) if spatial_squeeze else x
    else:
      end_points['pre_logits'] = x

    if num_classes:
      logits = tf.layers.conv2d(x, filters=num_classes,
                                kernel_size=1,
                                kernel_regularizer=kernel_regularizer)
      if global_pool and spatial_squeeze:
        logits = tf.squeeze(logits, [1, 2])
      end_points['logits'] = logits
      return logits, end_points
    else:
      return end_points['pre_logits'], end_points


revnet18 = functools.partial(revnet, num_layers=(2, 2, 2, 2),
                             inside_unit='resblock')
revnet34 = functools.partial(revnet, num_layers=(3, 4, 6, 3),
                             inside_unit='resblock')
revnet50 = functools.partial(revnet, num_layers=(3, 4, 6, 3),
                             inside_unit='bottleneck')
revnet101 = functools.partial(revnet, num_layers=(3, 4, 23, 3),
                              inside_unit='bottleneck')
revnet152 = functools.partial(revnet, num_layers=(3, 8, 36, 3),
                              inside_unit='bottleneck')


# Even more experimental code ########################################
# Fully-Reversible resnet aka iRevnet ################################


# Reimplementation of iRevnet from: https://openreview.net/forum?id=HJsjkMb0Z
def irevnet300(x,  # pylint: disable=missing-docstring
               is_training,
               num_classes=1000,
               weight_decay=1e-4,
               activation_fn=tf.nn.relu,
               normalization_fn=batch_norm):

  unit = functools.partial(bottleneck_rev, simple=True)

  end_points = {}

  kernel_regularizer = l2_regularizer(scale=weight_decay)

  # Do inv. pooling first
  x = tf.space_to_depth(x, 2)

  layer_counts = [1, 6, 16, 72, 5]
  params = {'activation_fn': activation_fn,
            'normalization_fn': normalization_fn,
            'training': is_training,
            'kernel_regularizer': kernel_regularizer
           }

  for num_block, layer_count in enumerate(layer_counts):
    for _ in range(layer_count):
      x = unit(x, **params)
    end_points['block%i' % (num_block + 1)] = x
    if num_block < (len(layer_counts) - 1):
      x = tf.space_to_depth(x, 2)

  x = normalization_fn(x, training=is_training)
  end_points['last_invertible'] = x

  # Non-invertible part starts here
  x = activation_fn(x)
  x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

  end_points['pre_logits'] = tf.squeeze(x, [1, 2])
  logits = tf.squeeze(tf.layers.conv2d(x, filters=num_classes,
                                       kernel_size=1,
                                       kernel_regularizer=kernel_regularizer),
                      [1, 2])

  end_points['logits'] = logits

  return logits, end_points
