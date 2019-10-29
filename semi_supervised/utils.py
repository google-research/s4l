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

"""Functions shared across semi-supervised models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.utils as model_utils
import utils


def apply_model_semi(img_unsup, img_sup, is_training, outputs, **kw):
  """Passes `img_unsup` and/or `img_sup` through the model.

  Args:
    img_unsup: The unsupervised input, could be None.
    img_sup: The supervised input, could be None.
    is_training: Train or test mode?
    outputs: A dict-like of {name: number} defining the desired output layers
      of the network. A linear layer with `number` outputs is added for each
      entry, with the given `name`.
    **kw: Extra keyword-args to be passed to `net()`.

  Returns:
    end_points: A dictionary of {name: tensor} mappings, partially dependent on
      which network is used. Additional entries are present for all entries in
      `outputs` and named accordingly.
      If both `img_unsup` and `img_sup` is given, every entry in `end_points`
      comes with two additional entries suffixed by "_unsup" and "_sup", which
      corresponds to the parts corresponding to the respective inputs.
  """
  # If both inputs are given, we concat them along the batch dimension.
  if img_unsup is not None and img_sup is not None:
    img_all = tf.concat([img_unsup, img_sup], axis=0)
  elif img_unsup is not None:
    img_all, split_idx = img_unsup, None
  elif img_sup is not None:
    img_all, split_idx = img_sup, None
  else:
    assert False, 'Either `img_unsup` or `img_sup` needs to be passed.'

  net = model_utils.get_net()
  _, end_points = net(img_all, is_training, spatial_squeeze=False, **kw)

  # TODO(xzhai): Try adding batch norm here.
  pre_logits = end_points['pre_logits']

  for name, nout in outputs.items():
    end_points[name] = utils.linear(pre_logits, nout, name)

  # Now, if both inputs were given, here we loop over all end_points, including
  # the final output we're usually interested in, and split them for
  # conveninece of the caller.
  if img_unsup is not None and img_sup is not None:
    split_idx = img_unsup.get_shape().as_list()[0]
    for name, val in end_points.copy().items():
      end_points[name + '_unsup'] = val[:split_idx]
      end_points[name + '_sup'] = val[split_idx:]

  elif img_unsup is not None:
    for name, val in end_points.copy().items():
      end_points[name + '_unsup'] = val

  elif img_sup is not None:
    for name, val in end_points.copy().items():
      end_points[name + '_sup'] = val

  else:
    raise ValueError('You must set at least one of {img_unsup, img_unsup}.')

  return end_points
