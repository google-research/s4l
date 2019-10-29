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

"""Helper functions for NN models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import absl.flags as flags

import models.resnet as resnet
import models.vggnet as vggnet

FLAGS = flags.FLAGS


def get_net(num_classes=None):  # pylint: disable=missing-docstring
  architecture = FLAGS.architecture
  task = FLAGS.task

  if "resnet18" in architecture:
    net = resnet.resnet18
  elif "resnet34" in architecture:
    net = resnet.resnet34
  elif "resnet50" in architecture or "resnext50" in architecture:
    net = resnet.resnet50
  elif "resnet101" in architecture or "resnext101" in architecture:
    net = resnet.resnet101
  elif "resnet152" in architecture or "resnext152" in architecture:
    net = resnet.resnet152
  elif "revnet18" in architecture:
    net = resnet.revnet18
  elif "revnet34" in architecture:
    net = resnet.revnet34
  elif "revnet50" in architecture:
    net = resnet.revnet50
  elif "revnet101" in architecture:
    net = resnet.revnet101
  elif "revnet152" in architecture:
    net = resnet.revnet152
  else:
    raise ValueError("Unsupported architecture: %s" % architecture)

  net = functools.partial(net, filters_factor=FLAGS.filters_factor, mode="v2")

  if "resnext" in architecture:
    net = functools.partial(net, groups=32)

  # Few things that are common across all models.
  net = functools.partial(
      net, num_classes=num_classes,
      weight_decay=FLAGS.weight_decay)

  return net
