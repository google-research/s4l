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

"""Utilities for implementing Virtual Adversarial Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def virtual_adversarial_perturbation_direction(
    x, logits, net, xi=1e-6, num_power_method_iters=1):
  """Approximate a perturbation p that maximizes KL[f(x+p)||f(x)] to 2nd order.

  Args:
    x: The input batch around which to find the adversarial perturbation
    logits: The logits batch predicted for x. Should be equal to net(x), but
        we avoid an extra call to net since presumably the prediction is
        already computed.
    net: Function mapping tensor inputs to tensor logits
    xi: Small constant used for finite difference estimate of Hessian vector
        products. Theoretically, the smaller the better, but practically,
        number stability becomes a problem. 1e-6 seems to work well.
    num_power_method_iters: Number of iterations of the power method when
        estimating the eigenvector of the top eigenvalue of the Hessian of
        the KL being approximated.

  Returns:
    A unit norm perturbation vector.
  """
  # Initialize a random perturbation
  perturbation = tf.random_normal(shape=tf.shape(x))

  # Run the power method to find an approximation of the top eigenvalue
  # of the Hessian of KL[softmax(net(x+perturbation)) || softmax(net(x))]
  # w.r.t. perturbation.
  #
  # (One iteration has worked for other datasets but maybe we will need more)
  for _ in range(num_power_method_iters):
    # Approximate Hd where H is the Hessian of KL[f(x+p)||f(x)] w.r.t. p.
    # Use a finite differences approximation to Hd, as described in
    # https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/.
    perturbation = xi * normalize(perturbation)
    perturbed_logits = net(x + perturbation)
    divergence = kl_divergence_from_logits(perturbed_logits, logits)
    hessian_perturbation_product = tf.gradients(
        tf.reduce_mean(divergence), [perturbation])[0]
    perturbation = tf.stop_gradient(hessian_perturbation_product)

  return normalize(perturbation)


def kl_divergence_from_logits(logits1, logits2):
  """Compute KL[softmax(logits1) || softmax(logits2)]."""
  p1 = tf.nn.softmax(logits1)
  p1logp1 = tf.reduce_sum(p1 * tf.math.log_softmax(logits1), axis=1)
  p1logp2 = tf.reduce_sum(p1 * tf.math.log_softmax(logits2), axis=1)
  return p1logp1 - p1logp2


def normalize(v):
  return tf.math.l2_normalize(v, epsilon=1e-12, axis=[1, 2, 3])
