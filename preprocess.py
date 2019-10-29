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

# pylint: disable=missing-docstring
"""Preprocessing methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np

import tensorflow.compat.v1 as tf

import inception_preprocessing as inception_pp
import utils


COLOR_PALETTE_PATH = ("/cns/vz-d/home/dune/representation/"
                      "color_palette.npy")


def crop(image, is_training, crop_size):
  h, w, c = crop_size[0], crop_size[1], image.shape[-1]

  if is_training:
    return tf.random_crop(image, [h, w, c])
  else:
    # Central crop for now. (See Table 5 in Appendix of
    # https://arxiv.org/pdf/1703.07737.pdf for why)
    dy = (tf.shape(image)[0] - h)//2
    dx = (tf.shape(image)[1] - w)//2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)


def get_inception_preprocess(is_training, im_size):
  def _inception_preprocess(data):
    data["image"] = inception_pp.preprocess_image(
        data["image"], im_size[0], im_size[1], is_training,
        add_image_summaries=False)
    return data
  return _inception_preprocess


def get_resize_small(smaller_size):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio."""
  def _resize_small_pp(data):
    image = data["image"]
    # A single image: HWC
    # A batch of images: BHWC
    h, w = tf.shape(image)[-3], tf.shape(image)[-2]

    # Figure out the necessary h/w.
    ratio = tf.to_float(smaller_size) / tf.to_float(tf.minimum(h, w))
    h = tf.to_int32(tf.round(tf.to_float(h) * ratio))
    w = tf.to_int32(tf.round(tf.to_float(w) * ratio))

    # NOTE: use align_corners=False for AREA resize, but True for Bilinear.
    # See also https://github.com/tensorflow/tensorflow/issues/6720
    static_rank = len(image.get_shape().as_list())
    if static_rank == 3:  # A single image: HWC
      data["image"] = tf.image.resize_area(image[None], [h, w])[0]
    elif static_rank == 4:  # A batch of images: BHWC
      data["image"] = tf.image.resize_area(image, [h, w])
    return data
  return _resize_small_pp


def get_multi_crop(crop_size):
  """Get multiple crops for test."""

  def _crop(image, offset, size):
    return tf.image.crop_to_bounding_box(image, offset[0], offset[1], size[0],
                                         size[1])

  def _multi_crop_pp(data):
    image = data["image"]
    h, w, c = crop_size[0], crop_size[1], image.shape[-1]

    tl = (0, 0)
    tr = (0, tf.shape(image)[1] - w)
    bl = (tf.shape(image)[0] - h, 0)
    br = (tf.shape(image)[0] - h, tf.shape(image)[1] - w)
    c = ((tf.shape(image)[0] - h) // 2, (tf.shape(image)[1] - w) // 2)
    data["image"] = tf.stack([
        _crop(image, c, crop_size),
        _crop(image, tl, crop_size),
        _crop(image, tr, crop_size),
        _crop(image, bl, crop_size),
        _crop(image, br, crop_size)
    ])
    return data
  return _multi_crop_pp


def get_crop(is_training, crop_size):
  """Returns a random (or central at test-time) crop of `crop_size`."""
  def _crop_pp(data):
    crop_fn = functools.partial(
        crop, is_training=is_training, crop_size=crop_size)
    data["image"] = utils.tf_apply_to_image_or_images(crop_fn, data["image"])

    return data
  return _crop_pp


def inception_crop(image, **kw):
  """Perform an "inception crop", without resize."""
  begin, size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image), tf.zeros([0, 0, 4], tf.float32),
      use_image_if_no_bounding_boxes=True, **kw)
  crop = tf.slice(image, begin, size)
  # Unfortunately, the above operation loses the depth-dimension. So we need
  # to Restore it the manual way.
  crop.set_shape([None, None, image.shape[-1]])
  return crop


def get_inception_crop(is_training, **kw):
  # kw of interest are: aspect_ratio_range, area_range.
  # Note that image is not resized yet here.
  def _inception_crop_pp(data):
    if is_training:
      data["image"] = inception_crop(data["image"], **kw)
    else:
      # TODO(lbeyer): Maybe do 87.5%-crop in test-mode by default?
      tf.logging.warn("inception_crop pre-processing keeps the full image in "
                      "eval mode for now. Contact lbeyer@ with your use-case "
                      "and propose a reasonable default behaviour.")
    return data
  return _inception_crop_pp


def get_random_flip_lr(is_training):
  def _random_flip_lr_pp(data):
    if is_training:
      data["image"] = utils.tf_apply_to_image_or_images(
          tf.image.random_flip_left_right, data["image"])
    return data
  return _random_flip_lr_pp


def get_resize_preprocess(fn_args, is_training):
  # This checks if the string "randomize_method" is present anywhere in the
  # args. If it is, during training, enable randomization, but not during test.
  # That's so that a call can look like `resize(256, randomize_method)` or
  # `resize(randomize_method, 256, 128)` and they all work as expected.
  try:
    fn_args.remove("randomize_method")
    randomize_resize_method = is_training
  except ValueError:
    randomize_resize_method = False
  im_size = utils.str2intlist(fn_args, 2)

  def _resize(image, method, align_corners):

    def _process():
      # The resized_images are of type float32 and might fall outside of range
      # [0, 255].
      resized = tf.cast(
          tf.image.resize_images(
              image, im_size, method, align_corners=align_corners),
          dtype=tf.float32)
      return resized

    return _process

  def _resize_pp(data):
    im = data["image"]

    if randomize_resize_method:
      # pick random resizing method
      r = tf.random_uniform([], 0, 3, dtype=tf.int32)
      im = tf.case({
          tf.equal(r, tf.cast(0, r.dtype)):
              _resize(im, tf.image.ResizeMethod.BILINEAR, True),
          tf.equal(r, tf.cast(1, r.dtype)):
              _resize(im, tf.image.ResizeMethod.NEAREST_NEIGHBOR, True),
          tf.equal(r, tf.cast(2, r.dtype)):
              _resize(im, tf.image.ResizeMethod.BICUBIC, True),
          # NOTE: use align_corners=False for AREA resize, but True for the
          # others. See https://github.com/tensorflow/tensorflow/issues/6720
          tf.equal(r, tf.cast(3, r.dtype)):
              _resize(im, tf.image.ResizeMethod.AREA, False),
      })
    else:
      im = tf.image.resize_images(im, im_size)
    data["image"] = im
    return data

  return _resize_pp


def get_rotate_preprocess(create_labels=True):
  """Returns a function that does 90deg rotations and sets according labels."""

  def _four_rots(img):
    # We use our own instead of tf.image.rot90 because that one broke
    # internally shortly before deadline...
    return tf.stack([
        img,
        tf.transpose(tf.reverse_v2(img, [1]), [1, 0, 2]),
        tf.reverse_v2(img, [0, 1]),
        tf.reverse_v2(tf.transpose(img, [1, 0, 2]), [1]),
    ])

  def _rotate_pp(data):
    # Create labels in the same structure as images!
    if create_labels:
      data["label"] = utils.tf_apply_to_image_or_images(
          lambda _: tf.constant([0, 1, 2, 3]), data["image"], dtype=tf.int32)
    data["image"] = utils.tf_apply_to_image_or_images(_four_rots, data["image"])
    return data

  return _rotate_pp


def get_copy_label_preprocess(new_name):
  """Returns a function that copies labels."""

  def _copy_label_pp(data):
    data[new_name] = data["label"]
    return data

  return _copy_label_pp


def get_value_range_preprocess(vmin=-1, vmax=1, dtype=tf.float32):
  """Returns a function that sends [0,255] image to [vmin,vmax]."""

  def _value_range_pp(data):
    img = tf.cast(data["image"], dtype)
    img = vmin + (img / tf.constant(255.0, dtype)) * (vmax - vmin)
    data["image"] = img
    return data
  return _value_range_pp


def get_hsvnoise_preprocess(sv_pow=(-2.0, 2.0), sv_mul=(-0.5, 0.5),
                            sv_add=(-0.1, 0.1), h_add=(-0.1, 0.1)):
  """Returns a function that randomises HSV similarly to the Exemplar paper.

  Requires the input to still be in [0-255] range.
  Transforms the input to HSV, applies rnd(mul)*S**(2**rnd(pow)) + rnd(add) to
  the S and V channels independently, and H + rnd(add) to the H channel, then
  converts back to RGB in float [0-255].

  Args:
    sv_pow: The min/max powers of two to which to take S/V.
    sv_mul: The min/max powers of two with which to scale S/V.
    sv_add: The min/max shift of S/V.
    h_add: The min/max shift of hue.

  Returns:
    A function applying random HSV augmentation to its input.
  """
  rnd = lambda *a: tf.random.uniform((), *a)
  rnd2 = lambda *a: tf.random.uniform((2,), *a)

  def _hsvnoise(rgb):
    hsv = tf.image.rgb_to_hsv(rgb / 255.0)  # Needs [0 1] input.
    h, sv = hsv[..., :1], hsv[..., 1:]
    h = tf.floormod(1. + h + rnd(*h_add), 1.)  # color cycle.
    pow_, mul, add = 2.0**rnd2(*sv_pow), 2.0**rnd2(*sv_mul), rnd2(*sv_add)
    sv = sv**pow_ * mul + add
    hsv = tf.clip_by_value(tf.concat([h, sv], axis=-1), 0, 1)
    return tf.image.hsv_to_rgb(hsv) * 255.0

  def _hsvnoise_pp(data):
    data["image"] = utils.tf_apply_to_image_or_images(_hsvnoise, data["image"])
    return data

  return _hsvnoise_pp


def get_standardize_preprocess():
  def _standardize_pp(data):
    data["image"] = tf.image.per_image_standardization(data["image"])
    return data
  return _standardize_pp


def get_inception_crop_patches(resize_size, num_patches):

  def _inception_crop_patches(img):
    return tf.stack([
        tf.image.resize(inception_crop(img), resize_size)
        for _ in range(num_patches)
    ])

  def _inception_crop_patches_pp(data):
    # The output becomes float32 because of the tf.image.resize.
    data["image"] = utils.tf_apply_to_image_or_images(
        _inception_crop_patches, data["image"], dtype=tf.float32)
    return data

  return _inception_crop_patches_pp


def get_inception_preprocess_patches(is_training, resize_size, num_patches):

  def _inception_preprocess_patches(data):
    patches = []
    for _ in range(num_patches):
      patches.append(
          inception_pp.preprocess_image(
              data["image"],
              resize_size[0],
              resize_size[1],
              is_training,
              add_image_summaries=False))
    patches = tf.stack(patches)
    data["image"] = patches
    return data

  return _inception_preprocess_patches


def get_to_gray_preprocess(grayscale_probability):

  def _to_gray(image):
    # Transform to grayscale by taking the mean of RGB.
    return tf.tile(tf.reduce_mean(image, axis=2, keepdims=True), [1, 1, 3])

  def _to_gray_pp(data):
    data["image"] = utils.tf_apply_to_image_or_images(
        lambda img: utils.tf_apply_with_probability(  # pylint:disable=g-long-lambda
            grayscale_probability, _to_gray, img),
        data["image"])
    return data

  return _to_gray_pp


def get_preprocess_fn(pp_pipeline, is_training):
  """Returns preprocessing function.

  The minilanguage is as follows:

    fn1|fn2(arg, arg2,...)|fn3(key1=val1, key2=val2, ...)|...

  And describes the successive application of the various `fn`s to the input,
  where each function can optionally have one or more arguments, which are
  either positional or key/value, as dictated by the `fn`.

  Args:
    pp_pipeline: A string describing the pre-processing pipeline.
    is_training: Whether this should be run in train or eval mode.
  Returns:
    preprocessing function

  Raises:
    ValueError: if preprocessing function name is unknown
  """

  def _fn(data):
    def parse_fn(fn_call):
      """Parses the fn(arg1,arg2,...) and fn(a=1,b=2) structures.

      Args:
        fn_call: string, the function call as a string.

      Returns:
        The function name, and either a list (possibly empty) or a dict,
        depending on the syntax of the function call.
      """
      if "(" in fn_call:
        fn_name, fn_args = fn_call.split("(")
        if "=" in fn_args:
          fn_args = dict(kv.split("=") for kv in fn_args[:-1].split(","))
          return fn_name, {k.strip(): v.strip() for k, v in fn_args.items()}
        else:
          return fn_name, [a.strip() for a in fn_args[:-1].split(",")]
      else:
        return fn_call, []

    def get(list_, index, default):
      """Return element at `index` in `list_` or the `default`."""
      try:
        return list_[index]
      except IndexError:
        return default

    def expand(fn_name, args):
      if fn_name == "plain_preprocess":
        yield lambda x: x
      elif fn_name == "0_to_1":
        yield get_value_range_preprocess(0, 1)
      elif fn_name == "-1_to_1":
        yield get_value_range_preprocess(-1, 1)
      elif fn_name == "value_range":
        yield get_value_range_preprocess(*map(float, args))
      elif fn_name == "resize":
        yield get_resize_preprocess(args, is_training)
      elif fn_name == "resize_small":
        yield get_resize_small(int(args[0]))
      elif fn_name == "crop":
        yield get_crop(is_training, utils.str2intlist(args, 2))
      elif fn_name == "central_crop":
        yield get_crop(False, utils.str2intlist(args, 2))
      elif fn_name == "multi_crop":
        yield get_multi_crop(utils.str2intlist(args, 2))
      elif fn_name == "inception_crop":
        yield get_inception_crop(is_training)
      elif fn_name == "flip_lr":
        yield get_random_flip_lr(is_training)
      elif fn_name == "hsvnoise":
        # TODO(lbeyer): expose the parameters? Or maybe just a scale parameter?
        yield get_hsvnoise_preprocess(*args)
      elif fn_name == "crop_inception_preprocess_patches":
        npatch = int(args[0])
        size = utils.str2intlist(args[1:], 2)
        yield get_inception_preprocess_patches(is_training, size, npatch)
      elif fn_name == "crop_inception_patches":
        npatch = int(args[0])
        size = utils.str2intlist(args[1:], 2)
        yield get_inception_crop_patches(size, npatch)
      elif fn_name == "to_gray":
        yield get_to_gray_preprocess(float(get(args, 0, 1.0)))
      elif fn_name == "standardize":
        yield get_standardize_preprocess()
      elif fn_name == "rotate":
        yield get_rotate_preprocess()
      elif fn_name == "copy_label":
        yield get_copy_label_preprocess(get(args, 0, "copy_label"))

      # Below this line specific combos decomposed.
      # It would be nice to move them to the configs at some point.

      elif fn_name == "inception_preprocess":
        yield get_inception_preprocess(is_training, utils.str2intlist(args, 2))
      else:
        raise ValueError("Not supported preprocessing %s" % fn_name)

    # Apply all the individual steps in sequence.
    tf.logging.info("Data before pre-processing:\n%s", data)
    for fn_name in pp_pipeline.split("|"):
      for p in expand(*parse_fn(fn_name.strip())):
        data = p(data)
        tf.logging.info("Data after `%s`:\n%s", p, data)
    return data

  return _fn
