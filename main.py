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

r"""TODO: Edit."""

from __future__ import absolute_import
from __future__ import division

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

import representation_lib


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', 'work-dir/', 'Where to store files.')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to use.')
flags.DEFINE_string('tpu_name', None, 'Name of the TPU node to use.')
flags.DEFINE_bool('run_eval', False, 'Run eval mode')

flags.DEFINE_string('checkpoint', None, 'TODO')
flags.DEFINE_string('path_to_initial_ckpt', None, 'TODO')
flags.DEFINE_string('vars_to_restore', '.*', 'TODO')
flags.DEFINE_integer('eval_timeout_mins', 20, 'TODO')
flags.DEFINE_bool('use_summaries', True, 'TODO')

flags.DEFINE_string('dataset', 'imagenet', 'Which dataset to use: `imagenet`')
flags.DEFINE_string('dataset_dir', None, 'Location of the dataset files.')
flags.mark_flag_as_required('dataset_dir')
flags.DEFINE_string('pseudo_label_key', None, 'Key used to retrieve pseudo labels.')
flags.DEFINE_string('original_label_key', None,
                    'Key used to retrieve original labels if available.')
flags.DEFINE_bool('cache_dataset', False, 'Whether to cache the dataset after'
                  'filtering. When using 10% of ImageNet, caching somehow makes'
                  ' cloud-TPU crash (OOM), and we need to disable caching. But'
                  ' when running locally, or when running on TPU with 1%, this'
                  ' is necessary for decent speed: 10x speed-up on 1% TPU.')

flags.DEFINE_string('filename_list_template', None, 'TODO')
flags.DEFINE_integer('num_supervised_examples', None, 'TODO')

flags.DEFINE_float('unsup_batch_mult', None, 'TODO')

flags.DEFINE_integer('enable_sup_data', 1, 'Use supervised data.')

flags.DEFINE_integer('rot_loss_sup', 1, 'Enable rotation loss for sup. data.')
flags.DEFINE_integer('rot_loss_unsup', 1, 'Enable rotation loss for unsup. data.')
flags.DEFINE_float('rot_loss_weight', 1.0, 'Weight of the rotation loss.')

flags.DEFINE_integer('triplet_loss_sup', 1, 'Enable triplet loss for sup. data.')
flags.DEFINE_integer('triplet_loss_unsup', 1, 'Enable triplet loss for unsup. data.')
flags.DEFINE_float('triplet_loss_weight', 1.0, 'Weight of the triplet loss.')

flags.DEFINE_integer('save_checkpoints_steps', 1000, 'Every how many steps '
                     'to save a checkpoint. Defaults to 1000.')

flags.DEFINE_string('serving_input_key', 'image', 'The name of the input tensor'
                    ' in the generated hub module. Just leave it at default.')
flags.DEFINE_string('serving_input_shape', 'None,None,None,3', 'The shape of '
                    'the input tensor in the stored hub module.')

flags.DEFINE_integer('random_seed', None, 'Seed to use. None is random.')

flags.DEFINE_string('task', None, 'Which pretext-task to learn from. Can be '
                    'one of `rotation`, `exemplar`, `jigsaw`, '
                    '`relative_patch_location`, `linear_eval`, `supervised`.')
flags.mark_flag_as_required('task')

flags.DEFINE_string('train_split', 'train', 'Which dataset split to train on. '
                    'Should only be `train` (default) or `trainval`.')
flags.DEFINE_string('val_split', 'val', 'Which dataset split to eval on. '
                    'Should only be `val` (default) or `test`.')
flags.DEFINE_string('test_split', None, 'Optionally evaluate the last '
                    'checkpoint on this split.')

flags.DEFINE_integer('batch_size', None, 'The global batch-size to use.')
flags.mark_flag_as_required('batch_size')

flags.DEFINE_integer('eval_batch_size', None, 'Optional different batch-size'
                     ' evaluation, defaults to the same as `batch_size`.')

flags.DEFINE_string('preprocessing', None, 'A comma-separated list of '
                    'pre-processing steps to perform on unlabeled data, '
                    'see preprocess.py.')
flags.mark_flag_as_required('preprocessing')
flags.DEFINE_string('sup_preprocessing', None, 'A comma-separated list of '
                    'pre-processing steps to perform on labeled data, '
                    'see preprocess.py.')
flags.mark_flag_as_required('sup_preprocessing')
flags.DEFINE_string('preprocessing_eval', None, 'Optionally, a different pre-'
                    'processing for the unlabelled data during evaluation.')
flags.DEFINE_string('sup_preprocessing_eval', None, 'Optionally, a separate '
                    'preprocessing for the labelled data during evaluation.')

flags.DEFINE_string('schedule', None, 'Learning-rate decay schedule.')
flags.mark_flag_as_required('schedule')

flags.DEFINE_string('architecture', None,
                    help='Which basic network architecture to use. '
                    'One of vgg19, resnet50, revnet50.')
flags.DEFINE_integer('filters_factor', 4, 'Widening factor for network '
                     'filters. For ResNet, default = 4 = vanilla ResNet.')
flags.DEFINE_float('weight_decay', 1e-4, 'Strength of weight-decay. '
                   'Defaults to 1e-4, and may be set to 0.')

flags.DEFINE_bool('polyak_averaging', False, 'If true, use polyak averaging.')

flags.DEFINE_float('lr', None, 'The base learning-rate to use for training.')
flags.mark_flag_as_required('lr')

flags.DEFINE_float('lr_scale_batch_size', None, 'The batch-size for which the '
                   'base learning-rate `lr` is defined. For batch-sizes '
                   'different from that, it is scaled linearly accordingly.'
                   'For example lr=0.1, batch_size=128, lr_scale_batch_size=32'
                   ', then actual lr=0.025.')
flags.mark_flag_as_required('lr_scale_batch_size')

flags.DEFINE_float('lr_decay_factor', 0.1, 'Factor by which to decay the '
                   'learning-rate at each decay step. Default 0.1.')

flags.DEFINE_string('optimizer', 'sgd', 'Which optimizer to use. '
                    'Only `sgd` (default) or `adam` are supported.')

flags.DEFINE_integer('triplet_embed_dim', 1000, 'Size of the embedding for the '
                     'triple loss.')


flags.DEFINE_float('sup_weight', 0.3, 'Only for MOAM: Weight of supervised loss')


# for VAT:
flags.DEFINE_float('vat_weight', 0.0, 'Weight multiplier for VAT loss')
flags.DEFINE_float('entmin_factor', 0.0, 'Weight multiplier for EntMin loss')
flags.DEFINE_float('vat_eps', 1.0, 'epsilon used for finite difference '
                   'approximation used in VAT')
flags.DEFINE_integer('vat_num_power_method_iters', 1, 'Number of power method '
                     'iterations used in VAT to approximate top eigenvalue')
flags.DEFINE_boolean('apply_vat_to_labeled', False, 'Apply VAT loss also to '
                     'labeled examples?')

def main(unused_argv):
  del unused_argv  # unused
  logging.info('workdir: %s', FLAGS.workdir)

  # Write a json file to the cns (and log) which contains the run's settings.
  if not FLAGS.run_eval and not tf.gfile.IsDirectory(FLAGS.workdir):
    tf.gfile.MakeDirs(FLAGS.workdir)

  results = representation_lib.train_and_eval()

  if results:
    logging.info('Result: %s', results)

  logging.info('I\'m done with my work, ciao!')


if __name__ == '__main__':
  app.run(main)
