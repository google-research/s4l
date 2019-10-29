#!/bin/sh
#
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

python main.py \
  --task supervised \
  --dataset imagenet \
  --cache_dataset \
  --train_split trainval \
  --val_split test \
  --batch_size 256 \
  --eval_batch_size 80 \
  --filename_list_template 'label_map_count_{}_index_0' \
  --num_supervised_examples 13000 \
  --architecture resnet50v2 \
  --filters_factor 4 \
  --weight_decay 0.01 \
  --lr 0.01 \
  --lr_scale_batch_size 256 \
  --save_checkpoints_steps 5000 \
  --eval_timeout_mins 60 \
  --schedule '5,700,800,900,1000' \
  --sup_preprocessing 'inception_crop|resize(224)|flip_lr|hsvnoise|-1_to_1' \
  --sup_preprocessing_eval 'resize_small(256)|crop(224)|-1_to_1' \
  --unsup_batch_mult 0.03125 \
  --preprocessing 'resize(224)|-1_to_1' \
  --preprocessing_eval 'resize(224)|-1_to_1' \
  "$@"

# NOTE: enabling `--cache_dataset` gives a 10x speed-up here,
#       and does not make cloud TPU crash.
