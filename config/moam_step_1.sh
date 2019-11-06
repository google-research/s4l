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
  --task rotation_vat \
  --dataset imagenet \
  --train_split trainval \
  --val_split test \
  --batch_size 128 \
  --eval_batch_size 80 \
  --filename_list_template 'label_map_count_{}_index_0' \
  --num_supervised_examples 128000 \
  --architecture resnet50v2 \
  --filters_factor 16 \
  --weight_decay 0.0002 \
  --lr 0.2 \
  --lr_scale_batch_size 256 \
  --save_checkpoints_steps 10000 \
  --eval_timeout_mins 240 \
  --schedule '10,100,150,190,200' \
  --sup_preprocessing 'copy_label|inception_crop|resize(224)|flip_lr|-1_to_1|rotate' \
  --sup_preprocessing_eval 'copy_label|resize_small(256)|crop(224)|-1_to_1|rotate' \
  --preprocessing 'inception_crop|resize(224)|flip_lr|-1_to_1|rotate' \
  --preprocessing_eval 'resize_small(256)|crop(224)|-1_to_1|rotate' \
  --apply_vat_to_labeled \
  --entmin_factor 0.3 \
  --vat_eps 32.0 \
  --sup_weight 0.3 \
  --polyak_averaging \
  --unsup_batch_mult 1 \
  --enable_sup_data 1 \
  --vat_weight 0.3 \
  "$@"

