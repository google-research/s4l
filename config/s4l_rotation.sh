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
  --task rotation \
  --dataset imagenet \
  --train_split trainval \
  --val_split test \
  --batch_size 256 \
  --eval_batch_size 16 \
  --architecture resnet50v2 \
  --filters_factor 4 \
  --weight_decay 1e-3 \
  --preprocessing 'inception_crop|resize(224)|flip_lr|-1_to_1|rotate' \
  --preprocessing_eval 'resize_small(256)|crop(224)|-1_to_1|rotate' \
  --sup_preprocessing 'copy_label|inception_crop|resize(224)|flip_lr|-1_to_1|rotate' \
  --sup_preprocessing_eval 'copy_label|resize_small(256)|crop(224)|-1_to_1|rotate' \
  --lr 0.03 \
  --lr_scale_batch_size 0 \
  --schedule '5,140,160,180,200' \
  --serving_input_shape 'None,224,224,3' \
  --enable_sup_data 1 \
  --filename_list_template 'label_map_count_{}_index_0' \
  --num_supervised_examples 128000 \
  --unsup_batch_mult 1 \
  --rot_loss_weight 1.0 \
  --rot_loss_sup 1 \
  --rot_loss_unsup 1 \
  --save_checkpoints_steps 5000 \
  --eval_timeout_mins 600 \
  "$@"
