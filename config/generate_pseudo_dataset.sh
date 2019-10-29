#!/bin/bash
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
#

# Example command line:
# config/generate_pseudo_dataset.sh \
#	--hub_module=gs://s4l-models/supervised_10p \
#	--input_glob=<Glob path to Google cloud bucket with preprocessed Imagenet Dataset> \
#	--output_folder=<Path to generated dataset> \
#	--tpu_name=<Your TPU host name>

python pseudo_labeling/generate_labels.py \
  --alsologtostderr \
  --labels_file=label_map_count_128000_index_0 \
  --target_label=all \
  --data_format=t2t \
  --batch_size=64 \
  --parallel_calls=100 \
  --representation_name=classes \
  "$@"
