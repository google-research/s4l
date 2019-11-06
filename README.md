# S4L: Self-Supervised Semi-Supervised Learning

Tensorflow implementation of experiments from
[our paper on self-supervised semi-supervised learning](http://arxiv.org/abs/1905.03670).

If you find this repository useful in your research, please consider citing:

```
@article{zhai2019s4l,
         title={S4L: Self-Supervised Semi-Supervised Learning},
         author={Zhai, Xiaohua and Avital Oliver and Kolesnikov, Alexander and Beyer, Lucas},
         journal={arXiv preprint arXiv:1901.09005},
         year={2019}
}
```

## Overview

This codebase allows to reproduce core experiments from our paper. In particular,
we release S4L-Rotation and S4L-Exemplar models along with our reimplementation of
popular semi-supervised learning baselines, such as pseudo-label, VAT and Entropy
Minization. Moreover, we provide code for training supervised baseline models on on 1% and 10%
of ImageNet images that are substatinally better than those previously reported in the
literature. 

### Preparing data

Please refer to the
[instructions in the slim library](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started)
for downloading and preprocessing ImageNet data.

### Clone the repository and install dependencies

```
git clone https://github.com/google-research/s4l
cd s4l
python -m pip install -e . --user
```

We depend on some external files that need to be downloaded and placed in the
root repository folder. You can run the following commands to download them:

```
wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/preprocessing/inception_preprocessing.py
```

### Running on Google Cloud using TPUs

#### Step 1:

Create your own TPU cloud instance by following the
[official documentation](https://cloud.google.com/tpu/docs/quickstart).

For quick reference, at the time of writing this, run the following command in cloud shell (modify according to your needs):

```
ctpu up --name host-v3-8 --zone europe-west4-a --tpu-size v3-8
```

#### Step 2:

SSH to the VM instance created for your TPU above in step one above.
Clone the repository and install dependencies as described earlier.

#### Step 3:

Run model training script. For example:

```
gsutil mb gs://workdir
./config/supervised_imagenet_1p.sh --tpu_name host-v3-8 --dataset_dir <Google cloud bucket with preprocessed Imagenet Dataset> --workdir gs://workdir
```

You could start a TensorBoard to visualize the training/evaluation progress:

```
tensorboard --port 2222 --logdir gs://workdir
```

After/during training, run the self supervised model evaluation script:

```
./config/supervised_imagenet_1p.sh --tpu_name host-v3-8 --dataset_dir <Google cloud bucket with preprocessed Imagenet Dataset> --workdir gs://workdir --run_eval
```

## Results and pretrained models

Since this is an open-source reproduction of the code we used for the paper's experiments, we validated that we do indeed reproduce the results.
The following table summarizes the results we got using this codebase trained on Cloud TPUs, and compares it to the numbers we report in the paper.
All results are on 10% of labelled data unless noted otherwise.

All pre-trained models are made available as TF Hub modules in the `gs://s4l-models` bucket, which you can use as-is, or [browse here](https://console.cloud.google.com/storage/browser/s4l-models).

| Model           | TPU size | Top5 (Paper) |  Top5 | Top1  |
| :---            |  :---:   | :---: | :---: | :---: |
| Supervised (1%) |   v3-8   | 48.43 | 48.31 | 25.09 |
| Supervised      |   v3-8   | 80.43 | 80.42 | 56.36 |
| S4L-Rotation    |   v3-32  | 83.82 | 83.85 | 61.37 |
| S4L-Exemplar    |   v3-32  | 83.72 | 83.71 | 62.17 |
| Pseudo-labels   |   v3-8   | 82.41 | 82.86 | 60.24 |
| VAT             |   v3-32  | 82.78 | 82.71 | 59.98 |
| VAT + EntMin    |   v3-32  | 83.39 | 83.28 | 60.87 |
| MOAM Step 1     |   v3-128 | 88.80 | 88.70 | 69.46 |
| MOAM Step 2     |   v3-128 | 89.96 | TODO  | TODO  |
| MOAM Step 3     |   v3-128 | 91.23 | TODO  | TODO  |

## Authors

- [Xiaohua Zhai](https://sites.google.com/site/xzhai89/)
- [Avital Oliver](http://aoliver.org/)
- [Alexander Kolesnikov](https://github.com/akolesnikoff)
- [Lucas Beyer](http://lucasb.eyer.be/)

### This is not an official Google product
