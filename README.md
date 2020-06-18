# Attention Mechanisms, Signal Encodings and Fusion Strategies for Improved Ad-hoc Video Search with Dual Encoding Networks

Python implementation of our ICMR'20 paper [1]. Based on the original implementation of dual encoding network, created by [Jianfeng Dong](https://github.com/danieljf24/dual_encoding). This software can be used for training a dual encoding network extended with two self-attention mechanisms in each modality.

## Dependencies
- Ubuntu 16.04
- CUDA 10.0
- Python 2.7
- PyTorch 1.3.0
- PyTorch transformers 1.2.0

### Data
#### Video datasets
The _tgif-msrvtt10k_ dataset is used for training the entire network, the _tv2016train_ dataset is used as validation dataset and the _iacc.3_ dataset for AVS evaluation.
For every dataset the ResNext-101 and ResNet-152 frame-level features were used. To download the pre-calculated features please refer to https://github.com/li-xirong/avs.
#### Sentence datasets
- Captions for _tgif-msrvtt10k_ and _tv2016train_ videos: please refer to https://github.com/li-xirong/avs
- Pre-trained word2vec model:  please refer to https://github.com/danieljf24/dual_encoding
- AVS ground truth and topics for [2016/2017/2018](AVS/)

## Data preparation
Extract both visual and textual features (downloadable files, as listed above) into the _rootpath_.
```shell
rootpath=$HOME/AVS_Search

# extract visual data
tar zxvf tgif_ResNext-101.tar.gz &rootpath
tar zxvf msrvtt10k_ResNext-101.tar &rootpath
tar zxvf tv2016train_ResNext-101.tar.gz &rootpath
tar zxvf iacc.3_ResNext-101.tar.gz &rootpath

tar zxvf tgif_ResNet-152.tar.gz &rootpath
tar zxvf msrvtt10k_ResNet-152.tar &rootpath
tar zxvf tv2016train_ResNet-152.tar.gz &rootpath
tar zxvf iacc.3_ResNet-152.tar.gz &rootpath

# extract textual data
tar zxvf tgif_textdata.tar.gz
tar zxvf msrvtt10k_textdata.tar.gz
tar zxvf tv2016train_textdata.tar.gz
tar zxvf word2vec.tar.gz

# combine feature of tgif and msrvtt10k
./do_combine_features.sh
```

## Training
To train a model with text-based self-attention mechanism (denoted as ATT) or a model with visual-based self-attention mechanism (denoted as ATV) with a specific configuration please follow the steps bellow:

```shell
rootpath=$HOME/AVS_Search

trainCollection=tgif-msrvtt10k
valCollection=tv2016train

visual_feature=pyresnext-101_rbps13k,flatten0_output,os_pyresnet-152_imagenet11k,flatten0_output,os
n_caption=2

optimizer=adam
learning_rate=0.00001
CUDA_VISIBLE_DEVICES=0 python ATT_w2v_bert_trainer.py    $trainCollection $valCollection --learning_rate $learning_rate --overwrite 0 --max_violation --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption  --concate full --optimizer $optimizer
CUDA_VISIBLE_DEVICES=0 python ATV_w2v_bert_trainer.py    $trainCollection $valCollection --learning_rate $learning_rate --overwrite 0 --max_violation --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption  --concate full --optimizer $optimizer

```
## AVS Evaluation
To evaluate a trained ATT or ATV model on iacc.3 dataset for TRECVID AVS 2016/2017/2018 topics use the following script:
```shell
rootpath=$HOME/AVS_Search
testCollection=iacc.3
logger_name=$rootpath/

python ATT_w2v_bert_AVS_evaluation.py  $testCollection --rootpath $rootpath --logger_name $logger_name
python ATV_w2v_bert_AVS_evaluation.py  $testCollection --rootpath $rootpath --logger_name $logger_name

```

## Citation
If you find this code useful in your work, please cite the following publication:

```
[1] D. Galanopoulos, V. Mezaris, "Attention Mechanisms, Signal Encodings and Fusion Strategies for Improved Ad-hoc Video Search with Dual Encoding Networks", Proc. ACM Int. Conf. on Multimedia Retrieval (ICMR'20), 2020, Dublin, Ireland.

bibtex entry:
@inproceedings{galanopoulos2020,
title={Attention Mechanisms, Signal Encodings and Fusion Strategies for Improved Ad-hoc Video Search with Dual Encoding Networks},
author={Galanopoulos, Damianos and Mezaris, Vasileios},
booktitle={Proceedings of the 2020 ACM International Conference on Multimedia Retrieval (ICMR'20)},
year={2020},
organization={ACM},
location = {Dublin, Ireland},
series = {ICMR ’20}
}
```
## Acknowledgement
This work was supported by the European Union Horizon 2020 research and innovation programme under contract H2020-780656 ReTV.
 