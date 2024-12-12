# UniCon: Universum-inspired Supervised Contrastive Learning

The intuition of this paper is simple: why don't we assign Mixup samples to a generalized negative class? Just as humans
may perceive, if an animal is half dog and half cat, it is actually of neither species.

<p align="center">
  <img src="pic/intuition.png" width="700">
</p>
With this intuition, our framework is like this:
<p align="center">
  <img src="pic/framework.png" width="700">
</p>

Here we present the implementation of UniCon(ours)[1], SupCon[2], SimClr[3], Xent and SupMix(a hybrid of SupCon and
Un-Mix[4] proposed by us) on Pytorch. We use CIFAR-100 as an example dataset.
[1]: Universum-inspired Supervised Contrastive Learning [paper](https://arxiv.org/abs/2204.10695)  
[2]: Supervised Contrastive Learning [paper](https://arxiv.org/abs/2004.11362)  
[3]: A Simple Framework for Contrastive Learning of Visual Representations [paper](https://arxiv.org/abs/2002.05709)  
[4]: Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation
Learning [paper](https://arxiv.org/abs/2003.05438)

## Comparison

Results on CAN-ML:
|Method|Architecture|Batch size|Accuracy(%)|F1|PREC|REC|
|---|---|---|---|---|---|---|
|Vision Transformer|NLP|128|97.5|0.97|0.97|0.97|0.97|
|Cross Entropy|ResNet-50|128|97.6|0.97|0.97|0.97|0.97|
|REC CNN|CNN|128|99.2|0.98|0.98|0.98|0.98|
|CNN LSTM|CNN|128|99.3|0.98|0.98|0.98|0.98|
|UniCon|ResNet-50|128|99.0|0.99|0.99|0.99|

## Running

**(1) Preprocessing CAN dataset**

```
python3 preprocessing.py --window_size=32 --strided=32 --indir=./data/Car-Hacking --outdir=./data/Car-Hacking> data_preprocessing_can.txt
```

**(2) Preprocessing ROAD Fabrication dataset**

```
python3 preprocessing_road.py --window_size=32 --strided=8 --attack_type=road_fab --indir=./data/road/fab_dataset --outdir=./data/road/preprocessed/fab_multi/TFRecord > data_preprocessing_roadfab_new.txt
```

**(3) Preprocessing ROAD Masquerade dataset**

```
python3 preprocessing_road.py --window_size=32 --strided=8 --attack_type=road_mas --indir=./data/road/mas_dataset --outdir=./data/road/preprocessed/mas_multi/TFRecord > data_preprocessing_roadfab.txt
```

**(4) Preprocessing CAN_ML dataset**

```
python3 preprocessing_can_ml.py --window_size=32 --strided=16 > data_preprocessing_can_ml.txt
```

**(5) Train/Val Split**

````
python3 train_test_split_all.py --data_path ./data/road/preprocessed/fab_multi  --window_size 32 --strided 8 --rid 2
python3 train_test_split_all.py --data_path ./data/can-ml/preprocessed/all_features --window_size 32 --strided 16 --rid 2
``` new preprocessing can-ml
python3 train_test_split_all.py --data_path ./data/can-ml/preprocessed/all_features_v2 --window_size 32 --strided 16 --rid 2

````

**(6) UniCon**

```
python main_unicon.py --batch_size 128
  --learning_rate 0.05 --temp 0.07
  --cosine --warm
```

You can change Mixup parameter with `--lamda 0.5`. Or you can use CutMix with `--mix cutmix`.

**(7) Cross Entropy**

```
python main_ce.py --batch_size 128
  --learning_rate 0.05
  --cosine --warm
```

**(8) Train Unicon - ROAD**

```
python3 main_unicon.py --batch_size 128 --learning_rate 0.05 --temp 0.1 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s15/2 --epochs 100 --test_freq 90  > ./save/cmd_save/100_epoch_ROAD.log &
```

**(9) Tensorboard**

```
tensorboard --logdir ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_cosine_warm/runs
```

## Reference
