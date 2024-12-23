**(1) Last run**

```
 -------ADAM---------
  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.001 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_adamw_v2 --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 --optimizer AdamW --save_freq 2 > ./save/cmd_save/100_epoch_ROAD_AdamW_v2.log

 -------SGD---------
  python3 main_unicon_v2.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_sgd --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 --optimizer SGD --save_freq 1 > ./save/cmd_save/100_epoch_ROAD_SGD.log
```

**(2) CAN 100 epochs - Using CAN ID + CAN DATA > (Binary)**

```
  python3 main_unicon_v2.py --batch_size 64 --learning_rate 0.05 --temp 0.1 --cosine --warm --n_classes 5 --dataset CAN --trial can_v2 --data_folder ./data/Car-Hacking/all_features/v2/TFRecord_w32_
  s32/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN.log &

/home/hieutt/UniCon/save/CAN_models/UniCon/UniCon_CAN_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.1_mixup_lambda_0.5_trial_can_v2_101424_160529_cosine_warm/ckpt_epoch_20.pth
  python3 main_unicon_v2.py --batch_size 64 --learning_rate 0.05 --temp 0.1 --cosine --warm --n_classes 5 --dataset CAN --trial can_v2 --data_folder ./data/Car-Hacking/all_features/v2/TFRecord_w32_s32/2 --epochs 100 --resume ./save/CAN_models/UniCon/UniCon_CAN_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.1_mixup_lambda_0.5_trial_can_v2_101424_160529_cosine_warm/ckpt_epoch_20.pth > ./save/cmd_save/100_epoch_CAN.log &
```

**(3) ROAD 100 epochs - Using CAN ID + CAN DATA + timestamp > (Binary) AdamW**

```
 python3 main_unicon_AdamW.py --batch_size 128 --learning_rate 0.05 --temp 0.1 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_adamw --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 > ./save/cmd_save/100_epoch_ROAD_AdamW.log &
```

**(4) Road 100 epochs - Using ConViT + AdamW**

```
  python3 main_unicon_ViT.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_vit --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 --optimizer AdamW --save_freq 1 > ./save/cmd_save/100_epoch_ROAD_ViT.log
```

**(5) Road 100 epochs - Using CEViT + AdamW**
python3 main_unicon_ViT.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_vit --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 100 --optimizer AdamW --save_freq 1 --method UniViT > ./save/cmd_save/100_epoch_ROAD_ViT.log

---- Can-ML ------

python main_CE_ViT.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can-vit --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 --save_freq 2 > ./save/cmd_save/100_epoch_CAN_ML_ViT.log

---v2---
python main_CE_ViT.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can-vit-v2 --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 --save_freq 2 > ./save/cmd_save/100_epoch_CAN_ML_ViT_v2.log

---v3---
python main_CE_ViT.py --batch_size 128 --learning_rate 0.001 --optimizer AdamW --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can-vit-v2 --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 --save_freq 2 > ./save/cmd_save/100_epoch_CAN_ML_ViT_v2.log

**(7) Road 150 epochs - Using UniconViT + AdamW**
python3 main_unicon_ViT.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_ConViT --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 200 --optimizer AdamW --save_freq 5 --method UniconViT > ./save/cmd_save/200_epoch_ROAD_ConViT.log

**(8) CAN-ML 200 epochs - Unicon**

```
/home/hieutt/UniCon/data/can-ml/preprocessed/TFRecord_w32_s16/2

  python main_unicon_v2.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_v3 --data_folder ./data/can-ml/preprocessed/all_features_v2/TFRecord_w32_s16/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN_ML_v3.log &

  ----Continue-----
    python main_unicon_v2.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml_v2 --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 --resume ckpt_epoch_100.pth > ./save/cmd_save/100_epoch_CAN_ML_v2_continue.log &
```

**(9) CAN-ML 200 epochs - Main CE**

```
/home/hieutt/UniCon/data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2

  python main_ce.py --batch_size 128 --learning_rate 0.05 --cosine --warm --augment --n_classes 10 --dataset CAN-ML --trial can_ml --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN_ML_CE.log &
```

**(10) CAN-ML 100 epochs - Baseline**

```
/home/hieutt/UniCon/data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2
----- RecCNN ------
  python main_baseline.py --batch_size 128 --learning_rate 0.05 --n_classes 10 --cosine --warm --dataset CAN-ML --trial can_ml_rec_cnn --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN_ML_RecCNN.log &

----- LSTM CNN ------
  python main_baseline.py --batch_size 128 --learning_rate 0.05 --n_classes 10 --cosine --warm --model lstm_cnn --dataset CAN-ML --trial can_ml_lstm_cnn --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 100 > ./save/cmd_save/100_epoch_CAN_ML_LSTM_CNN.log &
```
