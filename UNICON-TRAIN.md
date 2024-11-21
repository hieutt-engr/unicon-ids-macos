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

**(7) Road 150 epochs - Using UniconViT + AdamW**
python3 main_unicon_ViT.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_ConViT --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 200 --optimizer AdamW --save_freq 5 --method UniconViT > ./save/cmd_save/200_epoch_ROAD_ConViT.log

Continue
python3 main_unicon_ViT.py --batch_size 128 --learning_rate 0.005 --temp 0.07 --cosine --warm --n_classes 7 --dataset ROAD --trial road_fab_ConViT --data_folder ./data/road/preprocessed/fab_multi/TFRecord_w32_s8/2 --epochs 200 --optimizer AdamW --save_freq 5 --method UniconViT --resume ckpt_epoch_15.pth > ./save/cmd_save/200_epoch_ROAD_ConViT_resume_2.log

**(8) CAN 100 epochs - Using UniconViT + AdamW**
python3 main_unicon_ViT.py --batch_size 64 --learning_rate 0.0005 --temp 0.07 --cosine --warm --n_classes 5 --dataset CAN --trial can_ConViT --data_folder ./data/Car-Hacking/all_features/v2/TFRecord_w32_s32/2 --epochs 100 --optimizer AdamW --save_freq 1 --method UniconViT > ./save/cmd_save/100_epoch_CAN_ConViT.log

/home/hieutt/UniCon/data/Car-Hacking/all_features/v2

python3 main_unicon_ViT.py --batch_size 64 --learning_rate 0.0005 --temp 0.07 --cosine --warm --n_classes 5 --dataset CAN --trial can_ConViT --data_folder ./data/Car-Hacking/all_features/v2/TFRecord_w32_s32/2 --epochs 100 --optimizer AdamW --save_freq 1 --method UniconViT --resume ckpt_epoch_63.pth > ./save/cmd_save/100_epoch_CAN_ConViT.log

**(9) CAN-ML 200 epochs - Unicon**

```
/home/hieutt/UniCon/data/can-ml/preprocessed/TFRecord_w32_s16/2
  python main_unicon_v2.py --batch_size 128 --learning_rate 0.05 --temp 0.07 --cosine --warm --n_classes 10 --dataset CAN-ML --trial can_ml --data_folder ./data/can-ml/preprocessed/all_features/TFRecord_w32_s16/2 --epochs 200 --resume ./save/CAN-ML_models/UniCon/UniCon_CAN-ML_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_mixup_lambda_0.5_trial_can_ml_cosine_warm/ckpt_epoch_160.pth > ./save/cmd_save/200_epoch_CAN_ML_resume.log &
```
