#!/bin/bash
chmod 775 main.py
for i in `seq 0 9`;
do
    echo OVCARE Dataset Baseline + Fake Split "$i"
    ./main.py --epoch 10 --use_equalized_batch --use_pretrained --n_eval_samples 500 --test_augmentation --train_ids_file_name 768_gan_eval_ids/50_60_baseline_fake_train_"$i"_ids.txt --val_ids_file_name 768_gan_eval_ids/60_split_"$i"_val_ids.txt --model_name_prefix tvt_50_split_"$i"_baseline_fake --rep_intv 100 --save_dir /projects/ovcare/classification/ywang/project_save/gan_eval_save/
    ./main.py --epoch 10 --use_equalized_batch --use_pretrained --n_eval_samples 500 --mode Testing --test_augmentation --test_ids_file_name 768_gan_eval_ids/60_split_"$i"_test_ids.txt --model_name_prefix tvt_50_split_"$i"_baseline_fake --rep_intv 100 --save_dir /projects/ovcare/classification/ywang/project_save/gan_eval_save/
done