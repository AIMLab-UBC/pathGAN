#!/bin/bash
chmod 775 ./main.py
for i in `seq 0 9`;
do
    echo TCGA Dataset Baseline Split "$i"
    ./main.py --epoch 10 --use_equalized_batch --use_pretrained --is_tcga --preload_image_file_name images.h5 --n_eval_samples 500 --dataset_dir /projects/ovcare/classification/ywang/gan_tcga_dataset/ --train_ids_file_name patch_ids/baseline_"$i"_train_ids.txt --val_ids_file_name patch_ids/"$i"_val_ids.txt --model_name_prefix tcga_"$i"_baseline --rep_intv 2 --save_dir /projects/ovcare/classification/ywang/gan_tcga_dataset/saved_models/ --log_dir /projects/ovcare/classification/ywang/gan_tcga_dataset/logs/
    ./main.py --epoch 10 --use_equalized_batch --use_pretrained --is_tcga --preload_image_file_name images.h5 --n_eval_samples 500 --mode Testing --dataset_dir /projects/ovcare/classification/ywang/gan_tcga_dataset/ --test_ids_file_name patch_ids/"$i"_test_ids.txt --model_name_prefix tcga_"$i"_baseline --rep_intv 2 --save_dir /projects/ovcare/classification/ywang/gan_tcga_dataset/saved_models/ --log_dir /projects/ovcare/classification/ywang/gan_tcga_dataset/logs/
done
