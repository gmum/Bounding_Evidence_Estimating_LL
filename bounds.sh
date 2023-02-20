#!/bin/bash


for dataname in MNIST SVHN CelebA; do
  for modelname in VAE IWAE-5 IWAE-10; do
    python bounds.py --dataname $dataname --modelname $modelname --num_files 3 --save2file "scores.csv" --num_feature "8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768"
  done
done
