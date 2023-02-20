#!/bin/bash


if [[ "$1" == "IWAE" ]]; then

  export PYTHONPATH=${PYTHONPATH}:.

  for n in 5 10; do
    python ./IWAE/main.py --num_samples $n --config ./configs/vae_mnist.yaml --lr 1e-4 --num_epochs 75
    python ./IWAE/main.py --num_samples $n --config ./configs/vae_svhn.yaml --lr 1e-4 --num_epochs 75
    python ./IWAE/main.py --num_samples $n --config ./configs/vae_celeba.yaml --lr 1e-4 --num_epochs 100
  done

else

python run.py --config configs/vae_celeba.yaml
python run.py --config configs/vae_svhn.yaml
python run.py --config configs/vae_mnist.yaml

fi
