#!/bin/bash


config_file=??? # type path to trained model ('results/VAEmnist_${DATE}/config.yaml', 'results/VAEsvhn_${DATE}/config.yaml', 'results/VAEceleba_${DATE}/config.yaml')
num_repeat=10
n=1024
rate_extended=10

python calculate_components.py --config "${config_file}" --num_repeat "$num_repeat" --n "$n" --rate_extended "$rate_extended"
