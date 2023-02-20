# Bounding Evidence and Estimating Log-Likelihood in VAE

The code for the paper 'Bounding Evidence and Estimating Log-Likelihood in VAE' which was accepted for the AISTATS2023 conference.

Below we write commands to reproduce the experiments included in the paper:

### 1a). Training VAE models for a few datasets: MNIST, SVHN, CelebA

```bash
bash run.sh
```
### 1b). Training IWAE models for a few datasets: MNIST, SVHN, CelebA

```bash
bash run.sh IWAE
```

### 2. Estimation of boundaries of the likelihood of models

#### Generate samples:

```bash
bash calculate_components.sh
```
#### Calculate bounds:

```bash
bash bounds.sh
```
