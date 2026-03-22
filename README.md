# DeepLearning_A2

Variational Autoencoder (VAE) implementation for FashionMNIST, developed as part of a Deep Learning assignment.

## Overview

This project implements a **convolutional Variational Autoencoder (VAE)** to learn compact latent representations of FashionMNIST images and reconstruct them from latent space. Unlike a standard autoencoder, the VAE encodes each image as a **probability distribution** in latent space using a mean vector and log-variance vector, which makes the model suitable for both reconstruction and generative modelling.

The notebook explores:
- encoder and decoder design
- reparameterisation trick
- VAE loss combining reconstruction + KL divergence
- hyperparameter comparison across multiple model configurations
- selection of the best-performing model based on test loss

## Repository Contents

- `A2_VAE.ipynb` — main Jupyter notebook containing the full implementation, experiments, visualisations, and discussion
- `Variational Auto Encoders.pdf` — supporting theory notes and short written explanations for the assignment
- `models.txt` — saved comparison results for tested model configurations
- `vae_model-best.pt` — best model weights
- `vae_history-best.pt` — saved training history of the best model
- `data/FashionMNIST/raw/` — dataset files

## Project Objective

The goal of this assignment is to:
1. Build a convolutional VAE for FashionMNIST
2. Learn a meaningful latent representation of clothing images
3. Reconstruct images from compressed latent codes
4. Compare multiple hyperparameter settings
5. Analyse the trade-off between reconstruction quality and latent-space regularisation

## Model Summary

### Encoder
The encoder compresses a `1 x 28 x 28` FashionMNIST image into latent distribution parameters:
- `mu` (latent mean)
- `log_var` (latent log-variance)

It uses:
- convolutional layers
- batch normalization
- ReLU activation
- dropout regularisation

### Decoder
The decoder reconstructs the image from a sampled latent vector using:
- a fully connected expansion layer
- reshaping into feature maps
- transposed convolutions
- sigmoid output activation

### Reparameterisation Trick
Instead of directly sampling from the latent distribution, the VAE uses:

\[
z = \mu + \sigma \cdot \epsilon
\]

where:
- `\mu` is the latent mean
- `\sigma` is derived from `log_var`
- `\epsilon ~ N(0,1)`

This allows stochastic sampling while keeping training differentiable.

## Loss Function

The VAE is trained using two loss components:

### 1. Reconstruction Loss
Measures how close the reconstructed image is to the original input.

### 2. KL Divergence Loss
Regularises the latent distribution so that it stays close to a standard normal prior.

Overall loss:

\[
\text{Loss} = \text{Reconstruction Loss} + \lambda \times \text{KL Loss}
\]

where `λ` controls the balance between reconstruction fidelity and latent-space regularisation.

## Experiments

The notebook compares different model configurations by varying:
- convolution kernel/channel sizes
- dropout
- KL loss weight (`lambda`)
- learning rate
- latent size

The final comparison identifies the best model using **test loss** as the main criterion.

## Best Model

Among the selected configurations, the best-performing model used:

- `kernels = [64, 128, 256]`
- `dropout = 0.0`
- `lambda = 0.5`
- `latent_size = 10`
- `epochs = 30`

It achieved the **lowest final test loss of 229.01**.

## Key Observations

- Larger convolutional capacity performed slightly better than smaller configurations
- Increasing latent size generally improved reconstruction quality
- Increasing KL weight (`lambda`) strengthened latent-space regularisation but could reduce reconstruction sharpness
- The results demonstrate the expected VAE trade-off between:
  - **good reconstructions**
  - **well-structured latent space**

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd DeepLearning_A2
