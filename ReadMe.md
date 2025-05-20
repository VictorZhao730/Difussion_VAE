# VAE, Latent Diffusion, and Diffusion Prior Project

This project implements and compares three generative modeling approaches: **VAE (Variational Autoencoder)**, **Latent Diffusion**, and **Diffusion Prior**. All models are trained and evaluated using reconstruction Mean Squared Error (MSE).

## Table of Contents

- [VAE, Latent Diffusion, and Diffusion Prior Project](#vae-latent-diffusion-and-diffusion-prior-project)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Model Architectures](#model-architectures)
    - [1. VAE (Variational Autoencoder)](#1-vae-variational-autoencoder)
    - [2. Latent Diffusion](#2-latent-diffusion)
    - [3. Diffusion Prior](#3-diffusion-prior)
  - [Dataset Description](#dataset-description)
  - [Training and Evaluation](#training-and-evaluation)
    - [Evaluation Metric](#evaluation-metric)
  - [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Citation](#citation)

---

## Project Overview

This project aims to explore and compare the performance of three types of generative models on structured data:
- **VAE**: Learns latent representations of data directly.
- **Latent Diffusion**: Applies diffusion modeling and denoising in the latent space of a VAE.
- **Diffusion Prior**: Generates latent vectors conditioned on external information (type and part from a CSV file) using a prior network, then denoises and reconstructs the data via diffusion and the VAE decoder.

All models are evaluated based on reconstruction MSE.

---

## Model Architectures

### 1. VAE (Variational Autoencoder)
- **Encoder**: Encodes input data into a latent vector $z$.
- **Decoder**: Decodes $z$ back to the original data space.
- **Training Objective**: Minimize reconstruction loss and KL divergence.

### 2. Latent Diffusion
- **Workflow**:
    1. Use the VAE encoder to obtain a latent vector $z$.
    2. Add noise to $z$ to simulate the diffusion process.
    3. Use a diffusion model (diff_model) to denoise and recover a clean $z$.
    4. Use the VAE decoder to reconstruct the data.
- **Training Objective**: Minimize the MSE between the reconstructed and original data after denoising.

### 3. Diffusion Prior
- **Workflow**:
    1. Read the `type` and `part` columns from the CSV file and concatenate them as the conditional vector `cond`.
    2. Use a prior network (prior_net) to generate a conditional latent vector $z_{prior}$ from `cond`.
    3. Add noise to $z_{prior}$.
    4. Use the diffusion model (diff_model) to denoise and recover a clean latent vector.
    5. Use the VAE decoder to reconstruct the data.
- **Training Objective**: Minimize the MSE of the final reconstruction.

---

## Dataset Description

- **Main dataset**: `dataset/iclr_final_truncated_fixed_powers.h5`
- **Conditional information**: `dataset/tsoulos_dataset_1.csv`
    - The `type` and `part` columns are concatenated to form the condition input for the Diffusion Prior model.
- **Download**: Download the main dataset from `Release` and place it under `./dataset/`
---

## Training and Evaluation

Each model has independent training and evaluation routines:

- **Training**: Train the VAE, Latent Diffusion, and Diffusion Prior models separately.
- **Evaluation**: For each model, compute the reconstruction MSE as the performance metric.

### Evaluation Metric

- **Reconstruction MSE (Mean Squared Error)**: Measures the model's ability to accurately reconstruct the input data.

---

## Usage

1. **Prepare Data**  
   Place the dataset and CSV condition file in the `dataset/` directory.

2. **Train Models**
    - Train VAE:  
      ```bash
      python train_vae.py
      ```
    - Train Latent Diffusion:  
      ```bash
      python train_diffusion_latent.py
      ```
    - Train Diffusion Prior:  
      ```bash
      python train_diffusion_prior.py
      ```

3. **Evaluate Models**
    - Evaluate:  
      ```bash
      python evaluate.py
      ```


4. **Results**  
   The reconstruction MSE for each model will be printed in the terminal.

---

## Dependencies

- Python 3.8+
- PyTorch
- h5py
- pandas
- Other dependencies are listed in `requirements.txt`

---

## Citation

If this project is useful for your research, please consider citing it.

---

For any questions or suggestions, feel free to open an issue or pull request!

---

If you need further customization or more technical details, let me know!
