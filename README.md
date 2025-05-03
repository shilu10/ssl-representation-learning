# ssl-representation-learning

This repository contains implementations of several Self-Supervised Learning (SSL) algorithms along with custom optimizers, learning rate schedulers, and loss functions. The main objective is to leverage large amounts of unlabeled data to learn useful representations for downstream tasks. The following SSL models and tasks have been implemented:

## Algorithms
- **SimCLR** (Simple Contrastive Learning of Representations)
- **BYOL** (Bootstrap Your Own Latent)
- **MoCo** (Momentum Contrast)
- **PIRL** (Pretext-Invariant Representation Learning)
- **Jigsaw Puzzle Solving**
- **Rotation-Prediction**
- **Context-Encoder**
- **Context-Prediction**

## Features
- **Custom Optimizers**: 
  - **LARS** (Layer-wise Adaptive Rate Scaling) optimizer implemented from scratch.
- **Learning Rate Schedulers**:
  - **StepDecay**
  - **ExponentialDecay**
  - **LinearDecay**
  - **LinearDecay**
  - **PiecewiseConstantDecay**
  - **PolynomialDecay**
- **Loss Functions**:
  - **InfoNCE**
  - **NCE**
  - **BYOL Loss**
  - **NTXent Loss**

## Table of Contents

- [Overview](#overview)
- [Pretext Tasks](#pretext-tasks)
- [Contrastive Learning Methods](#contrastive-learning-methods)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Pretext Task Training](#pretext-task-training)
  - [Contrastive Learning Training](#contrastive-learning-training)
- [Arguments](#arguments)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations from unlabeled data. This repository includes implementations of various SSL methods, enabling users to train models on unlabeled datasets and evaluate their performance on downstream tasks.

## Pretext Tasks

The repository supports several pretext tasks, including:

- **Jigsaw Puzzle Solving**
- **Rotation Prediction**
- **Context Prediction**
- **Context Encoder**

These tasks serve as auxiliary objectives to learn useful representations without labeled data.

## Contrastive Learning Methods

Implementations of popular contrastive learning frameworks are available, such as:

- **SimCLR**
- **MoCo v1, v2, v3**

These methods learn representations by contrasting positive pairs against negative pairs.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Imutils
- Other dependencies listed in `requirements.txt`

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/shilu10/ssl-representation-learning.git
cd ssl-representation-learning
pip install -r requirements.txt
```
## Usage
### Pretext Task Training
To train a model using a pretext task, run the following command:
```bash
python main_pretext.py --config_path <path_to_config> --unlabeled_datapath <path_to_unlabeled_data> --num_epochs <num_epochs> --batch_size <batch_size>
```
Replace <path_to_config>, <path_to_unlabeled_data>, <num_epochs>, and <batch_size> with appropriate values.

### Contrastive Learning Training
```bash
python main.py --config_path <path_to_config> --unlabeled_datapath <path_to_unlabeled_data> --num_epochs <num_epochs> --batch_size <batch_size>
```

## Arguments
The following command-line arguments are available:

- --num_epochs: Number of epochs to train the model.

- --tensorboard: Directory to store TensorBoard logs.

- --history: Directory to save training history.

- --result_path: Directory to store results.

- --resume: Whether to resume from a checkpoint.

- --checkpoint: Directory to save checkpoints.

- --checkpoint_save_freq: Frequency of saving checkpoints.

- --checkpoint_max_keep: Maximum number of checkpoints to keep.

- --contrastive_task_type: Type of contrastive learning method (simclr, mocov1, mocov2).

- --unlabeled_datapath: Path to the unlabeled dataset.

- --batch_size: Batch size for training.

- --shuffle: Whether to shuffle the dataset.

-  --gpus: GPU device ID(s) to use.

- --config_path: Path to the configuration file.

- --use_validation: Whether to use a validation split.

```bash
python main.py --num_epochs 30 --tensorboard logs --history history --unlabeled_datapath ./stl10/unlabeled_images/ --batch_size 64 --shuffle True
```
This will start training using the SimCLR method on your unlabeled dataset stored in ./stl10/unlabeled_images/.

## License
This project is licensed under the Apache-2.0 License.

## Acknowledgments
[SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

[MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

[DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)



