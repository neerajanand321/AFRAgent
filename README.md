# AFRAgent: An Adaptive Feature Renormalization Based High Resolution Aware GUI Agent

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.00846)
[![Conference](https://img.shields.io/badge/WACV-2026-blue)](https://wacv2026.thecvf.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)

Official implementation of **"AFRAgent: An Adaptive Feature Renormalization Based High Resolution Aware GUI agent"** accepted at **WACV 2026**.

**Authors:** Neeraj Anand, Rishabh Jain, Sohan Patnaik, Balaji Krishnamurthy, Mausoom Sarkar

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Checkpoints](#model-checkpoints)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🔍 Overview

AFRAgent is a state-of-the-art multimodal architecture for smartphone GUI automation that achieves superior performance while being **less than one-fourth the size** of its nearest competitor. Built on InstructBLIP, AFRAgent introduces an innovative **Adaptive Feature Renormalization** technique (token-level affine transformation) that effectively enriches low-resolution image embeddings and fuses high-resolution details for improved spatial understanding and action prediction.

### Key Capabilities
- 🎯 **Accurate Widget Identification**: Enhanced spatial information through adaptive feature renormalization
- 🚀 **Efficient Architecture**: Compact model size with state-of-the-art performance
- 📱 **Smartphone Automation**: Autonomous execution of tasks on mobile user interfaces
- 🔄 **Multi-Resolution Processing**: Seamless fusion of low and high-resolution image features

## ✨ Key Features

1. **Adaptive Feature Renormalization (AFR)**: Token-level affine transformation technique that dynamically adjusts image embeddings
2. **Multi-Resolution Architecture**: Processes both low-resolution full images and high-resolution crops
3. **Compact Model**: InstructBLIP-based architecture that's significantly smaller than competing models
4. **State-of-the-Art Performance**: New baseline on Meta-GUI and AITW benchmarks

## 🏗️ Architecture

AFRAgent consists of several key components:

### Model Variants

- **`any_res_adain_queries_fusion.py`**: Adaptive Feature Renormalization with query-level fusion
- **`any_res_adain_mlp_fusion.py`**: AFR with MLP-based fusion mechanism
- **`any_res_img_embed_fusion.py`**: Direct image embedding fusion
- **`any_res_queries_embed_fusion.py`**: Query embedding fusion approach
- **`low_res_AdaIn.py`**: Low-resolution adaptive instance normalization
- **`low_res_qformer_MLP.py`**: Low-resolution Q-Former with MLP

The recommended model for training and evaluation is the **AnyResAdaIn** architecture which implements the full adaptive feature renormalization technique.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 12.4+ (for GPU support)
- 8 GPUs recommended for distributed training

### Step 1: Clone the Repository
```bash
git clone https://github.com/neerajanand321/AFRAgent.git
cd AFRAgent
```

### Step 2: Create Conda Environment
```bash
conda create -n afragent python=3.8
conda activate afragent
```

### Step 3: Install PyTorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- transformers[torch]
- jax[cpu]
- rich
- evaluate
- rouge_score
- tensorflow
- tf-keras
- wandb
- opencv-python
- deepspeed
- pillow
- awscli

## 📊 Dataset Preparation

AFRAgent is trained and evaluated on the **Android-in-the-Wild (AITW)** dataset, which contains five subsets:
- `general`: General smartphone interactions
- `single`: Single-step tasks
- `install`: App installation tasks
- `google_apps`: Google apps interactions
- `web_shopping`: Web shopping tasks

### Data Generation

The AITW dataset is stored in Google Cloud Storage. Run the following commands to download and prepare the data:

```bash
# Make the environment setup script executable
chmod +x createEnv.sh

# Generate dataset for each subset
python aitw_data_gen.py --dataset 'general'
python aitw_data_gen.py --dataset 'single'
python aitw_data_gen.py --dataset 'install'
python aitw_data_gen.py --dataset 'web_shopping'
python aitw_data_gen.py --dataset 'google_apps'
```

This will:
1. Download the TFRecord files from GCS
2. Parse episodes and extract images
3. Generate train/val/test splits
4. Save processed data in `dataset/aitw/` directory

### Dataset Structure
```
dataset/
├── aitw/
│   ├── general/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── single/
│   ├── install/
│   ├── google_apps/
│   └── web_shopping/
├── general_texts_splits.json
├── google_apps_texts_splits.json
├── install_texts_splits.json
├── single_texts_splits.json
└── web_shopping_texts_splits.json
```

## 🚀 Training

### Single GPU Training
```bash
python instructblip_main.py \
  --data_root 'dataset' \
  --bs 16 \
  --epoch 12 \
  --lr 5e-5 \
  --use_high_res True \
  --train_any_res_adain True \
  --all_data 1 \
  --all_eval True \
  --user_msg "afragent_training"
```

### Distributed Training with DeepSpeed (Recommended)

For multi-GPU training with DeepSpeed ZeRO-2:

```bash
# Make the run script executable
chmod +x run.sh

# Launch distributed training
accelerate launch \
  --config_file="./configs/deepspeed_config.yaml" \
  instructblip_main.py \
  --data_root 'dataset' \
  --bs 16 \
  --all_data 1 \
  --all_eval True \
  --user_msg "all_data_any_res_adain_finetuning" \
  --use_high_res True \
  --train_any_res_adain True
```

### Training Arguments

Key arguments:
- `--data_root`: Path to dataset directory
- `--bs`: Batch size per GPU
- `--epoch`: Number of training epochs (default: 12)
- `--lr`: Learning rate (default: 5e-5)
- `--use_high_res`: Enable high-resolution image processing
- `--train_any_res_adain`: Enable Adaptive Feature Renormalization
- `--all_data`: Train on all dataset subsets
- `--all_eval`: Evaluate on all subsets during training
- `--img_size`: Input image size (default: 224)
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory
- `--num_workers`: Number of data loading workers (default: 32)

### Monitoring Training

Training progress is logged to Weights & Biases. To view your training runs:
```bash
wandb login
```

Note: Update the WandB API key in `instructblip_main.py` line 30 with your own key.

## 🧪 Evaluation

### Evaluate Trained Model
```bash
python instructblip_main.py \
  --evaluate_dir /path/to/checkpoint \
  --data_root 'dataset' \
  --eval_bs 64 \
  --use_high_res True \
  --train_any_res_adain True \
  --all_eval True
```

### Evaluation Metrics

AFRAgent reports the following metrics:
- **Action Accuracy**: Percentage of correctly predicted actions (touch points, swipes, etc.)
- **Text Accuracy**: Accuracy of text input predictions
- **Type Accuracy**: Accuracy of action type classification
- **Overall Accuracy**: Combined accuracy across all evaluation subsets

Predictions and metrics are saved to JSON files in the checkpoint directory.

## 💾 Model Checkpoints

Trained model checkpoints will be available soon. The checkpoints include:
- Model weights
- Processor configuration
- Training configuration

Models are saved in the `experiments/` directory with the naming convention:
```
experiments/{user_msg}_{img_type}_lr{lr}_bs{batch_size}_ip{input_len}_op{output_len}_ep{epoch}/
```

## 📈 Results

AFRAgent achieves **state-of-the-art performance** on smartphone GUI automation benchmarks:

### AITW Benchmark
Performance on the Android-in-the-Wild dataset across different task categories.

### Meta-GUI Benchmark
Superior performance compared to larger competing models while maintaining a compact architecture.

### Model Efficiency
- **Model Size**: ~1/4 the size of nearest competitors
- **Inference Speed**: Optimized for real-time GUI automation
- **Training Efficiency**: Efficient training with DeepSpeed ZeRO-2

*Detailed quantitative results are available in the paper.*

## 📝 Citation

If you find AFRAgent useful for your research, please cite our paper:

```bibtex
@article{anand2025afragent,
  title={AFRAgent: An Adaptive Feature Renormalization Based High Resolution Aware GUI agent},
  author={Anand, Neeraj and Jain, Rishabh and Patnaik, Sohan and Krishnamurthy, Balaji and Sarkar, Mausoom},
  journal={arXiv preprint arXiv:2512.00846},
  year={2025},
  note={Accepted at WACV 2026}
}
```

## 🙏 Acknowledgements

- This work builds upon [InstructBLIP](https://github.com/salesforce/LAVIS) from Salesforce Research
- Dataset: [Android-in-the-Wild (AITW)](https://github.com/google-research/google-research/tree/master/android_in_the_wild) from Google Research
- We thank the authors of the AITW dataset and InstructBLIP for making their work publicly available

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: neerajanandfirst@gmail.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **Paper**: [arXiv:2512.00846](https://arxiv.org/abs/2512.00846)
- **Conference**: [WACV 2026](https://wacv2026.thecvf.com/)
- **Dataset**: [Android-in-the-Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild)

---

**Note**: This repository contains the official implementation for research purposes. For production deployments, additional testing and optimization may be required.

