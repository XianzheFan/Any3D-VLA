# 🦾 Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds ☁️

[![arXiv](https://img.shields.io/badge/arXiv-2602.00807-df2a2a.svg)](https://arxiv.org/abs/2602.00807)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://xianzhefan.github.io/Any3D-VLA.github.io/)

Existing Vision-Language-Action (VLA) models typically take 2D images as visual input, which limits their spatial understanding in complex scenes. How can we incorporate 3D information to enhance VLA capabilities? We conduct a pilot study across different observation spaces and visual representations. The results show that explicitly lifting visual input into point clouds yields representations that better complement their corresponding 2D representations. To address the challenges of (1) scarce 3D data and (2) the domain gap induced by cross-environment differences and depth-scale biases, we propose **Any3D-VLA**. It unifies the simulator, sensor, and model-estimated point clouds within a training pipeline, constructs diverse inputs, and learns domain-agnostic 3D representations that are fused with the corresponding 2D representations. Simulation and real-world experiments demonstrate Any3D-VLA's advantages in improving performance and mitigating the domain gap.

## 🔥 Updates

- [2026/02/03] Release the paper and code.

## 🛠️ Model Server

### Installation & Environment

We recommend using `Conda` to manage your environment. This project is optimized for CUDA 11.8.

```bash
conda create -n any3dvla_env python=3.12 -y
conda activate any3dvla_env

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Install the core package in editable mode
pip install -e src/vla_network

# Clone Concerto repo
git clone https://github.com/Pointcept/Concerto.git
pip install -e .
```

### Running the Server

To run the model server:

```bash
STORAGE_PATH=`readlink -f ../storage` python -m vla_network.scripts.serve --path ../storage/any3dvla/checkpoint/model.safetensors --port 6666
```

For faster inference, add `--compile` to the command. It will speed up the inference around 50\% with a cost of slower model loading.

## 💬 Supported Instructions

Our model accepts the following instructions:

- `pick up {object}`
- `stack {color} bowl onto {color} bowl`
- `stack {color} cube onto {color} cube`
- `move {object} to {container}`
- `move {object} to {color} {container}`
- `pick up {color} {object}`

## 🤖 Real-World Control Interface

Setup real-world controller following [this repo](https://github.com/XianzheFan/Any3D-VLA-real-world-controller).

<!-- ## Citation

```bibtex

``` -->