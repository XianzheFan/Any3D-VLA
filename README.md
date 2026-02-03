# Any3D-VLA

We propose **Any3D-VLA**. It unifies simulator, sensor, and model-estimated point clouds in the training pipeline, enabling diverse inputs and learning domain-agnostic 3D representations that are fused with the corresponding 2D representations.

---

## 🛠️ Installation & Environment

We recommend using `Conda` to manage your environment. This project is optimized for CUDA 11.8.

```bash
# 1. Create and activate the environment
conda create -n any3d_vla_env python=3.12 -y
conda activate any3d_vla_env

# 2. Install PyTorch with CUDA 11.8 support
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# 3. Install the core package in editable mode
pip install -e src/vla_network

```

---

```