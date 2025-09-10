# Rust Image Classifier (Burn)

Image classification in Rust using the [Burn](https://github.com/burn-rs/burn) framework.

## 🚀 Features
- Train a CNN on your own dataset
- Inference on single images
- Modular dataset loader and batcher
- Save & load trained models

## 📂 Project Structure
- `src/` → Rust source code
- `data/` → datasets (ignored in git, add your own)
- `models/` → trained weights (ignored in git)
- `scripts/` → helper scripts

## 🔧 Setup
```bash
git clone https://github.com/T12018/Rust-Image-classifier.git
cd Rust-Image-classifier
cargo build


Training
cargo run --release --wgpu

Inference
cargo run --bin inference --

