# hunyuan-lora-converter

A Python tool for converting LoRA weights trained in **HunyuanVideo (OneTrainer format, non-OMI)** to a **diffusion-pipe (ComfyUI-compatible)** format.

### ✅ What it does

- Converts LoRA weights from `single_blocks` and `double_blocks` in the Hunyuan format to the format expected by ComfyUI diffusion models.
- Merges LoRA components (e.g. `q`, `k`, `v`, `proj`) using matrix multiplication.
- Applies **SVD-based low-rank approximation** to produce a pair of `lora_A` and `lora_B` matrices per layer.
- Includes **accuracy checks** for each reconstructed layer.
- Outputs a **single `.safetensors` file** with the converted LoRA weights.

> ⚠️ Note: This is a **lossy** conversion due to rank-limited SVD, and takes time depending on matrix size and GPU availability.
> ⚠️ Note: Support for `single_blocks` and `double_blocks` only.
---

### 📦 Requirements

- `torch` (>= 1.12 recommended)
- `safetensors`
- `tqdm`
