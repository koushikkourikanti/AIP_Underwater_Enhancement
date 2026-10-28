
# Edge-Preserving Underwater Image Enhancement

## CSC 8260 — Advanced Image Processing

Georgia State University | Fall 2026

### Team Members
- Koushik Kourikanti
- Harika Chinnam
- Cecilia Muniz Siqueira

## Project Objective

Investigate whether guided edge-aware refinement can improve
fine object boundary and texture preservation in underwater
images without introducing excessive noise or halos.

## Research Models

### Baseline: Water-Net (2020)
Paper: https://arxiv.org/pdf/1901.05495

Source:
https://github.com/Li-Chongyi/Water-Net_Code

### Target: U-Shape Transformer (2023)
Paper: https://doi.org/10.1109/TIP.2023.3276332

Source:
https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement

### Supporting Method: Guided Image Filtering (2010)
Paper:
https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf

## Dataset

LSUI — Large-Scale Underwater Image Dataset

- 3,879 training files in the downloaded split
- 400 testing files in the downloaded split
- 10 selected pilot test images

Dataset source:
https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement

Matching reference images must be verified for final evaluation.

## Implementation Status

- Water-Net pretrained inference: completed on 10 pilot images
- U-Shape Transformer inference: completed on 10 pilot images
- Official U-Shape example validation: completed
- Guided-filter refinement: pending integration
- Full test-set evaluation: pending

## Environment

### Water-Net
- Python 3.11
- TensorFlow 2.15.1
- OpenCV 4.10
- Contributed Python preprocessing
- Pilot inference: 320 x 256

### U-Shape Transformer
- Python 3.11
- PyTorch 2.11.0 + CUDA 12.8
- NVIDIA RTX 5060 Laptop GPU
- Pretrained generator_795.pth
- Pilot inference: 256 x 256

The original authors' model repositories and checkpoints
must be downloaded separately.

## Current Validation

One official U-Shape Transformer example:

- Original input PSNR: 12.34 dB
- Enhanced output PSNR: 23.80 dB

These are single-image validation results, not final
LSUI benchmark results.

## Team Responsibilities

Koushik: Baseline and target model inference

Harika: Guided edge-aware refinement

Cecilia: PSNR, SSIM, edge F1, runtime, and ablation evaluation

All members: Final presentation and report
