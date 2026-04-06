# 3D Vision: Gaussian Splatting & Generative Video Priors for Pose Estimation

This repository explores advanced 3D computer vision techniques, focusing on explicit scene representation and the integration of generative models to solve geometric challenges. The project is divided into two main parts: enhancing pose estimation with video interpolation and implementing sparse-view 3D Gaussian Splatting.

## Part 1: Enhancing Pose Estimation with Generative Video Models
[cite_start]Foundation models for 3D vision, such as DUSt3R, exhibit impressive performance in standard scenarios but often struggle with wide-baseline image pairs[cite: 4434]. [cite_start]Inspired by the InterPose framework[cite: 4436], this project leverages generative video models to bridge this gap.

### Key Implementations & Analysis:
- [cite_start]**Pipeline Integration**: Evaluated the DUSt3R model on both original wide-baseline pairs and sequences interpolated by a video generative model[cite: 4443, 4450].
- [cite_start]**Quantitative Evaluation**: Analyzed the performance gap using Mean Rotation Error (MRE)[cite: 4451]. The addition of interpolated frames significantly improved the accuracy and stability of the pose estimation.
- [cite_start]**PnP Solver Diagnostics**: Conducted case studies on specific failure modes (e.g., Perspective-n-Point solver crashes due to degenerate, co-planar, or co-linear points)[cite: 4548, 4555]. [cite_start]Implemented robust error handling to gracefully manage solver exceptions[cite: 4562, 4564].

## Part 2: Sparse-View 3D Gaussian Splatting (3DGS)
[cite_start]The second part of this project implements 3D Gaussian Splatting, specifically tailored for sparse-view scenarios, referencing the InstantSplat architecture[cite: 4600, 4604]. 

### Key Implementations & Analysis:
- [cite_start]**Scene Representation**: Trained a set of 3D Gaussians to act as an explicit representation of a scene using only a limited set of posed training images[cite: 4600, 4601].
- [cite_start]**Novel View Synthesis**: Rendered novel viewpoints based on unseen test camera poses[cite: 4602, 4615].
- [cite_start]**Hyperparameter Tuning & Evaluation**: Systematically tuned critical training parameters (learning rate, densification strategies, iterations) and evaluated the rendering quality using standard metrics including PSNR, SSIM, and LPIPS[cite: 4618, 4619].
- [cite_start]**Architectural Comparison**: Analyzed the theoretical and practical trade-offs between 3D Gaussian Splatting and traditional Neural Radiance Fields (NeRFs)[cite: 4611].

## Environment Setup
To reproduce the environment and run the inference scripts, ensure you have a CUDA-enabled GPU and run:
```bash
# Core dependencies
pip install torch torchvision
# Please refer to the specific module requirements (DUSt3R and InstantSplat) for detailed package versions.
