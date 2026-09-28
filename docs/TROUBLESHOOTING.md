# Troubleshooting Guide

### 1. PyTorch / CUDA Warnings
- **Issue**: PyTorch prints warnings regarding CUDA device unavailability.
- **Solution**: AdaptiveRL uses CPU by default. You can run training entirely on CPU without any GPU hardware.

### 2. Gym / Gymnasium Compatibility
- **Issue**: `AttributeError: module 'gym' has no attribute 'Env'`
- **Solution**: AdaptiveRL uses modern `gymnasium>=1.0.0`. Ensure legacy `gym` is uninstalled.

### 3. Training Reproducibility
- **Issue**: Evaluation results differ slightly across different OS platforms.
- **Solution**: Pass an explicit integer seed (e.g. `--seed 42`) to lock random number generators for both Gymnasium and PyTorch.
