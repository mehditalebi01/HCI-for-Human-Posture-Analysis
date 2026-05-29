# Stage 04: OpenCap-Style Biomechanical Feature Extraction

Stage 4 computes OpenCap-inspired 2D biomechanical surrogate features from smoothed pose keypoints.

## Important Scope

This stage does not implement the full OpenCap system. It does not estimate calibrated 3D kinematics or dynamics. It computes simple 2D image-plane features that are useful for prototyping downstream biomechanics, action recognition, and risk-scoring stages.

## Current Implementation

- Reusable geometry helpers: `src/hpa/biomechanics/angles.py`
- Feature extraction: `src/hpa/biomechanics/opencap_features.py`
- CLI entry point: `src/scripts/extract_opencap_style_features.py`

## Standard Command

```bash
python src/scripts/extract_opencap_style_features.py --input data/processed/smoothed_keypoints/smoothed_keypoints.csv --output data/processed/biomechanics/opencap_style_features.csv --plots-dir data/outputs/reports/opencap_style_plots --confidence-threshold 0.3
```

## Features

- left knee angle
- right knee angle
- left hip angle
- right hip angle
- trunk lean angle
- knee angle asymmetry
- hip angle asymmetry
- mean pose confidence
- low confidence flag

## Future Work

Full OpenCap or Pose2Sim integration requires calibrated multi-camera data or the OpenCap capture workflow. The current stage should be treated as 2D surrogate feature extraction, not validated clinical biomechanics.

## References

- Uhlrich, S. D. et al. OpenCap: Human movement dynamics from smartphone videos. Nature Biomedical Engineering, 2023.
- Pagnon, D., Domalain, M., & Reveret, L. Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics. Sensors, 2021.
- Drazan, J. F. et al. Accuracy Assessment of Joint Angles Estimated from 2D and 3D Camera Measurements. Sensors, 2022.
