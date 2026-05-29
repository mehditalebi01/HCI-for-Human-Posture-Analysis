# Computer Vision Pipeline for Sports Posture Analysis

## Abstract

This project implements a modular computer vision pipeline for sports posture analysis using video input, frame extraction, 2D pose estimation, temporal smoothing, quality checking, and OpenCap-inspired 2D biomechanical feature extraction; the current system should be interpreted as a research prototype, not as a full clinical motion analysis or full OpenCap replacement [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), [RTMPose](https://arxiv.org/abs/2303.07399), [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/).

The pipeline uses RTMLib because it provides a lightweight interface to RTMPose and YOLOX-style person detection without requiring a full MMPose project setup, which makes the implementation practical for a student research environment and reproducible command-line experiments [RTMLib](https://github.com/Tau-J/rtmlib), [MMPose](https://mmpose.readthedocs.io/), [YOLOX](https://arxiv.org/abs/2107.08430).

## System Architecture

The repository is organized as a layered pipeline: reusable modules live under `src/hpa/`, while executable scripts live under `src/scripts/`, which separates core scientific logic from command-line execution and makes each stage independently testable [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

```text
data/raw/videos
  -> src/scripts/extract_frames.py
  -> src/scripts/run_pose_on_frames.py
  -> src/scripts/smooth_keypoints.py
  -> src/scripts/extract_opencap_style_features.py
  -> data/processed and data/outputs
```

This staged design is appropriate because video analysis, pose estimation, signal smoothing, and biomechanical feature extraction have different failure modes and should be inspected separately before drawing conclusions from downstream features [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), [pandas Rolling](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html), [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/).

## Stage 1: Video Input and Frame Extraction

The first stage reads a video with OpenCV and saves frames to `data/interim/frames`, using `cv2.VideoCapture` because it is the standard OpenCV interface for frame-by-frame video processing [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html).

```python
from hpa.io.video_io import extract_frames

result = extract_frames(
    video_path="data/raw/videos/test1.mp4",
    output_dir="data/interim/frames",
    step=30,
)
```

The default `step=30` is a practical baseline that reduces computational cost by sampling frames instead of processing every video frame, which is useful before running heavier pose estimation models [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html).

## Stage 2: 2D Human Pose Estimation

The second stage estimates 2D human body keypoints using RTMLib's `Body` wrapper, which internally performs human detection and pose estimation before returning keypoints and confidence scores [RTMLib](https://github.com/Tau-J/rtmlib).

```python
from hpa.pose.pose_estimator import RTMPoseEstimator

estimator = RTMPoseEstimator(
    device="cuda",
    backend="onnxruntime",
    mode="lightweight",
)

keypoints, scores = estimator.estimate_image(image)
```

RTMPose was selected because it is designed for real-time multi-person pose estimation and is part of the MMPose ecosystem, while YOLOX is used by RTMLib as a detector because it is a strong anchor-free object detection model with good speed-accuracy tradeoffs [RTMPose](https://arxiv.org/abs/2303.07399), [MMPose](https://mmpose.readthedocs.io/), [YOLOX](https://arxiv.org/abs/2107.08430).

The project uses ONNX Runtime because RTMLib runs ONNX models efficiently and can use `CUDAExecutionProvider` on NVIDIA GPUs, which is important for live visualization and faster batch inference [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Model and Parameter Choices

RTMLib exposes different model presets, and the project uses `mode="lightweight"` for live inference because it prioritizes FPS over maximum accuracy, while `balanced` is more appropriate when accuracy is more important than speed [RTMLib](https://github.com/Tau-J/rtmlib).

| Mode | Detector family | Pose family | Practical use |
|---|---|---|---|
| `lightweight` | YOLOX-Tiny style | RTMPose-S style | Live demo and fast iteration |
| `balanced` | YOLOX-M style | RTMPose-M style | Better accuracy-speed compromise |
| `performance` | Larger YOLOX/RTMPose models | Larger RTMPose style | Higher accuracy, lower FPS |

The live script also uses `--max-width 640` because resizing large frames before inference reduces GPU/CPU workload, although it can reduce small-joint localization accuracy if the subject becomes too small [OpenCV Resize](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html), [RTMPose](https://arxiv.org/abs/2303.07399).

## Stage 3: Pose Smoothing and Quality Checking

The third stage smooths the `x` and `y` coordinates per `person_id` and `keypoint_id` using a centered moving average, which is a simple and interpretable method for reducing frame-to-frame jitter in pose trajectories [pandas Rolling](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html).

```python
smoothed["x"] = smoothed.groupby(["person_id", "keypoint_id"])["x"].transform(
    lambda values: values.rolling(window=5, min_periods=1, center=True).mean()
)
```

Confidence is intentionally not smoothed because it is a model quality signal, and preserving it allows later stages to flag unreliable poses instead of hiding low-confidence detections inside the smoothing process [RTMPose](https://arxiv.org/abs/2303.07399).

The quality report currently measures total keypoints, average confidence, low-confidence count, low-confidence ratio, and processed frame count, which provides a minimal but useful quality screen before computing biomechanical features [pandas Documentation](https://pandas.pydata.org/docs/).

## Stage 4: OpenCap-Inspired Biomechanical Feature Extraction

The fourth stage computes OpenCap-inspired 2D surrogate features from smoothed keypoints, including knee angles, hip angles, trunk lean, left-right angle differences, mean pose confidence, and a low-confidence flag [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/), [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

```python
left_knee_angle = angle_between_three_points(
    left_hip,
    left_knee,
    left_ankle,
)
```

These features are explicitly not full OpenCap outputs because OpenCap estimates motion dynamics through a dedicated capture workflow and 3D musculoskeletal analysis, while this project currently computes 2D image-plane surrogate features from monocular keypoints [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/), [2D/3D Joint Angle Accuracy](https://www.mdpi.com/1424-8220/22/5/1729).

## Current Experimental Results

The latest video run produced 15,096 keypoint rows across 888 frames with one detected person, which is a much cleaner input condition than a multi-person scene because no cross-person identity mixing is visible in the processed CSV [RTMLib](https://github.com/Tau-J/rtmlib), [pandas Documentation](https://pandas.pydata.org/docs/).

| Metric | Value |
|---|---:|
| Frames processed | 888 |
| Keypoint rows | 15,096 |
| Detected person IDs | 1 |
| Average confidence | 0.8437 |
| Low-confidence keypoints | 0 |
| Low-confidence ratio | 0.0000 |

The confidence metrics suggest that the selected video is suitable for downstream feature extraction because the mean keypoint confidence is high and no keypoints fall below the configured threshold of 0.3 [RTMPose](https://arxiv.org/abs/2303.07399).

| Feature | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| Left knee angle | 121.962 | 34.419 | 64.077 | 131.240 | 171.589 |
| Right knee angle | 139.678 | 26.516 | 67.815 | 151.767 | 170.573 |
| Left hip angle | 165.653 | 9.381 | 144.927 | 167.220 | 179.665 |
| Right hip angle | 167.097 | 9.846 | 144.159 | 168.948 | 179.981 |
| Trunk lean angle | 1.214 | 0.752 | 0.000 | 1.176 | 2.769 |
| Knee angle difference | 49.033 | 27.972 | 0.064 | 51.134 | 91.522 |
| Hip angle difference | 15.008 | 9.020 | 0.079 | 12.950 | 34.542 |

The trunk lean result is biomechanically plausible for a treadmill-style running video because the trunk remains nearly vertical, while the large instantaneous knee angle difference should be interpreted as a phase-dependent left-right difference during gait rather than clinical asymmetry [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/), [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

## Visual Results

The annotated pose frames show that the estimated skeleton follows the visible body landmarks well in the latest single-person treadmill video, which supports the high confidence values reported by the quality stage [RTMPose](https://arxiv.org/abs/2303.07399), [RTMLib](https://github.com/Tau-J/rtmlib).

![Annotated pose frame 000014](../data/outputs/annotated_frames/frame_000014.jpg)

The second example frame shows a different gait phase, where one knee is flexed and the other is more extended, explaining why instantaneous left-right knee differences can be high even when the pose estimate is visually reasonable [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/), [2D/3D Joint Angle Accuracy](https://www.mdpi.com/1424-8220/22/5/1729).

![Annotated pose frame 000375](../data/outputs/annotated_frames/frame_000375.jpg)

The knee angle plot shows periodic changes consistent with running or treadmill gait, but it should not be interpreted as validated 3D knee kinematics because it is computed from 2D image-plane keypoints [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/), [2D/3D Joint Angle Accuracy](https://www.mdpi.com/1424-8220/22/5/1729).

![Knee angle plot](../data/outputs/reports/opencap_style_plots/knee_angles.png)

The hip angle plot is smoother and more bounded than the knee plot, which is expected because the torso and pelvis show less rapid apparent motion than the distal lower limb in this side-view sequence [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

![Hip angle plot](../data/outputs/reports/opencap_style_plots/hip_angles.png)

The trunk lean plot remains close to zero degrees throughout the video, which matches the visual observation of an upright treadmill-running posture and supports the internal consistency of the extracted trunk feature [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/).

![Trunk lean plot](../data/outputs/reports/opencap_style_plots/trunk_lean.png)

The asymmetry plot should currently be interpreted as instantaneous left-right angle difference, not as clinical gait asymmetry, because the project does not yet segment gait cycles or normalize left and right limb phases [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/), [2D/3D Joint Angle Accuracy](https://www.mdpi.com/1424-8220/22/5/1729).

![Asymmetry plot](../data/outputs/reports/opencap_style_plots/asymmetry.png)

## Limitations

The current pipeline does not yet perform true temporal tracking, so `person_id` is a per-frame detection index rather than a validated identity over time; this is acceptable for the latest single-person video but not sufficient for crowded sports scenes [RTMLib](https://github.com/Tau-J/rtmlib).

The current biomechanical features are 2D surrogate measures, so camera perspective, out-of-plane motion, body self-occlusion, and image scaling can bias the estimated angles compared with calibrated 3D motion analysis [2D/3D Joint Angle Accuracy](https://www.mdpi.com/1424-8220/22/5/1729), [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

The detector bounding boxes are not yet exported separately because RTMLib's `Body` wrapper hides the detection stage behind the pose API, so future work should separate detection output if tracking and detection quality analysis are required [RTMLib](https://github.com/Tau-J/rtmlib), [YOLOX](https://arxiv.org/abs/2107.08430).

## Recommended Next Steps

The next technical step should be to add tracking, because stable `track_id` values are required before multi-person videos can be used for longitudinal posture and biomechanics analysis [RTMLib PoseTracker](https://github.com/Tau-J/rtmlib), [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

The next scientific step should be to replace instantaneous left-right angle differences with gait-cycle-aware metrics, because running naturally alternates limb phases and high same-frame knee differences do not automatically imply pathological asymmetry [Pose2Sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/).

The next validation step should be manual annotation or comparison against a calibrated system, because visual inspection and confidence scores are useful but do not provide quantitative ground truth for joint angle accuracy [2D/3D Joint Angle Accuracy](https://www.mdpi.com/1424-8220/22/5/1729), [OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/).

## References

1. Uhlrich, S. D. et al. OpenCap: Human movement dynamics from smartphone videos. Nature Biomedical Engineering, 2023. https://pubmed.ncbi.nlm.nih.gov/37856442/
2. Pagnon, D., Domalain, M., and Reveret, L. Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics. Sensors, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/
3. Jiang, T. et al. RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. arXiv, 2023. https://arxiv.org/abs/2303.07399
4. Ge, Z. et al. YOLOX: Exceeding YOLO Series in 2021. arXiv, 2021. https://arxiv.org/abs/2107.08430
5. RTMLib GitHub repository. https://github.com/Tau-J/rtmlib
6. ONNX Runtime CUDA Execution Provider documentation. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
7. OpenCV video display and capture documentation. https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
8. pandas rolling window documentation. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
9. Matplotlib documentation. https://matplotlib.org/stable/
10. Drazan, J. F. et al. Accuracy Assessment of Joint Angles Estimated from 2D and 3D Camera Measurements. Sensors, 2022. https://www.mdpi.com/1424-8220/22/5/1729
