"""RTMLib-based 2D human body pose estimation."""

import os
import site
from pathlib import Path

import cv2
import numpy as np

from hpa.utils.logging_utils import get_logger
from hpa.utils.paths import list_images

_DLL_DIRECTORY_HANDLES = []


class RTMPoseEstimator:
    """Small wrapper around RTMLib's Body pose estimator.

    The wrapper keeps RTMLib setup in one place and makes scripts easier to read.
    """

    def __init__(
        self,
        device="cpu",
        backend="onnxruntime",
        mode="balanced",
        det_model_path=None,
        pose_model_path=None,
        det_input_size=(640, 640),
        pose_input_size=(192, 256),
        openpose_skeleton=False,
    ):
        self.device = device
        self.backend = backend
        self.mode = mode
        self.det_model_path = self._existing_model_path(det_model_path, "detection")
        self.pose_model_path = self._existing_model_path(pose_model_path, "pose")
        self.det_input_size = det_input_size
        self.pose_input_size = pose_input_size
        self.openpose_skeleton = openpose_skeleton
        self.logger = get_logger(self.__class__.__name__)
        self._preload_cuda_dlls()
        self._warn_if_cuda_provider_is_missing()

        try:
            from rtmlib import Body
        except ImportError as exc:
            raise ImportError(
                "rtmlib is not installed. Run: pip install -r requirements.txt"
            ) from exc

        if self.det_model_path and self.pose_model_path:
            self.logger.info("Using local RTMLib models from the models/ directory.")
        else:
            self.logger.warning(
                "Local model files were not fully found. RTMLib may download default "
                "models on first use. Run: python src/scripts/download_models.py"
            )

        self.model = Body(
            det=str(self.det_model_path) if self.det_model_path else None,
            det_input_size=self.det_input_size,
            pose=str(self.pose_model_path) if self.pose_model_path else None,
            pose_input_size=self.pose_input_size,
            mode=self.mode,
            to_openpose=self.openpose_skeleton,
            backend=self.backend,
            device=self.device,
        )

    def _preload_cuda_dlls(self):
        """Preload CUDA DLLs when using ONNX Runtime on Windows.

        ONNX Runtime can load CUDA/cuDNN DLLs from NVIDIA pip packages, but this
        needs to happen before RTMLib creates its inference sessions.
        """
        if self.backend != "onnxruntime" or self.device != "cuda":
            return

        try:
            import onnxruntime as ort
        except ImportError:
            return

        # On Windows, dependent cuDNN sub-libraries must also be discoverable
        # later during inference. Keep the handles alive for the full process.
        if hasattr(os, "add_dll_directory"):
            for site_package_dir in site.getsitepackages():
                nvidia_dir = Path(site_package_dir) / "nvidia"
                for bin_dir in nvidia_dir.glob("*\\bin"):
                    if bin_dir.exists():
                        handle = os.add_dll_directory(str(bin_dir))
                        _DLL_DIRECTORY_HANDLES.append(handle)

        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            return

        try:
            preload(directory="")
        except Exception as error:
            self.logger.warning("Could not preload CUDA DLLs: %s", error)

    def _warn_if_cuda_provider_is_missing(self):
        """Warn when CUDA is requested but ONNX Runtime cannot see it."""
        if self.backend != "onnxruntime" or self.device != "cuda":
            return

        try:
            import onnxruntime as ort
        except ImportError:
            self.logger.warning("onnxruntime is not installed.")
            return

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            self.logger.warning(
                "CUDA device was requested, but ONNX Runtime does not list "
                "CUDAExecutionProvider. Available providers: %s",
                providers,
            )

    @staticmethod
    def _existing_model_path(model_path, model_type):
        """Return a Path only when the requested model file exists."""
        if model_path is None:
            return None

        path = Path(model_path)
        if path.exists():
            return path

        print(f"Warning: local {model_type} model not found: {path}")
        return None

    @staticmethod
    def _normalize_outputs(keypoints, scores):
        """Convert outputs to predictable NumPy array shapes."""
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)

        if keypoints.size > 0 and scores.size > 0 and keypoints.ndim == 2:
            keypoints = keypoints[None, :, :]
            scores = scores[None, :]

        return keypoints, scores

    def estimate_image(self, image):
        """Estimate body keypoints for one image array."""
        keypoints, scores = self.model(image)
        return self._normalize_outputs(keypoints, scores)

    def estimate_folder(self, input_dir):
        """Estimate body keypoints for all images in a folder.

        Returns a list of dictionaries with frame path, image, keypoints, and scores.
        """
        results = []

        for image_path in list_images(input_dir):
            image = cv2.imread(str(image_path))

            if image is None:
                self.logger.warning("Could not read image, skipping: %s", image_path)
                continue

            keypoints, scores = self.estimate_image(image)
            results.append(
                {
                    "frame_path": image_path,
                    "image": image,
                    "keypoints": keypoints,
                    "scores": scores,
                }
            )

        return results
