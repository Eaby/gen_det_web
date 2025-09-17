# test_import.py
from controllers.video_controller import _get_mixed_detector

det = _get_mixed_detector(device="cpu")
print("✅ Loaded MixedModelDetector OK")
