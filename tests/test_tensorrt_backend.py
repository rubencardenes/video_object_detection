import json
from pathlib import Path
import tempfile
import unittest

from video_tools.detection.model import list_detection_models, resolve_model_path
from video_tools.detection.tensorrt_backend import engine_payload
from video_tools.detection.timing import InferenceTiming


class TensorRTSupportTests(unittest.TestCase):
    def test_raw_and_ultralytics_engine_files(self):
        raw = b"ftrt" + bytes(100)
        self.assertEqual(engine_payload(raw), raw)
        header = json.dumps({"imgsz": [640, 640], "task": "detect"}).encode()
        wrapped = len(header).to_bytes(4, "little") + header + raw
        self.assertEqual(engine_payload(wrapped), raw)
        invalid = (4).to_bytes(4, "little") + b"nope" + raw
        self.assertEqual(engine_payload(invalid), invalid)

    def test_model_discovery_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["yolo.engine", "other.plan", "yolo.onnx", "yolo.pt"]:
                (root / name).touch()
            self.assertEqual(list_detection_models(root), ["other.plan", "yolo.engine", "yolo.onnx"])
            self.assertEqual(resolve_model_path("yolo.engine", root), root / "yolo.engine")
            self.assertEqual(resolve_model_path(None, root), root / "yolo.onnx")
            (root / "yolo.onnx").unlink()
            self.assertEqual(resolve_model_path(None, root), root / "other.plan")

    def test_backend_label(self):
        timing = InferenceTiming("TensorRT")
        self.assertIn("TensorRT:", timing.label())
        for _ in range(15):
            timing.add(7)
        self.assertIn("TensorRT: 7.00 ms", timing.label())


if __name__ == "__main__":
    unittest.main()
