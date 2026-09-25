import unittest
from unittest.mock import Mock

import numpy as np

from video_tools.detection.model import DetectionModel, _class_aware_nms


class DetectionModelTests(unittest.TestCase):
    def test_rgb_preprocessing_is_contiguous_and_letterboxed(self):
        model = DetectionModel.__new__(DetectionModel)
        model._height = model._width = 640
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        tensor, scale, padding = model._preprocess_yolo(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(scale, 3.2)
        self.assertEqual(padding, (0, 160))
        np.testing.assert_allclose(tensor[0, :, 200, 100], [1, 0, 0])
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)

    def test_nms_preserves_classes_and_sorts_scores(self):
        boxes = np.array([[0, 0, 10, 10]] * 3, dtype=np.float32)
        selected = _class_aware_nms(
            boxes, np.array([0.8, 0.9, 0.95]), np.array([0, 0, 1]), 0.7
        )
        np.testing.assert_array_equal(selected, [2, 1])

    def test_border_boxes_are_suppressed_before_clipping(self):
        model = DetectionModel.__new__(DetectionModel)
        model._height = model._width = 100
        # These overlap at IoU 0.5 before clipping, but become identical after it.
        predictions = np.zeros((1, 84, 100), dtype=np.float32)
        predictions[0, :5, 0] = [0, 50, 20, 20, 0.9]
        predictions[0, :5, 1] = [5, 50, 10, 20, 0.8]
        model._run_inference = Mock(return_value=[predictions])
        detections = model._infer_yolo(np.zeros((100, 100, 3), np.uint8), 0.01)
        self.assertEqual(len(detections), 2)
        np.testing.assert_allclose(detections.xyxy, [[0, 40, 10, 60]] * 2)

    def test_cpu_execution_fallback(self):
        model = DetectionModel.__new__(DetectionModel)
        model._io_binding = None
        model._session = Mock()
        model._input_name = "images"
        output = model._run_inference(np.zeros((1, 3, 10, 10), np.float32))
        self.assertIs(output, model._session.run.return_value)
        self.assertGreaterEqual(model.last_inference_ms, 0)


if __name__ == "__main__":
    unittest.main()
