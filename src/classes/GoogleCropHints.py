import cv2
import numpy as np
from google.cloud import vision
from typing import List, Dict, Any

from sdks.novavision.src.base.model import BoundingBox
from capsules.GcpVision.src.models.PackageModel import Detection
from sdks.novavision.src.base.logger import LoggerManager


class GoogleCropHints:
    def __init__(self, obj, image: np.ndarray, uID: str):
        self.obj = obj
        self.image = image
        self.uID = uID
        self.logger = LoggerManager()
        self._vision_client = None

    @property
    def vision_client(self) -> vision.ImageAnnotatorClient:
        """Lazy initialization for Vision API client."""
        if self._vision_client is None:
            self._vision_client = vision.ImageAnnotatorClient()
            print("Vision API client initialized")
        return self._vision_client

    def _convert_image_to_bytes(self, image_array: np.ndarray) -> bytes:
        """Convert OpenCV image array to bytes."""
        success, buffer = cv2.imencode('.jpg', image_array)
        if not success:
            raise ValueError("Failed to encode image to bytes")
        print("Converted image to bytes")
        return buffer.tobytes()

    def _detect_crop_hints(self, image_bytes: bytes) -> List[vision.CropHint]:
        """Call Vision API to get crop hints."""
        image = vision.Image(content=image_bytes)
        print("self.obj.aspect_ratios:",self.obj.aspect_ratios)
        crop_hints_params = vision.CropHintsParams(aspect_ratios=[self.obj.aspect_ratios])
        image_context = vision.ImageContext(crop_hints_params=crop_hints_params)

        response = self.vision_client.crop_hints(image=image, image_context=image_context)

        if response.error.message:
            raise Exception(
                f"Vision API Error: {response.error.message}\n"
                "For more info: https://cloud.google.com/apis/design/errors"
            )
        print("Crop hints response")
        return response.crop_hints_annotation.crop_hints

    def _format_results(self, hints: List[vision.CropHint]) -> List[Dict[str, Any]]:
        """Convert raw crop hints into dictionary format."""
        results = []
        for hint in hints:
            bounding_box = [
                {"x": vertex.x, "y": vertex.y}
                for vertex in hint.bounding_poly.vertices
            ]
            results.append({"bounding_poly": bounding_box})
            print("_format_results")
        return results

    def detect_crop_hints(self) -> List[Dict[str, Any]]:
        """
        Detect crop hints and return them as dictionaries.

        Returns:
            List[Dict[str, Any]]: List of crop hint bounding boxes.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            hints = self._detect_crop_hints(image_bytes)
            return self._format_results(hints)
        except Exception as e:
            self.logger.info(f"Error during crop hints detection: {str(e)}")
            raise

    def detect_crop_hints_as_detections(self):
        """
        Detect crop hints and return them as Detection objects with empty class/confidence fields.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            hints = self._detect_crop_hints(image_bytes)

            for hint in hints:
                vertices = hint.bounding_poly.vertices
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                width = x_max - x_min
                height = y_max - y_min

                # BoundingBox oluştur
                bbox = BoundingBox(
                    left=int(x_min),
                    top=int(y_min),
                    width=int(width),
                    height=int(height)
                )

                self.obj.detections.append(
                    Detection(
                        boundingBox=bbox,
                        confidence=0.0,
                        classId=-1,
                        classLabel="",
                        imgUID=self.uID
                    )
                )
        except Exception as e:
            self.logger.info(f"Error during crop hints to detections: {str(e)}")
            raise


