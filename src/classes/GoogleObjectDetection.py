import cv2
import numpy as np
from google.cloud import vision
from typing import List, Dict, Any

from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.model import BoundingBox
from capsules.GcpVision.src.models.PackageModel import Detection


class GoogleObjectDetection:
    def __init__(self, obj, image: np.ndarray, uID: str):
        self.obj = obj
        self.image = image
        self.uID = uID
        self.logger = LoggerManager()
        self._vision_client = None

    @property
    def vision_client(self) -> vision.ImageAnnotatorClient:
        if self._vision_client is None:
            self._vision_client = vision.ImageAnnotatorClient()
            print("Vision API client initialized")
        return self._vision_client

    def _convert_image_to_bytes(self, image_array: np.ndarray) -> bytes:
        success, buffer = cv2.imencode('.jpg', image_array)
        if not success:
            raise ValueError("Failed to encode image to bytes")
        print("Converted image to bytes")
        return buffer.tobytes()

    def _localize_objects(self, image_bytes: bytes) -> List[vision.LocalizedObjectAnnotation]:
        image = vision.Image(content=image_bytes)
        response = self.vision_client.object_localization(image=image)

        if response.error.message:
            raise Exception(
                f"Vision API Error: {response.error.message}\n"
                "For more info: https://cloud.google.com/apis/design/errors"
            )
        print(f"Object localization response received, {len(response.localized_object_annotations)} objects found")
        return response.localized_object_annotations

    def detect_objects(self) -> List[Dict[str, Any]]:
        """
        Returns list of detected objects with name, confidence and bounding box coordinates.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            objects = self._localize_objects(image_bytes)

            results = []
            for obj in objects:
                # Normalized bounding box vertices (x,y in [0,1])
                normalized_vertices = obj.bounding_poly.normalized_vertices
                # Convert normalized coordinates to absolute pixel values
                height, width = self.image.shape[:2]
                bounding_box = [
                    {"x": int(vertex.x * width), "y": int(vertex.y * height)}
                    for vertex in normalized_vertices
                ]

                results.append({
                    "name": obj.name,
                    "confidence": obj.score,
                    "bounding_poly": bounding_box
                })
            print("Formatted object localization results")
            return results

        except Exception as e:
            self.logger.info(f"Error during object localization: {str(e)}")
            raise

    def detect_objects_as_detections(self):
        """
        Detect objects and add them as Detection objects to self.obj.detections.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            objects = self._localize_objects(image_bytes)

            height, width = self.image.shape[:2]

            for obj in objects:
                normalized_vertices = obj.bounding_poly.normalized_vertices
                x_coords = [int(vertex.x * width) for vertex in normalized_vertices]
                y_coords = [int(vertex.y * height) for vertex in normalized_vertices]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                bbox = BoundingBox(
                    left=x_min,
                    top=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min
                )

                self.obj.detections.append(
                    Detection(
                        boundingBox=bbox,
                        confidence=obj.score,
                        classId=-1,
                        classLabel=obj.name,
                        imgUID=self.uID
                    )
                )
            print(f"Added {len(objects)} object detections.")
        except Exception as e:
            self.logger.info(f"Error during object detections creation: {str(e)}")
            raise
