import cv2
import numpy as np
from google.cloud import vision
from typing import List

from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.model import BoundingBox
from capsules.GcpVision.src.models.PackageModel import Detection


class GoogleLogoDetection:
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

    def _detect_logos(self, image_bytes: bytes) -> List[vision.EntityAnnotation]:
        """Call Vision API to detect logos."""
        image = vision.Image(content=image_bytes)
        response = self.vision_client.logo_detection(image=image)

        if response.error.message:
            raise Exception(
                f"Vision API Error: {response.error.message}\n"
                "For more info: https://cloud.google.com/apis/design/errors"
            )
        print("Logo detection response received")
        return response.logo_annotations

    def detect_logos(self) -> List[str]:
        """
        Detect logos and return list of logo descriptions.

        Returns:
            List[str]: Detected logo descriptions.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            logos = self._detect_logos(image_bytes)
            descriptions = [logo.description for logo in logos]
            print("Detected logos:", descriptions)
            return descriptions
        except Exception as e:
            self.logger.info(f"Error during logo detection: {str(e)}")
            raise

    def detect_logos_as_detections(self):
        """
        Detect logos and add them as Detection objects to self.obj.detections.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            logos = self._detect_logos(image_bytes)

            for logo in logos:
                if logo.bounding_poly and logo.bounding_poly.vertices:
                    vertices = logo.bounding_poly.vertices
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    width = x_max - x_min
                    height = y_max - y_min

                    bbox = BoundingBox(
                        left=int(x_min),
                        top=int(y_min),
                        width=int(width),
                        height=int(height)
                    )
                else:
                    # Eğer bounding box yoksa tüm resmi bbox olarak alabiliriz ya da atlayabiliriz.
                    height, width = self.image.shape[:2]
                    bbox = BoundingBox(left=0, top=0, width=width, height=height)

                self.obj.detections.append(
                    Detection(
                        boundingBox=bbox,
                        # Logo'nun bulunduğu alanı temsil eden BoundingBox objesi (koordinatlar ve boyutlar)
                        confidence=logo.score if hasattr(logo, 'score') else 0.0,
                        # Logonun tespit güven skoru (varsa), yoksa 0.0
                        classId=-1,  # Logo sınıf kimliği, burada -1 olarak atanıyor (tanımlı bir sınıf ID'si yok)
                        classLabel=logo.description if logo.description else "",
                        # Logo ismi (örneğin "Nike"), yoksa boş string
                        imgUID=self.uID  # Bu tespitin ait olduğu görselin benzersiz kimliği (UID)
                    )
                )
            print(f"Added {len(logos)} logo detections.")
        except Exception as e:
            self.logger.info(f"Error during logos to detections: {str(e)}")
            raise
