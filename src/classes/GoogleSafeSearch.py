import cv2
import numpy as np
from google.cloud import vision
from typing import Dict, Any

from sdks.novavision.src.base.logger import LoggerManager
from capsules.GcpVision.src.models.PackageModel import SafeSearchResult


class GoogleSafeSearch:
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

    def detect_safe_search(self) -> SafeSearchResult:
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            image = vision.Image(content=image_bytes)

            response = self.vision_client.safe_search_detection(image=image)

            if response.error.message:
                raise Exception(
                    f"Vision API Error: {response.error.message}\n"
                    "For more info: https://cloud.google.com/apis/design/errors"
                )

            safe = response.safe_search_annotation

            likelihood_name = (
                "UNKNOWN",
                "VERY_UNLIKELY",
                "UNLIKELY",
                "POSSIBLE",
                "LIKELY",
                "VERY_LIKELY",
            )

            results = SafeSearchResult(
                adult=likelihood_name[safe.adult],
                medical=likelihood_name[safe.medical],
                spoofed=likelihood_name[safe.spoof],
                violence=likelihood_name[safe.violence],
                racy=likelihood_name[safe.racy],
            )

            print("Safe search results:", results.dict())
            return results

        except Exception as e:
            self.logger.info(f"Error during safe search detection: {str(e)}")
            raise


