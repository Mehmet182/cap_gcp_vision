import cv2
import numpy as np
from google.cloud import vision
from sdks.novavision.src.base.logger import LoggerManager
from capsules.GcpVision.src.models.PackageModel import DominantColor , RGBColor

class GoogleImageProperties:
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
            print("Vision API client initialized (for image properties)")
        return self._vision_client

    def _convert_image_to_bytes(self, image_array: np.ndarray) -> bytes:
        """Convert OpenCV image array to bytes."""
        success, buffer = cv2.imencode('.jpg', image_array)
        if not success:
            raise ValueError("Failed to encode image to bytes")
        return buffer.tobytes()

    def detect_image_properties(self) -> list:
        """Detects dominant colors and returns a list of color dicts."""

        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            image = vision.Image(content=image_bytes)

            response = self.vision_client.image_properties(image=image)

            if response.error.message:
                raise Exception(
                    f"Vision API Error: {response.error.message}\n"
                    "For more info: https://cloud.google.com/apis/design/errors"
                )

            props = response.image_properties_annotation
            dominant_colors = []

            for color_info in props.dominant_colors.colors:
                rgb = RGBColor(
                    red=color_info.color.red,
                    green=color_info.color.green,
                    blue=color_info.color.blue,
                    alpha=getattr(color_info.color, "alpha", 1.0)
                )

                color = DominantColor(
                    color=rgb,
                    score=color_info.score,
                    pixelFraction=color_info.pixel_fraction
                )

                dominant_colors.append(color)

            return dominant_colors

        except Exception as e:
            self.logger.info(f"Error during image properties detection: {str(e)}")
            raise


