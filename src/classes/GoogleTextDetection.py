
import cv2
from google.cloud import vision
from typing import Optional, List, Dict, Any


class GoogleTextDetection:
    def __init__(self):
        self._vision_client = None

    @property
    def vision_client(self) -> vision.ImageAnnotatorClient:
        """Lazy initialization of Vision API client."""
        if self._vision_client is None:
            self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    def _convert_image_to_bytes(self, image_array) -> bytes:
        """Convert OpenCV image array to bytes format for Vision API."""
        success, buffer = cv2.imencode('.jpg', image_array)
        if not success:
            raise ValueError("Failed to encode image to bytes")
        return buffer.tobytes()

    def _extract_text_annotations(self, vision_image) -> List:
        """Extract text annotations from Vision API response."""
        response = self.vision_client.text_detection(image=vision_image)

        if response.error.message:
            raise Exception(
                f'Vision API Error: {response.error.message}\n'
                f'For more information: https://cloud.google.com/apis/design/errors'
            )

        return response.text_annotations

    def _parse_text_details(self, text_annotations: List) -> Dict[str, Any]:
        """Parse detailed text information including positions."""
        if not text_annotations:
            return {"full_text": None, "text_details": []}

        full_text = text_annotations[0].description
        text_details = []

        # Skip first annotation (full text) and process individual words/phrases
        for annotation in text_annotations[1:]:
            vertices = [
                f"({vertex.x},{vertex.y})"
                for vertex in annotation.bounding_poly.vertices
            ]

            text_detail = {
                "text": annotation.description,
                "bounding_box": vertices,
                "coordinates": ",".join(vertices)
            }
            text_details.append(text_detail)

        return {
            "full_text": full_text,
            "text_details": text_details
        }

    def detect_text(self, image_array) -> Optional[str]:
        """
        Main method to perform text detection on the given image.

        Args:
            image_array: OpenCV image array

        Returns:
            Extracted text or None if no text found
        """
        try:
            # Convert image to bytes
            image_bytes = self._convert_image_to_bytes(image_array)

            # Create Vision API image object
            vision_image = vision.Image(content=image_bytes)

            # Get text annotations
            text_annotations = self._extract_text_annotations(vision_image)

            # Parse text details
            parsed_results = self._parse_text_details(text_annotations)

            return parsed_results["full_text"]

        except Exception as e:
            print(f"Error during text detection: {str(e)}")
            raise