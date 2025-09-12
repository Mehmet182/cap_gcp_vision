import cv2
import numpy as np
from google.cloud import vision
from typing import List
from sdks.novavision.src.base.logger import LoggerManager
from capsules.GcpVision.src.models.PackageModel import WebDetection , BestGuessLabel, WebImage,WebPage,WebEntity


class GoogleWebDetection:
    def __init__(self, obj, image: np.ndarray):
        self.obj = obj
        self.image = image
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
        return buffer.tobytes()

    def detect_web(self) -> WebDetection:
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            image = vision.Image(content=image_bytes)
            response = self.vision_client.web_detection(image=image)

            if response.error.message:
                raise Exception(f"Vision API Error: {response.error.message}")

            annotations = response.web_detection

            # Model objesi için verileri dönüştürme
            best_guess_labels = [
                BestGuessLabel(label=label.label)
                for label in annotations.best_guess_labels
            ] if annotations.best_guess_labels else []

            pages_with_matching_images = []
            if annotations.pages_with_matching_images:
                for page in annotations.pages_with_matching_images:
                    full_imgs = [WebImage(url=img.url) for img in page.full_matching_images] if page.full_matching_images else []
                    partial_imgs = [WebImage(url=img.url) for img in page.partial_matching_images] if page.partial_matching_images else []
                    pages_with_matching_images.append(
                        WebPage(
                            url=page.url,
                            full_matching_images=full_imgs,
                            partial_matching_images=partial_imgs,
                        )
                    )

            web_entities = [
                WebEntity(score=entity.score, description=entity.description)
                for entity in annotations.web_entities
            ] if annotations.web_entities else []

            visually_similar_images = [
                WebImage(url=img.url)
                for img in annotations.visually_similar_images
            ] if annotations.visually_similar_images else []

            result = WebDetection(
                best_guess_labels=best_guess_labels,
                pages_with_matching_images=pages_with_matching_images,
                web_entities=web_entities,
                visually_similar_images=visually_similar_images
            )

            print("Web detection result:", result.json(indent=2))
            return result

        except Exception as e:
            self.logger.info(f"Error during web detection: {str(e)}")
            raise
