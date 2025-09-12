import cv2
import numpy as np
from google.cloud import vision

from sdks.novavision.src.base.model import BoundingBox
from capsules.GcpVision.src.models.PackageModel import Detection
from sdks.novavision.src.base.logger import LoggerManager


class GoogleLandmarkDetection:
    def __init__(self,obj,image: np.ndarray,uID: str,):
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
        return buffer.tobytes()

    def detect_landmarks_as_detections(self):
        """
        Detect landmarks in the image and append them as Detection objects.
        Applies min_confidence threshold.
        BoundingBox is None as landmark detection returns no bounding box.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            image = vision.Image(content=image_bytes)

            response = self.vision_client.landmark_detection(image=image)

            if response.error.message:
                raise Exception(
                    f"Vision API Error: {response.error.message}\n"
                    "For more info: https://cloud.google.com/apis/design/errors"
                )

            landmarks = response.landmark_annotations
            print(f"[INFO] Number of landmarks detected: {len(landmarks)}")

            if not landmarks:
                print("[INFO] No landmarks detected in the image.")
                return

            for landmark in landmarks:
                score = getattr(landmark, "score", 0.0)
                print(
                    f"[DEBUG] Landmark: '{landmark.description}', score: {score}, threshold: {self.obj.min_confidence}")

                if score < self.obj.min_confidence:
                    print(f"[DEBUG] Skipping '{landmark.description}' due to low confidence.")
                    continue

                locations = [
                    {"latitude": loc.lat_lng.latitude, "longitude": loc.lat_lng.longitude}
                    for loc in landmark.locations
                ]

                first_lat = landmark.locations[0].lat_lng.latitude if landmark.locations else None
                first_lng = landmark.locations[0].lat_lng.longitude if landmark.locations else None

                detection = Detection(
                    boundingBox=None,  # Landmark verisi bounding box sağlamaz

                    # Vision API'nin landmark için verdiği skor (0.0 - 1.0)
                    confidence=score,

                    # Landmark'ın adı / açıklaması (örneğin "Eiffel Tower", "Mount Fuji")
                    classLabel=landmark.description,

                    # classId kullanılmadığı için -1 atanıyor
                    classId=-1,

                    # Görüntünün benzersiz ID'si
                    imgUID=self.uID,

                    # Landmark'a ait tüm konum bilgileri (lat-lng çiftleri)
                    locations=locations,

                    # Tespit tipi – burada sabit olarak "landmark"
                    segmentType="landmark",

                    # İlk konumun enlemi (varsa)
                    locations_latitude=first_lat,

                    # İlk konumun boylamı (varsa)
                    locations_longitude=first_lng
                )

                print(f"[INFO] Adding detection: {detection.classLabel} with confidence {detection.confidence}")
                self.obj.detections.append(detection)

            print(f"[INFO] Total detections added: {len(self.obj.detections)}")

        except Exception as e:
            self.logger.info(f"Error during landmark detection: {str(e)}")
            print(f"[ERROR] Exception during landmark detection: {str(e)}")
            raise

