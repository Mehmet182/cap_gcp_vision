import cv2
import numpy as np
from google.cloud import vision
from typing import List
from sdks.novavision.src.base.logger import LoggerManager
from capsules.GcpVision.src.models.PackageModel import Detection


class GoogleLabelDetection:
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
            print("Vision API client initialized (Label Detection)")
        return self._vision_client

    def _convert_image_to_bytes(self, image_array: np.ndarray) -> bytes:
        """Convert OpenCV image array to bytes."""
        success, buffer = cv2.imencode('.jpg', image_array)
        if not success:
            raise ValueError("Failed to encode image to bytes")
        print("Converted image to bytes")
        return buffer.tobytes()

    def detect_labels_as_detections(self):
        """
        Detect labels in the image and append them as Detection objects.
        Filters out labels below score threshold from config (default 0.5).
        Since label detection has no bounding box, `boundingBox` will be None.
        """
        try:
            image_bytes = self._convert_image_to_bytes(self.image)
            print(f"Encoded image bytes length: {len(image_bytes)}")
            image = vision.Image(content=image_bytes)

            response = self.vision_client.label_detection(image=image)
            print("Full API Response:")
            print(response)

            if response.error.message:
                raise Exception(
                    f"Vision API Error: {response.error.message}\n"
                    "For more info: https://cloud.google.com/apis/design/errors"
                )

            labels = response.label_annotations
            if not labels:
                print("No labels detected in the image.")
            else:
                print(f"{len(labels)} labels detected:")
                for i, label in enumerate(labels):
                    print(f"Label {i + 1}:")
                    print(f"  Description: {label.description}")
                    print(f"  Score: {label.score}")

                    if hasattr(label, "locations") and label.locations:
                        print(f"  Locations:")
                        for loc in label.locations:
                            print(f"    Latitude: {loc.lat_lng.latitude}, Longitude: {loc.lat_lng.longitude}")
                    else:
                        print("  No location info available for this label.")

            for idx, label in enumerate(labels):
                print("self.obj.threshold:",self.obj.threshold)
                if label.score < self.obj.threshold:
                    print(f"Skipping label '{label.description}' due to low confidence: {label.score}")
                    continue

                detection = Detection(
                    boundingBox=None,  # Label detection bounding box dönmez

                    # Bu label için modelin güven skoru (0.0 - 1.0)
                    confidence=label.score,

                    # Label’ın metinsel açıklaması (örneğin: "Mountain", "Dog" vs.)
                    classLabel=label.description,

                    # Sınıf ID’si kullanılmıyor, -1 atanıyor.
                    classId=-1,

                    # Görüntüye ait benzersiz ID (takip için)
                    imgUID=self.uID,

                    # Ekstra alan: Etiketin görüntüyle ne kadar alakalı olduğunu belirten değer (0.0 - 1.0)
                    # Bazı durumlarda olmayabileceği için getattr ile None fallback kullanılır.
                    topicality=getattr(label, "topicality", None)
                )

                self.obj.detections.append(detection)

            print(f"{len(self.obj.detections)} label(s) above threshold converted to detections.")


        except Exception as e:
            self.logger.info(f"Error during label detection: {str(e)}")
            print(f"Exception during label detection: {str(e)}")  # Konsolda da görünmesi için
            raise


