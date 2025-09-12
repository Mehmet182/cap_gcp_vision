import cv2
import numpy as np
from google.cloud import vision
from typing import List

from sdks.novavision.src.base.model import BoundingBox
from capsules.GcpVision.src.models.PackageModel import Detection
from sdks.novavision.src.base.logger import LoggerManager


class GoogleFaceDetection:
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

    def _detect_faces(self, image_bytes: bytes) -> List[vision.FaceAnnotation]:
        """Call Vision API to detect faces."""
        image = vision.Image(content=image_bytes)
        response = self.vision_client.face_detection(image=image)

        if response.error.message:
            raise Exception(
                f"Vision API Error: {response.error.message}\n"
                "For more info: https://cloud.google.com/apis/design/errors"
            )

        print(f"Detected {len(response.face_annotations)} faces.")
        return response.face_annotations

    def detect_faces_as_detections(self):
        """Detect faces using Google Cloud Vision API and convert results to Detection objects."""

        try:
            # OpenCV görüntüsünü byte dizisine çevir
            image_bytes = self._convert_image_to_bytes(self.image)

            # Vision API ile yüz tespiti
            faces = self._detect_faces(image_bytes)

            likelihood_name = [
                "UNKNOWN",
                "VERY_UNLIKELY",
                "UNLIKELY",
                "POSSIBLE",
                "LIKELY",
                "VERY_LIKELY"
            ]

            for face in faces:
                # Ana bounding box (yüzü içeren kutu)
                vertices = face.bounding_poly.vertices
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

                # Keypoints (göz, burun, ağız vb.) + z ekseni ve landmark tipi
                keypoints = []
                for landmark in face.landmarks:
                    keypoints.append({
                        "cx": landmark.position.x,
                        "cy": landmark.position.y,
                        "cz": getattr(landmark.position, "z", 0.0),
                        "type": landmark.type_.name  # LANDMARK tipi örn: NOSE_TIP, LEFT_EYE
                    })

                # Etiket: Duygu tahmini
                label = (
                    f"joy:{likelihood_name[face.joy_likelihood]},"
                    f"anger:{likelihood_name[face.anger_likelihood]},"
                    f"surprise:{likelihood_name[face.surprise_likelihood]}"
                )

                # Rotasyona göre normalize edilmiş daha iyi bir kutu (isteğe bağlı)
                if face.fd_bounding_poly and face.fd_bounding_poly.vertices:
                    fd_vertices = face.fd_bounding_poly.vertices
                    x_fd = [v.x for v in fd_vertices]
                    y_fd = [v.y for v in fd_vertices]

                    x_fd_min, x_fd_max = min(x_fd), max(x_fd)
                    y_fd_min, y_fd_max = min(y_fd), max(y_fd)
                    fd_width = x_fd_max - x_fd_min
                    fd_height = y_fd_max - y_fd_min

                    bbox_fd = BoundingBox(
                        left=int(x_fd_min),
                        top=int(y_fd_min),
                        width=int(fd_width),
                        height=int(fd_height)
                    )

                    # fdBoundingPoly verisini özel alan olarak ekle
                    #detection.__dict__["boundingBoxFd"] = bbox_fd

                # Detection objesi oluşturuluyor
                detection = Detection(
                    boundingBox=bbox,
                    confidence=face.detection_confidence,  # Gerçek confidence değeri
                    classId=-1,
                    classLabel=label,
                    imgUID=self.uID,
                    keyPoints=keypoints,  # Landmark listesi
                    segmentType=None , # Kullanılmıyor ama API yapısı bunu bekliyor
                    sorrowLikelihood= likelihood_name[face.sorrow_likelihood],
                    headwearLikelihood= likelihood_name[face.headwear_likelihood],
                    blurredLikelihood = likelihood_name[face.blurred_likelihood],
                    underExposedLikelihood = likelihood_name[face.under_exposed_likelihood],
                    rollAngle = face.roll_angle,
                    panAngle = face.pan_angle,
                    tiltAngle = face.tilt_angle,
                    landmarkingConfidence=face.landmarking_confidence,
                    boundingBoxFd = bbox_fd
                )
                # Listeye ekle
                self.obj.detections.append(detection)

        except Exception as e:
            # Hata durumunda logla ve hatayı yükselt
            self.logger.info(f"Error during face detection: {str(e)}")
            raise


