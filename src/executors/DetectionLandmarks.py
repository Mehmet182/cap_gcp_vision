
import os
import sys
from google.cloud import vision

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.GcpVision.src.utils.utils import API_AUTH
from sdks.novavision.src.helper.executor import Executor
from capsules.GcpVision.src.utils.response import build_response_detection_landmarks
from capsules.GcpVision.src.models.PackageModel import PackageModel
from capsules.GcpVision.src.classes.GoogleLandmarkDetection import GoogleLandmarkDetection


class DetectionLandmarks(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.min_confidence=self.request.get_param("minConfidence")

        self.detections =[]

    @staticmethod
    def bootstrap(config: dict) -> dict:
        API_AUTH(config)
        return {}

    def run(self):
        image = Image.get_frame(img=self.image, redis_db=self.redis_db)

        self.detector = GoogleLandmarkDetection(self,image.value, image.uID)
        self.detector.detect_landmarks_as_detections()
        print("Detections:", self.detections)
        package_model = build_response_detection_landmarks(context=self)
        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()