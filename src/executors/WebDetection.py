
import os
import sys
from google.cloud import vision

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.GcpVision.src.utils.utils import API_AUTH
from sdks.novavision.src.helper.executor import Executor
from capsules.GcpVision.src.utils.response import build_response_web_detection
from capsules.GcpVision.src.models.PackageModel import PackageModel
from capsules.GcpVision.src.classes.GoogleWebDetection import GoogleWebDetection


class WebDetction(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

        self.detections =[]

    @staticmethod
    def bootstrap(config: dict) -> dict:
        API_AUTH(config)
        return {}

    def run(self):
        image = Image.get_frame(img=self.image, redis_db=self.redis_db)

        web_detector = GoogleWebDetection(self, image.value)
        self.web_detection_result =web_detector.detect_web()

        print("web_detection_result:", self.web_detection_result)
        package_model = build_response_web_detection(context=self)
        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()