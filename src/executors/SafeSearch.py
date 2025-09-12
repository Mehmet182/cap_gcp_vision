
import os
import sys
from google.cloud import vision

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.GcpVision.src.utils.utils import API_AUTH
from sdks.novavision.src.helper.executor import Executor
from capsules.GcpVision.src.utils.response import build_response_safe_search
from capsules.GcpVision.src.models.PackageModel import PackageModel
from capsules.GcpVision.src.classes.GoogleSafeSearch import GoogleSafeSearch


class SafeSearch(Capsule):
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

        self.safe_search_detector = GoogleSafeSearch(self, image.value, image.uID)
        self.safe_search_result = self.safe_search_detector.detect_safe_search()

        print("safe_search_result:", self.safe_search_result)
        package_model = build_response_safe_search(context=self)
        return package_model



if __name__ == "__main__":
    Executor(sys.argv[1]).run()