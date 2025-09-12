
import os
import sys
from google.cloud import vision

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.GcpVision.src.utils.utils import API_AUTH
from sdks.novavision.src.helper.executor import Executor
from capsules.GcpVision.src.utils.response import build_response_crop_hints
from capsules.GcpVision.src.models.PackageModel import PackageModel
from capsules.GcpVision.src.classes.GoogleCropHints import GoogleCropHints


class CropHints(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.aspect_ratios=self.request.get_param("CropAspectRatios")
        print(self.aspect_ratios)
        self.crop_hints = ""
        self.detections =[]

    @staticmethod
    def bootstrap(config: dict) -> dict:
        API_AUTH(config)
        return {}

    def run(self):
        image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.crop_hints = GoogleCropHints(self, image.value, image.uID)
        self.crop_hints.detect_crop_hints_as_detections()
        print("Detections:", self.detections)
        package_model = build_response_crop_hints(context=self)
        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()