
import os
import sys
from google.cloud import vision

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.GcpVision.src.utils.utils import API_AUTH
from sdks.novavision.src.helper.executor import Executor
from capsules.GcpVision.src.utils.response import build_response_object_detection
from capsules.GcpVision.src.models.PackageModel import PackageModel
from capsules.GcpVision.src.classes.GoogleObjectDetection import GoogleObjectDetection


class ObjectDetection(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

        self.detections =[]

    @staticmethod
    def bootstrap(config: dict) -> dict:
        API_AUTH(config)
        return {}

    def run_objects(self):
        # Görseli numpy array ve UID ile al
        image = Image.get_frame(img=self.image, redis_db=self.redis_db)

        # GoogleObjectLocalizer sınıfını oluştur
        self.object_localizer = GoogleObjectDetection(self, image.value, image.uID)

        # Obje lokalizasyonunu yap ve Detection objeleri olarak detections listesine ekle
        self.object_localizer.detect_objects_as_detections()

        # Tespit edilen sonuçları model formatına dönüştür (senin yapına göre fonksiyon)
        package_model = build_response_object_detection(context=self)

        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()