
import os
import sys
from google.cloud import vision

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
from capsules.GcpVision.src.utils.utils import API_AUTH
from sdks.novavision.src.helper.executor import Executor
from capsules.GcpVision.src.utils.response import build_response_text_detection_file
from capsules.GcpVision.src.models.PackageModel import PackageModel
from capsules.GcpVision.src.classes.GoogleTextDetectionFile import GoogleTextDetectionFile


class TextDetectionFile(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.sourceUrl = self.request.get_param("SourceUrl")
        self.destinationUrl=self.request.get_param("DestinationUrl")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        API_AUTH(config)
        return {}

    def run(self):

        self.GoogleTextDetectionFile.process_document(self)
        package_model = build_response_text_detection_file(context=self)
        return package_model

if __name__ == "__main__":
    Executor(sys.argv[1]).run()