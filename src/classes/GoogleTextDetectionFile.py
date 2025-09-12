import re
import json
from google.cloud import vision
from google.cloud import storage
from sdks.novavision.src.base.logger import LoggerManager


class GoogleTextDetectionFile:
    def __init__(self, obj,):
        self.obj = obj
        self.logger = LoggerManager()
        self._vision_client = None
        self._storage_client = None

    @property
    def vision_client(self) -> vision.ImageAnnotatorClient:
        if self._vision_client is None:
            self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    @property
    def storage_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    def process_document(self, mime_type="application/pdf", batch_size=2, timeout=600):
        try:
            feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            gcs_source = vision.GcsSource(uri=self.obj.sourceUrl)
            input_config = vision.InputConfig(gcs_source=gcs_source, mime_type=mime_type)
            gcs_destination = vision.GcsDestination(uri=self.obj.destinationUrl)
            output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=batch_size)

            async_request = vision.AsyncAnnotateFileRequest(
                features=[feature],
                input_config=input_config,
                output_config=output_config
            )

            operation = self.vision_client.async_batch_annotate_files(requests=[async_request])
            operation.result(timeout=timeout)

            self.logger.info("OCR operation completed.")
            # **Hiçbir şey döndürmüyoruz**
            return None

        except Exception as e:
            self.logger.info(f"OCR processing failed: {str(e)}")
            raise