
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Detection, KeyPoints
from pydantic import BaseModel ,HttpUrl

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class WebImage(BaseModel):
    url: HttpUrl

class WebPage(BaseModel):
    url: HttpUrl
    full_matching_images: Optional[List[WebImage]] = []
    partial_matching_images: Optional[List[WebImage]] = []

class WebEntity(BaseModel):
    score: float
    description: Optional[str] = None

class BestGuessLabel(BaseModel):
    label: str

class WebDetection(BaseModel):
    best_guess_labels: Optional[List[BestGuessLabel]] = []
    pages_with_matching_images: Optional[List[WebPage]] = []
    web_entities: Optional[List[WebEntity]] = []
    visually_similar_images: Optional[List[WebImage]] = []


class SafeSearchResult(BaseModel):
    adult: str
    medical: str
    spoofed: str
    violence: str
    racy: str

class KeyPoints(KeyPoints):
    confidence: Optional[float] = None


class Detection(Detection):
    keyPoints: Optional[List[KeyPoints]] = None
    imgUID: Optional[str] = None
    segmentType: Optional[str] = None

class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value:  List[str] 
    type: Literal["string"] = "string"

    class Config:
        title = "Data"

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class OutputSafeSearch(Output):
    name: Literal["outputSafeSearch"] = "outputSafeSearch"
    value: List[SafeSearchResult]
    type: Literal["list"] = "list"

    class Config:
        title = "Safe Search"

class OutputWebSearch(Output):
    name: Literal["outputWebSearch"] = "outputWebSearch"
    value: List[WebDetection]
    type: Literal["list"] = "list"

    class Config:
        title = "Web Search "

class StorageSource(Config):
    name: Literal["storageSource"] = "storageSource"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"

    class Config:
        json_schema_extra = {
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": [
                    "name"
                ],
                "name": "app::Object_wide"
            }
        }
        title = "Storage Source"

class SourceUrl(Config):
    """
    PDF URL
    """
    name: Literal["sourceUrl"] = "sourceUrl"
    value: str
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "File Source Url"


class DestinationUrl(Config):
    """
    Destination URL
    """
    name: Literal["destinationUrl"] = "destinationUrl"
    value: str
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Destination Url"

class StoragePath(Config):
    name: Literal["storagePath"] = "storagePath"
    value: str
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "File Name"


class ConfigStorage(Config):
    name: Literal["ConfigStorage"] = "ConfigStorage"
    storageSource: StorageSource
    value: Literal["ConfigStorage"] = "ConfigStorage"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Storage Source"


class ConfigPath(Config):
    name: Literal["ConfigPath"] = "ConfigPath"
    storagePath: StoragePath
    value: Literal["ConfigPath"] = "ConfigPath"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Local Path"


class TokenSelection(Config):
    """
        Guide: https://docs.google.com/document/d/1JVK_cOYd0MJDi2mqfRRGXOW-09pIYx2pXHueO-E8qmY/edit?usp=sharing
    """
    name: Literal["tokenSelection"] = "tokenSelection"
    value: Union[ConfigPath, ConfigStorage]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Token Source Selection"

class MinConfidence(Config):
    """
       Sets how sure the model must be about a prediction.
    """
    name: Literal["minConfidence"] = "minConfidence"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Confidence"

class Threshold(Config):
    """
       Sets how sure the model must be about a prediction.
    """
    name: Literal["threshold"] = "threshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Threshold"


class CropAspectRatios(Config):
    """
    The aspect ratio defines the width-to-height ratio of the image.
    """
    name: Literal["CropAspectRatios"] = "CropAspectRatios"
    value: float =(Field(default=1.77))
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Aspect Ratios"


class WebDetectionInputs(Inputs):
    inputImage: InputImage

class WebDetectionConfigs(Configs):
    tokenSelection: TokenSelection


class WebDetectionOutputs(Outputs):
    outputWebSearch: OutputWebSearch

class WebDetectionRequest(Request):
    inputs: Optional[WebDetectionInputs]
    configs: WebDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class WebDetectionResponse(Response):
    outputs: WebDetectionOutputs

class WebDetectionExecutor(Config):
    name: Literal["WebDetection"] = "WebDetection"
    value: Union[WebDetectionRequest, WebDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Web Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class ObjectDetectionInputs(Inputs):
    inputImage: InputImage

class ObjectDetectionConfigs(Configs):
    tokenSelection: TokenSelection

class ObjectDetectionOutputs(Outputs):
    outputDetections: OutputDetections

class ObjectDetectionRequest(Request):
    inputs: Optional[ObjectDetectionInputs]
    configs: ObjectDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class ObjectDetectionResponse(Response):
    outputs: ObjectDetectionOutputs

class ObjectDetectionExecutor(Config):
    name: Literal["ObjectDetection"] = "ObjectDetection"
    value: Union[ObjectDetectionRequest, ObjectDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Object Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class SafeSearchInputs(Inputs):
    inputImage: InputImage

class SafeSearchConfigs(Configs):
    tokenSelection: TokenSelection

class SafeSearchOutputs(Outputs):
    outputSafeSearch: OutputSafeSearch

class SafeSearchRequest(Request):
    inputs: Optional[SafeSearchInputs]
    configs: SafeSearchConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class SafeSearchResponse(Response):
    outputs: SafeSearchOutputs

class SafeSearchExecutor(Config):
    name: Literal["SafeSearch"] = "SafeSearch"
    value: Union[SafeSearchRequest, SafeSearchResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Safe Search"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class TextDetectFileConfigs(Configs):
    tokenSelection: TokenSelection
    sourceUrl:SourceUrl
    destinationUrl:DestinationUrl

class TextDetectFileOutputs(Outputs):
    outputData: OutputData

class TextDetectFileRequest(Request):
    configs: TextDetectFileConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class TextDetectFileResponse(Response):
    outputs: TextDetectFileOutputs

class TextDetectFileExecutor(Config):
    name: Literal["TextDetectFile"] = "TextDetectFile"
    value: Union[TextDetectFileRequest, TextDetectFileResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Detection File"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }
class LogoDetectionInputs(Inputs):
    inputImage: InputImage

class LogoDetectionConfigs(Configs):
    tokenSelection: TokenSelection


class LogoDetectionOutputs(Outputs):
    outputDetections: OutputDetections

class LogoDetectionRequest(Request):
    inputs: Optional[LogoDetectionInputs]
    configs: LogoDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class LogoDetectionResponse(Response):
    outputs: LogoDetectionOutputs

class LogoDetectionExecutor(Config):
    name: Literal["LogoDetection"] = "LogoDetection"
    value: Union[LogoDetectionRequest, LogoDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Logo Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class DetectionLandmarksInputs(Inputs):
    inputImage: InputImage

class DetectionLandmarksConfigs(Configs):
    tokenSelection: TokenSelection
    minConfidence:MinConfidence

class DetectionLandmarksOutputs(Outputs):
    outputDetections: OutputDetections

class DetectionLandmarksRequest(Request):
    inputs: Optional[DetectionLandmarksInputs]
    configs: DetectionLandmarksConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }



class DetectionLandmarksResponse(Response):
    outputs: DetectionLandmarksOutputs

class DetectionLandmarksExecutor(Config):
    name: Literal["DetectionLandmarks"] = "DetectionLandmarks"
    value: Union[DetectionLandmarksRequest, DetectionLandmarksResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Detection Landmarks"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class LabelDetectionInputs(Inputs):
    inputImage: InputImage

class LabelDetectionConfigs(Configs):
    tokenSelection: TokenSelection
    Threshold:Threshold

class LabelDetectionOutputs(Outputs):
    outputDetections: OutputDetections

class LabelDetectionRequest(Request):
    inputs: Optional[LabelDetectionInputs]
    configs: LabelDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class LabelDetectionResponse(Response):
    outputs: LabelDetectionOutputs

class LabelDetectionExecutor(Config):
    name: Literal["LabelDetection"] = "LabelDetection"
    value: Union[LabelDetectionRequest, LabelDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "label Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class FaceDetectionInputs(Inputs):
    inputImage: InputImage

class FaceDetectionConfigs(Configs):
    tokenSelection: TokenSelection

class FaceDetectionOutputs(Outputs):
    outputDetections: OutputDetections

class FaceDetectionRequest(Request):
    inputs: Optional[FaceDetectionInputs]
    configs: FaceDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class FaceDetectionResponse(Response):
    outputs: FaceDetectionOutputs

class FaceDetectionExecutor(Config):
    name: Literal["FaceDetection"] = "FaceDetection"
    value: Union[FaceDetectionRequest, FaceDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Face Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class CropHintsInputs(Inputs):
    inputImage: InputImage

class CropHintsConfigs(Configs):
    tokenSelection: TokenSelection
    cropAspectRatios:CropAspectRatios

class CropHintsOutputs(Outputs):
    outputDetections: OutputDetections

class CropHintsRequest(Request):
    inputs: Optional[CropHintsInputs]
    configs: CropHintsConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class CropHintsResponse(Response):
    outputs: CropHintsOutputs

class CropHintsExecutor(Config):
    name: Literal["CropHints"] = "CropHints"
    value: Union[CropHintsRequest, CropHintsResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Crop Hints"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class TextDetectionInputs(Inputs):
    inputImage: InputImage

class TextDetectionConfigs(Configs):
    tokenSelection: TokenSelection

class TextDetectionOutputs(Outputs):
    outputData: OutputData

class TextDetectionRequest(Request):
    inputs: Optional[TextDetectionInputs]
    configs: TextDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class TextDetectionResponse(Response):
    outputs: TextDetectionOutputs

class TextDetectionExecutor(Config):
    name: Literal["TextDetection"] = "TextDetection"
    value: Union[TextDetectionRequest, TextDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[TextDetectionExecutor,CropHintsExecutor,FaceDetectionExecutor,LabelDetectionExecutor,DetectionLandmarksExecutor,LogoDetectionExecutor,WebDetectionExecutor,ObjectDetectionExecutor,SafeSearchExecutor,TextDetectFileExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Type"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["GcpVision"] = "GcpVision"
