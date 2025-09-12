
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Detection, KeyPoints
from pydantic import BaseModel

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


class RGBColor(BaseModel):
    red: float
    green: float
    blue: float
    alpha: Optional[float] = 1.0  # Eğer alpha eksikse varsayılan 1.0 olsun


class DominantColor(BaseModel):
    color: RGBColor
    score: float
    pixelFraction: float

class KeyPoints(KeyPoints):
    confidence: Optional[float] = None


class Detection(Detection):
    keyPoints: Optional[List[KeyPoints]] = None
    imgUID: Optional[str] = None
    segmentType: Optional[str] = None

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"

class OutputColors(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[DominantColor]
    type: Literal["list"] = "list"

    class Config:
        title = "Colors"


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


class ImagePropertiesInputs(Inputs):
    inputImage: InputImage

class ImagePropertiesConfigs(Configs):
    tokenSelection: TokenSelection

class ImagePropertiesOutputs(Outputs):
    outputColors: OutputColors

class ImagePropertiesRequest(Request):
    inputs: Optional[ImagePropertiesInputs]
    configs: ImagePropertiesConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class ImagePropertiesResponse(Response):
    outputs: ImagePropertiesOutputs

class ImagePropertiesExecutor(Config):
    name: Literal["ImageProperties"] = "ImageProperties"
    value: Union[ImagePropertiesRequest, ImagePropertiesResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Image Properties"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ImagePropertiesExecutor,LabelDetectionExecutor,DetectionLandmarksExecutor,LogoDetectionExecutor]
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
