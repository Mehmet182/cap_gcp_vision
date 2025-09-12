
from sdks.novavision.src.helper.package import PackageHelper


from capsules.GcpVision.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, OutputData,OutputDetections
from capsules.GcpVision.src.models.PackageModel import TextDetectionOutputs, TextDetectionResponse, TextDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import CropHintsOutputs, CropHintsResponse, CropHintsExecutor
from capsules.GcpVision.src.models.PackageModel import FaceDetectionOutputs, FaceDetectionResponse, FaceDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import ImagePropertiesOutputs, ImagePropertiesResponse, ImagePropertiesExecutor,OutputColors
from capsules.GcpVision.src.models.PackageModel import DetectionLandmarksOutputs, DetectionLandmarksResponse, DetectionLandmarksExecutor
from capsules.GcpVision.src.models.PackageModel import LogoDetectionOutputs, LogoDetectionResponse, LogoDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import LabelDetectionOutputs, LabelDetectionResponse, LabelDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import ObjectDetectionOutputs, ObjectDetectionResponse,ObjectDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import SafeSearchOutputs, SafeSearchResponse,SafeSearchExecutor,OutputSafeSearch
from capsules.GcpVision.src.models.PackageModel import WebDetectionOutputs, WebDetectionResponse,WebDetectionExecutor,OutputWebSearch
from capsules.GcpVision.src.models.PackageModel import TextDetectFileOutputs, TextDetectFileResponse,TextDetectFileExecutor

def build_response_text_detection(context):
    outputData = OutputData(value=context.text)
    textDetectionOutputs = TextDetectionOutputs(outputData=outputData)
    textDetectionResponse = TextDetectionResponse(outputs=textDetectionOutputs)
    textDetectionExecutor = TextDetectionExecutor(value=textDetectionResponse)
    executor = ConfigExecutor(value=textDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_crop_hints(context):
    outputDetections = OutputDetections(value=context.detections)
    cropHintsOutputs = CropHintsOutputs(outputDetections=outputDetections)
    cropHintsResponse = CropHintsResponse(outputs=cropHintsOutputs)
    cropHintsExecutor = CropHintsExecutor(value=cropHintsResponse)
    executor = ConfigExecutor(value=cropHintsExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_face_detection(context):
    outputDetections = OutputDetections(value=context.detections)
    faceDetectionOutputs = FaceDetectionOutputs(outputDetections=outputDetections)
    faceDetectionResponse = FaceDetectionResponse(outputs=faceDetectionOutputs)
    faceDetectionExecutor = FaceDetectionExecutor(value=faceDetectionResponse)
    executor = ConfigExecutor(value=faceDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_image_properties(context):
    outputDetections = OutputColors(value=context.dominant_colors)
    imagePropertiesOutputs = ImagePropertiesOutputs(outputDetections=outputDetections)
    imagePropertiesResponse = ImagePropertiesResponse(outputs=imagePropertiesOutputs)
    imagePropertiesExecutor = ImagePropertiesExecutor(value=imagePropertiesResponse)
    executor = ConfigExecutor(value=imagePropertiesExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_label_detection(context):
    outputDetections = OutputDetections(value=context.detections)
    labelDetectionOutputs = LabelDetectionOutputs(outputDetections=outputDetections)
    labelDetectionResponse = LabelDetectionResponse(outputs=labelDetectionOutputs)
    labelDetectionExecutor = LabelDetectionExecutor(value=labelDetectionResponse)
    executor = ConfigExecutor(value=labelDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_detection_landmarks(context):
    outputDetections = OutputDetections(value=context.detections)
    detectionLandmarksOutputs = DetectionLandmarksOutputs(outputDetections=outputDetections)
    detectionLandmarksResponse = DetectionLandmarksResponse(outputs=detectionLandmarksOutputs)
    detectionLandmarksExecutor = DetectionLandmarksExecutor(value=detectionLandmarksResponse)
    executor = ConfigExecutor(value=detectionLandmarksExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_logo_detection(context):
    outputDetections = OutputDetections(value=context.detections)
    logoDetectionOutputs = LogoDetectionOutputs(outputDetections=outputDetections)
    logoDetectionResponse = LogoDetectionResponse(outputs=logoDetectionOutputs)
    logoDetectionExecutor = LogoDetectionExecutor(value=logoDetectionResponse)
    executor = ConfigExecutor(value=logoDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_object_detection(context):
    outputDetections = OutputDetections(value=context.detections)
    objectDetectionOutputs = ObjectDetectionOutputs(outputDetections=outputDetections)
    objectDetectionResponse = ObjectDetectionResponse(outputs=objectDetectionOutputs)
    objectDetectionExecutor = ObjectDetectionExecutor(value=objectDetectionResponse)
    executor = ConfigExecutor(value=objectDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_safe_search(context):
    outputDetections = OutputSafeSearch(value=context.safe_search_result)
    safeSearchOutputs = SafeSearchOutputs(outputDetections=outputDetections)
    safeSearchResponse = SafeSearchResponse(outputs=safeSearchOutputs)
    safeSearchExecutor = SafeSearchExecutor(value=safeSearchResponse)
    executor = ConfigExecutor(value=safeSearchExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_web_detection(context):
    outputDetections = OutputWebSearch(value=context.web_detection_result)
    webDetectionOutputs = WebDetectionOutputs(outputDetections=outputDetections)
    webDetectionResponse = WebDetectionResponse(outputs=webDetectionOutputs)
    webDetectionExecutor = WebDetectionExecutor(value=webDetectionResponse)
    executor = ConfigExecutor(value=webDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_text_detection_file(context):
    outputDetections = OutputData(value=context.destinationUrl)
    textDetectFileOutputs = TextDetectFileOutputs(outputDetections=outputDetections)
    textDetectFileResponse = TextDetectFileResponse(outputs=textDetectFileOutputs)
    textDetectFileExecutor = TextDetectFileExecutor(value=textDetectFileResponse)
    executor = ConfigExecutor(value=textDetectFileExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
