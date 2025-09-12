
from sdks.novavision.src.helper.package import PackageHelper
from capsules.GcpVision.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor,OutputData,OutputDetections
from capsules.GcpVision.src.models.PackageModel import ImagePropertiesOutputs, ImagePropertiesResponse, ImagePropertiesExecutor,OutputColors
from capsules.GcpVision.src.models.PackageModel import LabelDetectionOutputs, LabelDetectionResponse, LabelDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import DetectionLandmarksOutputs, DetectionLandmarksResponse, DetectionLandmarksExecutor
from capsules.GcpVision.src.models.PackageModel import LogoDetectionOutputs, LogoDetectionResponse, LogoDetectionExecutor

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


