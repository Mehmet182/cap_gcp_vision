
from sdks.novavision.src.helper.package import PackageHelper
from capsules.GcpVision.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, OutputData,OutputDetections
from capsules.GcpVision.src.models.PackageModel import ObjectDetectionOutputs, ObjectDetectionResponse,ObjectDetectionExecutor
from capsules.GcpVision.src.models.PackageModel import SafeSearchOutputs, SafeSearchResponse,SafeSearchExecutor,OutputSafeSearch
from capsules.GcpVision.src.models.PackageModel import WebDetectionOutputs, WebDetectionResponse,WebDetectionExecutor,OutputWebSearch
from capsules.GcpVision.src.models.PackageModel import TextDetectFileOutputs, TextDetectFileResponse,TextDetectFileExecutor


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
