
from sdks.novavision.src.helper.package import PackageHelper
from capsules.GcpVision.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor
from capsules.GcpVision.src.models.PackageModel import ImagePropertiesOutputs, ImagePropertiesResponse, ImagePropertiesExecutor,OutputColors

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

