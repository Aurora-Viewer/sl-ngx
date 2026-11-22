// Prints the size and field offsets of every Streamline struct the viewer
// passes, so the Rust bindings' layout tests can hardcode them.
#include "sl/sl.h"
#include "sl/sl_dlss.h"
#include "sl/sl_dlss_d.h"
#include <vulkan/vulkan.h>
#include "sl/sl_helpers_vk.h"
#include <cstddef>
#include <cstdio>

#define S(t) std::printf("%-24s size %zu align %zu\n", #t, sizeof(sl::t), alignof(sl::t))
#define F(t, f) std::printf("  %-36s %zu\n", #f, offsetof(sl::t, f))

// a struct's type id (its GUID's 16 bytes as laid out) and the version a
// default-constructed one carries
template <typename T> void id(const char* name)
{
    T v{};
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v.structType);
    std::printf("%-24s id", name);
    for (int i = 0; i < 16; i++)
        std::printf(" %02x", b[i]);
    std::printf(" version %zu\n", (size_t)v.structVersion);
}
#define ID(t) id<sl::t>(#t)

#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
int main()
{
    S(BaseStructure); F(BaseStructure, next); F(BaseStructure, structType); F(BaseStructure, structVersion);
    S(Resource); F(Resource, type); F(Resource, native); F(Resource, memory); F(Resource, view); F(Resource, state);
    F(Resource, width); F(Resource, height); F(Resource, nativeFormat); F(Resource, mipLevels); F(Resource, arrayLayers);
    F(Resource, gpuVirtualAddress); F(Resource, flags); F(Resource, usage); F(Resource, internalFlags);
    S(ResourceTag); F(ResourceTag, resource); F(ResourceTag, type); F(ResourceTag, lifecycle); F(ResourceTag, extent);
    S(Extent);
    S(Constants); F(Constants, cameraViewToClip); F(Constants, clipToCameraView); F(Constants, clipToLensClip);
    F(Constants, clipToPrevClip); F(Constants, prevClipToClip); F(Constants, jitterOffset); F(Constants, mvecScale);
    F(Constants, cameraPinholeOffset); F(Constants, cameraPos); F(Constants, cameraUp); F(Constants, cameraRight);
    F(Constants, cameraFwd); F(Constants, cameraNear); F(Constants, cameraFar); F(Constants, cameraFOV);
    F(Constants, cameraAspectRatio); F(Constants, motionVectorsInvalidValue); F(Constants, depthInverted);
    F(Constants, cameraMotionIncluded); F(Constants, motionVectors3D); F(Constants, reset);
    F(Constants, orthographicProjection); F(Constants, motionVectorsDilated); F(Constants, motionVectorsJittered);
    F(Constants, minRelativeLinearDepthObjectSeparation);
    S(DLSSOptions); F(DLSSOptions, mode); F(DLSSOptions, outputWidth); F(DLSSOptions, outputHeight);
    F(DLSSOptions, sharpness); F(DLSSOptions, preExposure); F(DLSSOptions, exposureScale);
    F(DLSSOptions, colorBuffersHDR); F(DLSSOptions, indicatorInvertAxisX); F(DLSSOptions, indicatorInvertAxisY);
    F(DLSSOptions, dlaaPreset); F(DLSSOptions, qualityPreset); F(DLSSOptions, balancedPreset);
    F(DLSSOptions, performancePreset); F(DLSSOptions, ultraPerformancePreset); F(DLSSOptions, ultraQualityPreset);
    F(DLSSOptions, useAutoExposure); F(DLSSOptions, alphaUpscalingEnabled);
    S(DLSSOptimalSettings); F(DLSSOptimalSettings, optimalRenderWidth); F(DLSSOptimalSettings, optimalRenderHeight);
    F(DLSSOptimalSettings, optimalSharpness); F(DLSSOptimalSettings, renderWidthMin); F(DLSSOptimalSettings, renderHeightMin);
    F(DLSSOptimalSettings, renderWidthMax); F(DLSSOptimalSettings, renderHeightMax);
    S(DLSSState); F(DLSSState, estimatedVRAMUsageInBytes);
    S(DLSSDOptions); F(DLSSDOptions, mode); F(DLSSDOptions, outputWidth); F(DLSSDOptions, outputHeight);
    F(DLSSDOptions, sharpness); F(DLSSDOptions, preExposure); F(DLSSDOptions, exposureScale);
    F(DLSSDOptions, colorBuffersHDR); F(DLSSDOptions, indicatorInvertAxisX); F(DLSSDOptions, indicatorInvertAxisY);
    F(DLSSDOptions, normalRoughnessMode); F(DLSSDOptions, worldToCameraView); F(DLSSDOptions, cameraViewToWorld);
    F(DLSSDOptions, alphaUpscalingEnabled); F(DLSSDOptions, dlaaPreset); F(DLSSDOptions, qualityPreset);
    F(DLSSDOptions, balancedPreset); F(DLSSDOptions, performancePreset); F(DLSSDOptions, ultraPerformancePreset);
    F(DLSSDOptions, ultraQualityPreset);
    S(DLSSDOptimalSettings); S(DLSSDState);
    S(Preferences); F(Preferences, showConsole); F(Preferences, logLevel); F(Preferences, pathsToPlugins);
    F(Preferences, numPathsToPlugins); F(Preferences, pathToLogsAndData); F(Preferences, allocateCallback);
    F(Preferences, releaseCallback); F(Preferences, logMessageCallback); F(Preferences, flags);
    F(Preferences, featuresToLoad); F(Preferences, numFeaturesToLoad); F(Preferences, applicationId);
    F(Preferences, engine); F(Preferences, engineVersion); F(Preferences, projectId); F(Preferences, renderAPI);
    S(VulkanInfo); F(VulkanInfo, device); F(VulkanInfo, instance); F(VulkanInfo, physicalDevice);
    F(VulkanInfo, computeQueueIndex); F(VulkanInfo, computeQueueFamily); F(VulkanInfo, graphicsQueueIndex);
    F(VulkanInfo, graphicsQueueFamily); F(VulkanInfo, opticalFlowQueueIndex); F(VulkanInfo, opticalFlowQueueFamily);
    F(VulkanInfo, useNativeOpticalFlowMode); F(VulkanInfo, computeQueueCreateFlags);
    F(VulkanInfo, graphicsQueueCreateFlags); F(VulkanInfo, opticalFlowQueueCreateFlags);
    S(FeatureRequirements); F(FeatureRequirements, flags); F(FeatureRequirements, maxNumCPUThreads);
    F(FeatureRequirements, maxNumViewports); F(FeatureRequirements, numRequiredTags); F(FeatureRequirements, requiredTags);
    F(FeatureRequirements, osVersionDetected); F(FeatureRequirements, osVersionRequired);
    F(FeatureRequirements, driverVersionDetected); F(FeatureRequirements, driverVersionRequired);
    F(FeatureRequirements, vkNumComputeQueuesRequired); F(FeatureRequirements, vkNumGraphicsQueuesRequired);
    F(FeatureRequirements, vkNumDeviceExtensions); F(FeatureRequirements, vkDeviceExtensions);
    F(FeatureRequirements, vkNumInstanceExtensions); F(FeatureRequirements, vkInstanceExtensions);
    F(FeatureRequirements, vkNumFeatures12); F(FeatureRequirements, vkFeatures12);
    F(FeatureRequirements, vkNumFeatures13); F(FeatureRequirements, vkFeatures13);
    F(FeatureRequirements, vkNumOpticalFlowQueuesRequired);
    S(AdapterInfo); F(AdapterInfo, deviceLUID); F(AdapterInfo, deviceLUIDSizeInBytes); F(AdapterInfo, vkPhysicalDevice);
    S(ViewportHandle); S(FeatureVersion);
    std::printf("kSDKVersion %#llx\n", (unsigned long long)sl::kSDKVersion);
    ID(Preferences); ID(VulkanInfo); ID(FeatureRequirements); ID(Resource); ID(ResourceTag); ID(Constants);
    ID(DLSSOptions); ID(DLSSOptimalSettings); ID(DLSSState); ID(DLSSDOptions); ID(DLSSDOptimalSettings);
    ID(DLSSDState); ID(ViewportHandle); ID(AdapterInfo); ID(FeatureVersion);
    return 0;
}
