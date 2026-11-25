// SPDX-License-Identifier: MIT
//
// sl-ngx: the DLSS and DLSS Ray Reconstruction part of NVIDIA Streamline's
// public interface, implemented on NVIDIA NGX for Vulkan.
//
// The interface (include/sl/) is NVIDIA's own, MIT licensed. This file holds
// no NVIDIA code: NGX's declarations come from the NVIDIA DLSS SDK the user
// builds against (DLSS_SDK/include), and its core is linked from that SDK's
// static library. Which NGX parameters each Streamline field feeds follows
// Streamline's MIT DLSS and DLSS-D plugins (source/plugins/sl.dlss*), v2.14.16.

#include <vulkan/vulkan.h>

#include "sl/sl.h"
#include "sl/sl_dlss.h"
#include "sl/sl_dlss_d.h"
#include "sl/sl_helpers_vk.h"

#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#define SL_NGX_EXPORT __declspec(dllexport)
#else
#define SL_NGX_EXPORT __attribute__((visibility("default")))
#endif

namespace
{

using sl::Result;

// How many frame tokens after its last use a feature replaced inside
// slEvaluateFeature is kept before NGX releases it (Streamline keeps three
// frames in flight).
constexpr uint64_t kReleaseAfterFrames = 4;
constexpr uint32_t kTokenRing = 16;
constexpr uint32_t kFramesKept = 8;

struct Token final : sl::FrameToken
{
    uint32_t index = 0;
    operator uint32_t() const override { return index; }
};

struct Tag
{
    sl::Resource resource{sl::ResourceType::eTex2d, nullptr};
    sl::Extent extent{};
};

struct Frame
{
    bool hasConstants = false;
    sl::Constants constants{};
    std::map<sl::BufferType, Tag> tags;
};

// What a feature was made with; a change makes a new one.
struct Shape
{
    sl::Feature feature = 0;
    sl::DLSSMode mode = sl::DLSSMode::eOff;
    uint32_t outWidth = 0, outHeight = 0;
    uint32_t renderWidth = 0, renderHeight = 0;
    int flags = 0;
    uint32_t presets[6]{};
    uint32_t roughnessMode = 0;
    bool operator==(const Shape& o) const { return std::memcmp(this, &o, sizeof(Shape)) == 0; }
};

struct Viewport
{
    bool hasDlss = false, hasDlssd = false;
    sl::DLSSOptions dlss{};
    sl::DLSSDOptions dlssd{};
    NVSDK_NGX_Handle* handle = nullptr;
    Shape shape{};
    uint64_t lastUsed = 0;
    bool logNextEvaluate = false;
    std::map<uint32_t, Frame> frames;
    // frame generation: its own feature beside the upscaler's
    NVSDK_NGX_Handle* fgHandle = nullptr;
    uint32_t fgWidth = 0, fgHeight = 0, fgFormat = 0, fgRenderWidth = 0, fgRenderHeight = 0;
    unsigned long long fgFrameId = 0;
};

struct Retired
{
    NVSDK_NGX_Handle* handle;
    uint64_t after;
};

struct State
{
    std::mutex lock;
    bool initialized = false;
    sl::PFun_LogMessageCallback* log = nullptr;
    sl::LogLevel logLevel = sl::LogLevel::eDefault;
    std::string projectId, engineVersion;
    NVSDK_NGX_EngineType engine = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    std::wstring dataPath;
    std::vector<std::wstring> modelPaths;
    std::vector<const wchar_t*> modelPathPtrs;
    NVSDK_NGX_FeatureCommonInfo common{};

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    // the instance and physical device slGetFeatureRequirements asks NGX
    // about: the last VulkanInfo's, with a device or without
    VkInstance queryInstance = VK_NULL_HANDLE;
    VkPhysicalDevice queryPhysicalDevice = VK_NULL_HANDLE;
    bool started = false;
    NVSDK_NGX_Parameter* params = nullptr;

    Token tokens[kTokenRing];
    uint32_t nextToken = 0;
    uint32_t frameCounter = 0;
    uint64_t issued = 0;

    std::map<uint32_t, Viewport> viewports;
    std::vector<Retired> retired;

    // NGX's optimal settings by (feature, mode, output width, height)
    struct Optimal
    {
        uint32_t w, h, minW, minH, maxW, maxH;
        float sharpness;
    };
    std::map<std::tuple<sl::Feature, uint32_t, uint32_t, uint32_t>, Optimal> optimal;

    // slGetFeatureRequirements' answers live here until slShutdown
    std::map<sl::Feature, std::vector<std::string>> instanceExtNames, deviceExtNames;
    std::map<sl::Feature, std::vector<const char*>> instanceExts, deviceExts;
};

State g;

void logf(sl::LogType type, const char* fmt, ...)
{
    if (!g.log || g.logLevel == sl::LogLevel::eOff)
        return;
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    g.log(type, buf);
}
#define LOG_INFO(...) logf(sl::LogType::eInfo, __VA_ARGS__)
#define LOG_WARN(...) logf(sl::LogType::eWarn, __VA_ARGS__)
#define LOG_ERROR(...) logf(sl::LogType::eError, __VA_ARGS__)

bool sameType(const sl::StructType& a, const sl::StructType& b)
{
    return std::memcmp(&a, &b, sizeof(sl::StructType)) == 0;
}

NVSDK_NGX_Feature ngxFeature(sl::Feature f)
{
    if (f == sl::kFeatureDLSS_G)
        return NVSDK_NGX_Feature_FrameGeneration;
    return f == sl::kFeatureDLSS_RR ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling;
}

bool ours(sl::Feature f)
{
    return f == sl::kFeatureDLSS || f == sl::kFeatureDLSS_RR || f == sl::kFeatureDLSS_G;
}

// The folder holding this library: where the setup puts NVIDIA's model
// library beside it, and so NGX's search path when the caller names none.
std::wstring ownFolder()
{
#ifdef _WIN32
    HMODULE module = nullptr;
    wchar_t buf[MAX_PATH];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&ownFolder), &module) ||
        !GetModuleFileNameW(module, buf, MAX_PATH))
        return {};
    std::wstring path = buf;
    auto slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring(L".") : path.substr(0, slash);
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&ownFolder), &info) || !info.dli_fname)
        return {};
    std::string path = info.dli_fname;
    auto slash = path.rfind('/');
    path = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    return std::wstring(path.begin(), path.end());
#endif
}

NVSDK_NGX_FeatureDiscoveryInfo discovery(sl::Feature feature)
{
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = ngxFeature(feature);
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc.ProjectId = g.projectId.c_str();
    info.Identifier.v.ProjectDesc.EngineType = g.engine;
    info.Identifier.v.ProjectDesc.EngineVersion = g.engineVersion.c_str();
    info.ApplicationDataPath = g.dataPath.c_str();
    info.FeatureInfo = &g.common;
    return info;
}

void releaseNow(NVSDK_NGX_Handle* handle)
{
    if (handle)
    {
        NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_ReleaseFeature(handle);
        if (NVSDK_NGX_FAILED(r))
            LOG_WARN("sl-ngx: NGX did not release a feature: %#x", (unsigned)r);
    }
}

void releaseDue(bool all)
{
    std::vector<Retired> kept;
    for (auto& r : g.retired)
    {
        if (all || g.issued >= r.after)
            releaseNow(r.handle);
        else
            kept.push_back(r);
    }
    g.retired.swap(kept);
}

void stopNgx()
{
    if (!g.started)
        return;
    for (auto& [id, vp] : g.viewports)
    {
        releaseNow(vp.handle);
        vp.handle = nullptr;
        releaseNow(vp.fgHandle);
        vp.fgHandle = nullptr;
    }
    releaseDue(true);
    if (g.params)
        NVSDK_NGX_VULKAN_DestroyParameters(g.params);
    g.params = nullptr;
    NVSDK_NGX_VULKAN_Shutdown1(g.device);
    g.optimal.clear();
    g.started = false;
}

// Leaves NGX started on the device it was started on, with no NGX call at
// all: that device may be lost, and releasing or shutting down there can
// hang. NGX keeps one instance per device, so it starts again beside it on
// the next one. The caller keeps the old VkDevice alive (NGX still holds
// objects on it).
void abandonNgx()
{
    if (!g.started)
        return;
    size_t kept = g.retired.size();
    for (auto& [id, vp] : g.viewports)
        kept += (vp.handle ? 1 : 0) + (vp.fgHandle ? 1 : 0);
    g.viewports.clear();
    g.retired.clear();
    g.params = nullptr;
    g.optimal.clear();
    g.started = false;
    LOG_WARN("sl-ngx: NGX left started on the previous device (%zu features kept there)", kept);
}

Result startNgx()
{
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        g.projectId.c_str(), g.engine, g.engineVersion.c_str(), g.dataPath.c_str(), g.instance, g.physicalDevice,
        g.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &g.common, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r))
    {
        LOG_ERROR("sl-ngx: NGX did not start: %#x", (unsigned)r);
        return Result::eErrorNGXFailed;
    }
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g.params);
    if (NVSDK_NGX_FAILED(r) || !g.params)
    {
        LOG_ERROR("sl-ngx: no NGX capability parameters: %#x", (unsigned)r);
        NVSDK_NGX_VULKAN_Shutdown1(g.device);
        g.params = nullptr;
        return Result::eErrorNGXFailed;
    }
    g.started = true;
    LOG_INFO("sl-ngx: NGX started");
    return Result::eOk;
}

// Whether NGX offers the feature on the device it started on, and why not.
Result supported(sl::Feature feature)
{
    if (!g.started)
        return Result::eErrorDeviceNotCreated;
    int available = 0, needsDriver = 0, init = 0;
    if (feature == sl::kFeatureDLSS_G)
    {
        NVSDK_NGX_Parameter_GetI(g.params, NVSDK_NGX_Parameter_FrameGeneration_Available, &available);
        if (available)
            return Result::eOk;
        NVSDK_NGX_Parameter_GetI(g.params, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needsDriver);
        NVSDK_NGX_Parameter_GetI(g.params, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &init);
        LOG_WARN("sl-ngx: NGX offers no frame generation here (init result %#x, newer driver needed: %d)",
            (unsigned)init, needsDriver);
        return needsDriver ? Result::eErrorDriverOutOfDate : Result::eErrorFeatureNotSupported;
    }
    const bool rr = feature == sl::kFeatureDLSS_RR;
    NVSDK_NGX_Parameter_GetI(g.params,
        rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_Available : NVSDK_NGX_Parameter_SuperSampling_Available,
        &available);
    if (available)
        return Result::eOk;
    NVSDK_NGX_Parameter_GetI(g.params,
        rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver
           : NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
        &needsDriver);
    NVSDK_NGX_Parameter_GetI(g.params,
        rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult
           : NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
        &init);
    LOG_WARN("sl-ngx: NGX offers no %s here (init result %#x, newer driver needed: %d)",
        rr ? "Ray Reconstruction" : "DLSS", (unsigned)init, needsDriver);
    return needsDriver ? Result::eErrorDriverOutOfDate : Result::eErrorFeatureNotSupported;
}

Result optimal(sl::Feature feature, sl::DLSSMode mode, uint32_t outW, uint32_t outH, uint32_t* w, uint32_t* h,
    uint32_t* minW, uint32_t* minH, uint32_t* maxW, uint32_t* maxH, float* sharpness)
{
    if (!g.started)
        return Result::eErrorDeviceNotCreated;
    if (mode == sl::DLSSMode::eOff || mode >= sl::DLSSMode::eCount || outW == 0 || outH == 0 ||
        outW == sl::INVALID_UINT || outH == sl::INVALID_UINT)
        return Result::eErrorInvalidParameter;
    // asking NGX rewrites the shared parameters: once per shape is enough
    const auto key = std::make_tuple(feature, (uint32_t)mode, outW, outH);
    auto cached = g.optimal.find(key);
    if (cached != g.optimal.end())
    {
        const auto& o = cached->second;
        *w = o.w, *h = o.h, *minW = o.minW, *minH = o.minH, *maxW = o.maxW, *maxH = o.maxH;
        *sharpness = o.sharpness;
        return Result::eOk;
    }
    auto quality = (NVSDK_NGX_PerfQuality_Value)((uint32_t)mode - 1);
    NVSDK_NGX_Result r;
    if (feature == sl::kFeatureDLSS)
    {
        r = NGX_DLSS_GET_OPTIMAL_SETTINGS(g.params, outW, outH, quality, w, h, maxW, maxH, minW, minH, sharpness);
    }
    else
    {
        // DLSS-D answers through its own callback, read the same way
        void* callback = nullptr;
        NVSDK_NGX_Parameter_GetVoidPointer(g.params, NVSDK_NGX_Parameter_DLSSDOptimalSettingsCallback, &callback);
        if (!callback)
            return Result::eErrorFeatureNotSupported;
        NVSDK_NGX_Parameter_SetUI(g.params, NVSDK_NGX_Parameter_Width, outW);
        NVSDK_NGX_Parameter_SetUI(g.params, NVSDK_NGX_Parameter_Height, outH);
        NVSDK_NGX_Parameter_SetI(g.params, NVSDK_NGX_Parameter_PerfQualityValue, quality);
        NVSDK_NGX_Parameter_SetI(g.params, NVSDK_NGX_Parameter_RTXValue, 0);
        r = reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback>(callback)(g.params);
        if (!NVSDK_NGX_FAILED(r))
        {
            NVSDK_NGX_Parameter_GetUI(g.params, NVSDK_NGX_Parameter_OutWidth, w);
            NVSDK_NGX_Parameter_GetUI(g.params, NVSDK_NGX_Parameter_OutHeight, h);
            *minW = *maxW = *w;
            *minH = *maxH = *h;
            NVSDK_NGX_Parameter_GetUI(g.params, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Width, maxW);
            NVSDK_NGX_Parameter_GetUI(g.params, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Height, maxH);
            NVSDK_NGX_Parameter_GetUI(g.params, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Width, minW);
            NVSDK_NGX_Parameter_GetUI(g.params, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Height, minH);
            *sharpness = 0.0f;
        }
    }
    if (NVSDK_NGX_FAILED(r))
    {
        LOG_ERROR("sl-ngx: no optimal settings for mode %u at %ux%u: %#x", (unsigned)mode, outW, outH, (unsigned)r);
        return Result::eErrorNGXFailed;
    }
    if (*w == 0 || *h == 0)
    {
        LOG_WARN("sl-ngx: NGX has no render size for mode %u at %ux%u", (unsigned)mode, outW, outH);
        return Result::eErrorInvalidParameter;
    }
    g.optimal[key] = {*w, *h, *minW, *minH, *maxW, *maxH, *sharpness};
    return Result::eOk;
}

bool isDepth(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

NVSDK_NGX_Resource_VK ngxImage(const sl::Resource& r, bool readWrite)
{
    auto format = (VkFormat)r.nativeFormat;
    VkImageSubresourceRange range{};
    range.aspectMask = isDepth(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = VK_REMAINING_MIP_LEVELS;
    range.baseArrayLayer = 0;
    range.layerCount = VK_REMAINING_ARRAY_LAYERS;
    return NVSDK_NGX_Create_ImageView_Resource_VK(reinterpret_cast<VkImageView>(r.view),
        reinterpret_cast<VkImage>(r.native), range, format, r.width, r.height, readWrite);
}

sl::Extent extentOf(const Tag& t)
{
    if (t.extent.width && t.extent.height)
        return t.extent;
    sl::Extent e{};
    e.width = t.resource.width;
    e.height = t.resource.height;
    return e;
}

const Tag* findTag(const std::map<sl::BufferType, Tag>& tags, sl::BufferType type)
{
    auto it = tags.find(type);
    return it == tags.end() ? nullptr : &it->second;
}

// The newest constants set for a viewport, for a feature made outside a frame.
const sl::Constants* latestConstants(Viewport& vp)
{
    for (auto it = vp.frames.rbegin(); it != vp.frames.rend(); ++it)
        if (it->second.hasConstants)
            return &it->second.constants;
    return nullptr;
}

Result shapeFor(sl::Feature feature, Viewport& vp, const sl::Constants& c, const Tag* colorIn, const Tag* mvec,
    bool exposureTagged, Shape& s)
{
    const bool rr = feature == sl::kFeatureDLSS_RR;
    if (rr ? !vp.hasDlssd : !vp.hasDlss)
        return Result::eErrorMissingInputParameter;
    s = Shape{};
    s.feature = feature;
    sl::DLSSPreset p[6];
    if (rr)
    {
        const auto& o = vp.dlssd;
        s.mode = o.mode;
        s.outWidth = o.outputWidth;
        s.outHeight = o.outputHeight;
        s.roughnessMode = o.normalRoughnessMode == sl::DLSSDNormalRoughnessMode::ePacked
                              ? NVSDK_NGX_DLSS_Roughness_Mode_Packed
                              : NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
        const sl::DLSSDPreset q[6] = {o.dlaaPreset, o.qualityPreset, o.balancedPreset, o.performancePreset,
            o.ultraPerformancePreset, o.ultraQualityPreset};
        for (int i = 0; i < 6; i++)
            s.presets[i] = (uint32_t)q[i];
    }
    else
    {
        const auto& o = vp.dlss;
        s.mode = o.mode;
        s.outWidth = o.outputWidth;
        s.outHeight = o.outputHeight;
        const sl::DLSSPreset q[6] = {o.dlaaPreset, o.qualityPreset, o.balancedPreset, o.performancePreset,
            o.ultraPerformancePreset, o.ultraQualityPreset};
        for (int i = 0; i < 6; i++)
            s.presets[i] = (uint32_t)q[i];
        (void)p;
    }
    if (s.mode == sl::DLSSMode::eOff)
        return Result::eErrorInvalidParameter;
    uint32_t minW, minH, maxW, maxH;
    float sharp;
    Result r = optimal(feature, s.mode, s.outWidth, s.outHeight, &s.renderWidth, &s.renderHeight, &minW, &minH,
        &maxW, &maxH, &sharp);
    if (r != Result::eOk)
        return r;

    const sl::Boolean hdr = rr ? vp.dlssd.colorBuffersHDR : vp.dlss.colorBuffersHDR;
    const sl::Boolean alpha = rr ? vp.dlssd.alphaUpscalingEnabled : vp.dlss.alphaUpscalingEnabled;
    int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (hdr == sl::Boolean::eTrue)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    if (c.depthInverted == sl::Boolean::eTrue)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (c.motionVectorsJittered == sl::Boolean::eTrue)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
    if (alpha == sl::Boolean::eTrue)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_AlphaUpscaling;
    if (!rr && (vp.dlss.useAutoExposure == sl::Boolean::eTrue || !exposureTagged))
        flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (colorIn && mvec)
    {
        auto ce = extentOf(*colorIn), me = extentOf(*mvec);
        if (me.width > ce.width || me.height > ce.height)
            flags &= ~NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    }
    s.flags = flags;
    return Result::eOk;
}

Result create(VkCommandBuffer cmd, Viewport& vp, const Shape& s)
{
    const bool rr = s.feature == sl::kFeatureDLSS_RR;
    static const char* const kDlssHints[6] = {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
        NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality};
    static const char* const kRrHints[6] = {NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality};
    // The parameter object is shared by every feature NGX makes: each
    // creation writes all of both features' hints, the other's as default.
    for (int i = 0; i < 6; i++)
    {
        NVSDK_NGX_Parameter_SetUI(g.params, kDlssHints[i], rr ? 0u : s.presets[i]);
        NVSDK_NGX_Parameter_SetUI(g.params, kRrHints[i], rr ? s.presets[i] : 0u);
    }
    auto quality = (NVSDK_NGX_PerfQuality_Value)((uint32_t)s.mode - 1);
    NVSDK_NGX_Handle* handle = nullptr;
    NVSDK_NGX_Result r;
    if (rr)
    {
        NVSDK_NGX_DLSSD_Create_Params p{};
        p.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        p.InRoughnessMode = (NVSDK_NGX_DLSS_Roughness_Mode)s.roughnessMode;
        p.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
        p.InWidth = s.renderWidth;
        p.InHeight = s.renderHeight;
        p.InTargetWidth = s.outWidth;
        p.InTargetHeight = s.outHeight;
        p.InPerfQualityValue = quality;
        p.InFeatureCreateFlags = s.flags;
        p.InEnableOutputSubrects = false;
        r = NGX_VULKAN_CREATE_DLSSD_EXT1(g.device, cmd, 1, 1, &handle, g.params, &p);
    }
    else
    {
        // a Ray Reconstruction made before leaves its mode on the shared
        // parameters (Streamline's DLSS-D plugin resets it on release)
        NVSDK_NGX_Parameter_SetI(g.params, NVSDK_NGX_Parameter_DLSS_Denoise_Mode, NVSDK_NGX_DLSS_Denoise_Mode_Off);
        NVSDK_NGX_Parameter_SetUI(g.params, NVSDK_NGX_Parameter_DLSS_Roughness_Mode, 0);
        NVSDK_NGX_Parameter_SetUI(g.params, NVSDK_NGX_Parameter_Use_HW_Depth, 0);
        NVSDK_NGX_DLSS_Create_Params p{};
        p.Feature.InWidth = s.renderWidth;
        p.Feature.InHeight = s.renderHeight;
        p.Feature.InTargetWidth = s.outWidth;
        p.Feature.InTargetHeight = s.outHeight;
        p.Feature.InPerfQualityValue = quality;
        p.InFeatureCreateFlags = s.flags;
        p.InEnableOutputSubrects = false;
        r = NGX_VULKAN_CREATE_DLSS_EXT1(g.device, cmd, 1, 1, &handle, g.params, &p);
    }
    if (NVSDK_NGX_FAILED(r) || !handle)
    {
        LOG_ERROR("sl-ngx: NGX did not make the %s feature for %ux%u from %ux%u: %#x", rr ? "Ray Reconstruction" : "DLSS",
            s.outWidth, s.outHeight, s.renderWidth, s.renderHeight, (unsigned)r);
        return Result::eErrorNGXFailed;
    }
    if (vp.handle)
        g.retired.push_back({vp.handle, g.issued + kReleaseAfterFrames});
    vp.handle = handle;
    vp.shape = s;
    LOG_INFO("sl-ngx: %s feature made, %ux%u to %ux%u, mode %u, flags %#x, presets %u %u %u %u %u %u%s",
        rr ? "Ray Reconstruction" : "DLSS", s.renderWidth, s.renderHeight, s.outWidth, s.outHeight, (unsigned)s.mode,
        (unsigned)s.flags, s.presets[0], s.presets[1], s.presets[2], s.presets[3], s.presets[4], s.presets[5],
        rr ? (s.roughnessMode == NVSDK_NGX_DLSS_Roughness_Mode_Packed ? ", roughness packed" : ", roughness unpacked")
           : "");
    vp.logNextEvaluate = true;
    return Result::eOk;
}

Result requirementsFor(sl::Feature feature)
{
    auto& inst = g.instanceExtNames[feature];
    auto& dev = g.deviceExtNames[feature];
    inst.clear();
    dev.clear();
    auto add = [](std::vector<std::string>& to, const char* name) {
        for (auto& n : to)
            if (n == name)
                return;
        to.emplace_back(name);
    };
    // what Streamline's DLSS plugins ask for on any NVIDIA device
    for (auto n : {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
             VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME})
        add(inst, n);
    for (auto n : {VK_NVX_BINARY_IMPORT_EXTENSION_NAME, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME,
             VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME})
        add(dev, n);

    // and what NGX itself names
    auto info = discovery(feature);
    uint32_t count = 0;
    VkExtensionProperties* props = nullptr;
    if (!NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &count, &props)))
        for (uint32_t i = 0; i < count && props; i++)
            add(inst, props[i].extensionName);
    if (g.queryInstance && g.queryPhysicalDevice)
    {
        count = 0;
        props = nullptr;
        if (!NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                g.queryInstance, g.queryPhysicalDevice, &info, &count, &props)))
            for (uint32_t i = 0; i < count && props; i++)
                add(dev, props[i].extensionName);
    }
    auto& ip = g.instanceExts[feature];
    auto& dp = g.deviceExts[feature];
    ip.clear();
    dp.clear();
    for (auto& n : inst)
        ip.push_back(n.c_str());
    for (auto& n : dev)
        dp.push_back(n.c_str());
    return Result::eOk;
}

// ---- feature functions, reached through slGetFeatureFunction ----

Result dlssGetOptimalSettings(const sl::DLSSOptions& o, sl::DLSSOptimalSettings& s)
{
    std::lock_guard<std::mutex> l(g.lock);
    return optimal(sl::kFeatureDLSS, o.mode, o.outputWidth, o.outputHeight, &s.optimalRenderWidth,
        &s.optimalRenderHeight, &s.renderWidthMin, &s.renderHeightMin, &s.renderWidthMax, &s.renderHeightMax,
        &s.optimalSharpness);
}

Result dlssdGetOptimalSettings(const sl::DLSSDOptions& o, sl::DLSSDOptimalSettings& s)
{
    std::lock_guard<std::mutex> l(g.lock);
    return optimal(sl::kFeatureDLSS_RR, o.mode, o.outputWidth, o.outputHeight, &s.optimalRenderWidth,
        &s.optimalRenderHeight, &s.renderWidthMin, &s.renderHeightMin, &s.renderWidthMax, &s.renderHeightMax,
        &s.optimalSharpness);
}

// What NGX's own estimate says the feature's features hold in video memory.
Result vramInUse(sl::Feature feature, uint64_t* out)
{
    *out = 0;
    if (!g.started)
        return Result::eErrorDeviceNotCreated;
    unsigned long long bytes = 0;
    NVSDK_NGX_Result r;
    if (feature == sl::kFeatureDLSS)
    {
        r = NGX_DLSS_GET_STATS(g.params, &bytes);
    }
    else
    {
        // DLSS-D reports through its own callback (as Streamline's DLSS-D plugin reads it)
        void* callback = nullptr;
        NVSDK_NGX_Parameter_GetVoidPointer(g.params, NVSDK_NGX_Parameter_DLSSDGetStatsCallback, &callback);
        if (!callback)
            return Result::eErrorFeatureNotSupported;
        r = reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetStatsCallback>(callback)(g.params);
        if (!NVSDK_NGX_FAILED(r))
            NVSDK_NGX_Parameter_GetULL(g.params, NVSDK_NGX_Parameter_SizeInBytes, &bytes);
    }
    if (NVSDK_NGX_FAILED(r))
        return Result::eErrorNGXFailed;
    *out = bytes;
    return Result::eOk;
}

Result dlssGetState(const sl::ViewportHandle&, sl::DLSSState& s)
{
    std::lock_guard<std::mutex> l(g.lock);
    uint64_t bytes;
    Result r = vramInUse(sl::kFeatureDLSS, &bytes);
    s.estimatedVRAMUsageInBytes = bytes;
    return r;
}

Result dlssdGetState(const sl::ViewportHandle&, sl::DLSSDState& s)
{
    std::lock_guard<std::mutex> l(g.lock);
    uint64_t bytes;
    Result r = vramInUse(sl::kFeatureDLSS_RR, &bytes);
    s.estimatedVRAMUsageInBytes = bytes;
    return r;
}

Result dlssSetOptions(const sl::ViewportHandle& viewport, const sl::DLSSOptions& o)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!sameType(o.structType, sl::DLSSOptions::s_structType) || o.structVersion < sl::kStructVersion3)
        return Result::eErrorInvalidParameter;
    auto& vp = g.viewports[(uint32_t)viewport];
    vp.dlss = o;
    vp.dlss.next = nullptr;
    vp.hasDlss = true;
    return Result::eOk;
}

Result dlssdSetOptions(const sl::ViewportHandle& viewport, const sl::DLSSDOptions& o)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!sameType(o.structType, sl::DLSSDOptions::s_structType) || o.structVersion < sl::kStructVersion3)
        return Result::eErrorInvalidParameter;
    auto& vp = g.viewports[(uint32_t)viewport];
    vp.dlssd = o;
    vp.dlssd.next = nullptr;
    vp.hasDlssd = true;
    return Result::eOk;
}

// ---- frame generation (sl-ngx's own use of the Streamline interface) ----
//
// Streamline's DLSS-G hooks the swapchain and is never evaluated by the
// application. sl-ngx has no hook: the caller evaluates kFeatureDLSS_G itself
// once per rendered frame and presents the frame it gets back before the
// rendered one. Tags: kBufferTypeHUDLessColor (the rendered frame, display
// size, without the interface), depth and motion vectors (the frame's, as
// for DLSS, at render size), and, handed with the evaluate itself (the
// frame's own are the upscaler's): the output, kBufferTypeScalingOutputColor,
// and optionally the frame as presented, with its interface, as
// kBufferTypeScalingInputColor (Streamline has no tag for the backbuffer, which
// it takes from the swapchain). Without it, the frame without interface is
// the backbuffer. Motion vectors are taken as Streamline defines them:
// scaled by mvecScale into [-1,1], current to previous.
Result evaluateFg(VkCommandBuffer cmd, Viewport& vp, const sl::Constants& c, const std::map<sl::BufferType, Tag>& tags,
    const std::map<sl::BufferType, Tag>& local)
{
    const Tag* hudless = findTag(tags, sl::kBufferTypeHUDLessColor);
    const Tag* shown = findTag(local, sl::kBufferTypeScalingInputColor);
    const Tag* back = shown ? shown : hudless;
    const Tag* out = findTag(local, sl::kBufferTypeScalingOutputColor);
    const Tag* depth = findTag(tags, sl::kBufferTypeDepth);
    const Tag* mvec = findTag(tags, sl::kBufferTypeMotionVectors);
    if (!back || !out || !depth || !mvec)
    {
        LOG_ERROR("sl-ngx: frame generation needs the frame without interface, depth and motion vectors tagged, "
                  "and its output handed with the evaluate");
        return Result::eErrorMissingInputParameter;
    }
    const auto bExt = extentOf(*back), oExt = extentOf(*out), dExt = extentOf(*depth), mExt = extentOf(*mvec);
    if (!vp.fgHandle || vp.fgWidth != bExt.width || vp.fgHeight != bExt.height ||
        vp.fgFormat != back->resource.nativeFormat || vp.fgRenderWidth != mExt.width ||
        vp.fgRenderHeight != mExt.height)
    {
        if (vp.fgHandle)
            g.retired.push_back({vp.fgHandle, g.issued + kReleaseAfterFrames});
        vp.fgHandle = nullptr;
        NVSDK_NGX_DLSSG_Create_Params p{};
        p.Width = bExt.width;
        p.Height = bExt.height;
        p.NativeBackbufferFormat = back->resource.nativeFormat;
        p.RenderWidth = mExt.width;
        p.RenderHeight = mExt.height;
        p.DynamicResolutionScaling = false;
        NVSDK_NGX_Result r = NGX_VK_CREATE_DLSSG(cmd, 1, 1, &vp.fgHandle, g.params, &p);
        if (NVSDK_NGX_FAILED(r) || !vp.fgHandle)
        {
            vp.fgHandle = nullptr;
            LOG_ERROR("sl-ngx: NGX did not make frame generation for %ux%u (format %u) from %ux%u: %#x", bExt.width,
                bExt.height, back->resource.nativeFormat, mExt.width, mExt.height, (unsigned)r);
            return Result::eErrorNGXFailed;
        }
        vp.fgWidth = bExt.width;
        vp.fgHeight = bExt.height;
        vp.fgFormat = back->resource.nativeFormat;
        vp.fgRenderWidth = mExt.width;
        vp.fgRenderHeight = mExt.height;
        vp.fgFrameId = 0;
        LOG_INFO("sl-ngx: frame generation made, %ux%u (format %u), render %ux%u", bExt.width, bExt.height,
            back->resource.nativeFormat, mExt.width, mExt.height);
    }
    vp.lastUsed = g.issued;

    NVSDK_NGX_Resource_VK backRes = ngxImage(back->resource, false);
    NVSDK_NGX_Resource_VK hudlessRes = hudless ? ngxImage(hudless->resource, false) : NVSDK_NGX_Resource_VK{};
    NVSDK_NGX_Resource_VK outRes = ngxImage(out->resource, true);
    NVSDK_NGX_Resource_VK depthRes = ngxImage(depth->resource, false);
    NVSDK_NGX_Resource_VK mvecRes = ngxImage(mvec->resource, false);
    NVSDK_NGX_VK_DLSSG_Eval_Params e{};
    e.pBackbuffer = &backRes;
    if (shown && hudless)
        e.pHudless = &hudlessRes;
    e.pDepth = &depthRes;
    e.pMVecs = &mvecRes;
    e.pOutputInterpFrame = &outRes;
    NVSDK_NGX_DLSSG_Opt_Eval_Params o{};
    o.multiFrameCount = 1;
    o.multiFrameIndex = 1;
    std::memcpy(o.cameraViewToClip, &c.cameraViewToClip, sizeof o.cameraViewToClip);
    std::memcpy(o.clipToCameraView, &c.clipToCameraView, sizeof o.clipToCameraView);
    std::memcpy(o.clipToLensClip, &c.clipToLensClip, sizeof o.clipToLensClip);
    std::memcpy(o.clipToPrevClip, &c.clipToPrevClip, sizeof o.clipToPrevClip);
    std::memcpy(o.prevClipToClip, &c.prevClipToClip, sizeof o.prevClipToClip);
    // the backbuffer is the upscaler's output: no jitter left in it
    o.jitterOffset[0] = o.jitterOffset[1] = 0.0f;
    o.mvecScale[0] = c.mvecScale.x;
    o.mvecScale[1] = c.mvecScale.y;
    o.cameraPinholeOffset[0] = c.cameraPinholeOffset.x;
    o.cameraPinholeOffset[1] = c.cameraPinholeOffset.y;
    std::memcpy(o.cameraPos, &c.cameraPos, sizeof o.cameraPos);
    std::memcpy(o.cameraUp, &c.cameraUp, sizeof o.cameraUp);
    std::memcpy(o.cameraRight, &c.cameraRight, sizeof o.cameraRight);
    std::memcpy(o.cameraFwd, &c.cameraFwd, sizeof o.cameraFwd);
    o.cameraNear = c.cameraNear;
    o.cameraFar = c.cameraFar;
    o.cameraFOV = c.cameraFOV;
    o.cameraAspectRatio = c.cameraAspectRatio;
    o.colorBuffersHDR = false;
    o.depthInverted = c.depthInverted == sl::Boolean::eTrue;
    o.cameraMotionIncluded = c.cameraMotionIncluded == sl::Boolean::eTrue;
    o.reset = c.reset == sl::Boolean::eTrue || vp.fgFrameId == 0;
    o.orthoProjection = c.orthographicProjection == sl::Boolean::eTrue;
    o.motionVectorsInvalidValue = c.motionVectorsInvalidValue;
    o.motionVectorsDilated = c.motionVectorsDilated == sl::Boolean::eTrue;
    o.mvecsSubrectBase = {mExt.left, mExt.top};
    o.mvecsSubrectSize = {mExt.width, mExt.height};
    o.depthSubrectBase = {dExt.left, dExt.top};
    o.depthSubrectSize = {dExt.width, dExt.height};
    o.backbufferSubrectBase = {bExt.left, bExt.top};
    o.backbufferSubrectSize = {bExt.width, bExt.height};
    o.outputInterpSubrectBase = {oExt.left, oExt.top};
    o.outputInterpSubrectSize = {oExt.width, oExt.height};
    NVSDK_NGX_Parameter_SetULL(g.params, NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, ++vp.fgFrameId);
    NVSDK_NGX_Result r = NGX_VK_EVALUATE_DLSSG(cmd, vp.fgHandle, g.params, &e, &o);
    // the shared parameters must not keep pointers to these locals
    for (const char* name : {NVSDK_NGX_DLSSG_Parameter_Backbuffer, NVSDK_NGX_DLSSG_Parameter_MVecs,
             NVSDK_NGX_DLSSG_Parameter_Depth, NVSDK_NGX_DLSSG_Parameter_HUDLess, NVSDK_NGX_DLSSG_Parameter_UI,
             NVSDK_NGX_DLSSG_Parameter_UIAlpha, NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField,
             NVSDK_NGX_DLSSG_Parameter_OutputInterpolated, NVSDK_NGX_DLSSG_Parameter_OutputReal,
             NVSDK_NGX_DLSSG_Parameter_OutputDisableInterpolation, NVSDK_NGX_DLSSG_Parameter_CameraViewToClip,
             NVSDK_NGX_DLSSG_Parameter_ClipToCameraView, NVSDK_NGX_DLSSG_Parameter_ClipToLensClip,
             NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip, NVSDK_NGX_DLSSG_Parameter_PrevClipToClip})
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, name, nullptr);
    if (NVSDK_NGX_FAILED(r))
    {
        LOG_ERROR("sl-ngx: NGX frame generation failed: %#x", (unsigned)r);
        return Result::eErrorNGXFailed;
    }
    if (vp.fgFrameId == 1)
        LOG_INFO("sl-ngx: first frame generation: %ux%u from render %ux%u, motion scale %.6f %.6f, depth inverted %d, "
                 "interface %s",
            bExt.width, bExt.height, mExt.width, mExt.height, o.mvecScale[0], o.mvecScale[1], (int)o.depthInverted,
            shown && hudless ? "from the hudless difference" : "none");
    return Result::eOk;
}

} // namespace

// ---- the exported interface (sl_core_api.h, sl_helpers_vk.h) ----

extern "C" {

SL_NGX_EXPORT Result slInit(const sl::Preferences& pref, uint64_t sdkVersion)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (g.initialized)
        return Result::eErrorInvalidState;
    if ((sdkVersion & 0xffff) != sl::kSDKVersionMagic)
        return Result::eErrorInvalidParameter;
    if (!sameType(pref.structType, sl::Preferences::s_structType))
        return Result::eErrorInvalidParameter;
    g.log = pref.logMessageCallback;
    g.logLevel = pref.logLevel;
    if (pref.renderAPI != sl::RenderAPI::eVulkan)
    {
        LOG_ERROR("sl-ngx: only Vulkan is implemented");
        return Result::eErrorUnsupportedInterface;
    }
    g.projectId = pref.projectId ? pref.projectId : "";
    g.engineVersion = pref.engineVersion ? pref.engineVersion : "";
    if (g.projectId.empty())
    {
        LOG_ERROR("sl-ngx: Preferences.projectId is required");
        return Result::eErrorMissingInputParameter;
    }
    g.engine = (NVSDK_NGX_EngineType)pref.engine;
    g.dataPath = pref.pathToLogsAndData ? std::wstring(pref.pathToLogsAndData) : ownFolder();
    g.modelPaths.clear();
    for (uint32_t i = 0; i < pref.numPathsToPlugins && pref.pathsToPlugins; i++)
        if (pref.pathsToPlugins[i])
            g.modelPaths.emplace_back(pref.pathsToPlugins[i]);
    if (g.modelPaths.empty())
        g.modelPaths.push_back(ownFolder());
    g.modelPathPtrs.clear();
    for (auto& p : g.modelPaths)
        g.modelPathPtrs.push_back(p.c_str());
    g.common = NVSDK_NGX_FeatureCommonInfo{};
    g.common.PathListInfo.Path = g.modelPathPtrs.data();
    g.common.PathListInfo.Length = (unsigned)g.modelPathPtrs.size();
    for (uint32_t i = 0; i < kTokenRing; i++)
        g.tokens[i].index = 0;
    g.initialized = true;
    LOG_INFO("sl-ngx: initialized");
    return Result::eOk;
}

SL_NGX_EXPORT Result slShutdown()
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorNotInitialized;
    stopNgx();
    g.viewports.clear();
    g.instanceExts.clear();
    g.deviceExts.clear();
    g.instanceExtNames.clear();
    g.deviceExtNames.clear();
    g.instance = VK_NULL_HANDLE;
    g.physicalDevice = VK_NULL_HANDLE;
    g.device = VK_NULL_HANDLE;
    g.queryInstance = VK_NULL_HANDLE;
    g.queryPhysicalDevice = VK_NULL_HANDLE;
    g.initialized = false;
    return Result::eOk;
}

// A VulkanInfo without a device names the instance and physical device
// only, so slGetFeatureRequirements can ask NGX for that device's
// extensions before the device is made (real Streamline makes a device of
// its own for that), and leaves a running start as it is. With a device,
// NGX starts on it. A different device
// while NGX runs (a new device after a device loss) leaves NGX on the old
// one without calling it there (abandonNgx); slShutdown is the clean end,
// for a device that still works and is idle.
SL_NGX_EXPORT Result slSetVulkanInfo(const sl::VulkanInfo& info)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!sameType(info.structType, sl::VulkanInfo::s_structType) || !info.instance || !info.physicalDevice)
        return Result::eErrorInvalidParameter;
    g.queryInstance = info.instance;
    g.queryPhysicalDevice = info.physicalDevice;
    // without a device: only what slGetFeatureRequirements asks about, the
    // running start untouched (a new instance is probed before its device)
    if (!info.device)
        return Result::eOk;
    if (g.started && info.device == g.device)
        return Result::eOk;
    if (g.started)
        abandonNgx();
    g.instance = info.instance;
    g.physicalDevice = info.physicalDevice;
    g.device = info.device;
    return startNgx();
}

SL_NGX_EXPORT Result slIsFeatureSupported(sl::Feature feature, const sl::AdapterInfo& adapter)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!ours(feature))
        return Result::eErrorFeatureMissing;
    if (adapter.vkPhysicalDevice && adapter.vkPhysicalDevice != (void*)g.physicalDevice)
        return Result::eErrorAdapterNotSupported;
    return supported(feature);
}

SL_NGX_EXPORT Result slGetFeatureRequirements(sl::Feature feature, sl::FeatureRequirements& req)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!ours(feature))
        return Result::eErrorFeatureMissing;
    Result r = requirementsFor(feature);
    if (r != Result::eOk)
        return r;
    static const sl::BufferType kDlssTags[] = {sl::kBufferTypeDepth, sl::kBufferTypeMotionVectors,
        sl::kBufferTypeScalingInputColor, sl::kBufferTypeScalingOutputColor};
    static const sl::BufferType kRrTags[] = {sl::kBufferTypeDepth, sl::kBufferTypeMotionVectors,
        sl::kBufferTypeScalingInputColor, sl::kBufferTypeScalingOutputColor, sl::kBufferTypeAlbedo,
        sl::kBufferTypeSpecularAlbedo, sl::kBufferTypeNormalRoughness};
    static const sl::BufferType kFgTags[] = {sl::kBufferTypeDepth, sl::kBufferTypeMotionVectors,
        sl::kBufferTypeHUDLessColor};
    static const char* kFeatures12[] = {"timelineSemaphore", "descriptorIndexing", "bufferDeviceAddress"};
    const bool rr = feature == sl::kFeatureDLSS_RR;
    const bool fg = feature == sl::kFeatureDLSS_G;
    req.flags = sl::FeatureRequirementFlags::eVulkanSupported;
    req.maxNumCPUThreads = 1;
    req.maxNumViewports = 0;
    req.numRequiredTags = fg ? 3 : rr ? 7 : 4;
    req.requiredTags = fg ? kFgTags : rr ? kRrTags : kDlssTags;
    req.vkNumComputeQueuesRequired = 0;
    req.vkNumGraphicsQueuesRequired = 0;
    req.vkNumDeviceExtensions = (uint32_t)g.deviceExts[feature].size();
    req.vkDeviceExtensions = g.deviceExts[feature].data();
    req.vkNumInstanceExtensions = (uint32_t)g.instanceExts[feature].size();
    req.vkInstanceExtensions = g.instanceExts[feature].data();
    if (req.structVersion >= sl::kStructVersion2)
    {
        req.vkNumFeatures12 = 3;
        req.vkFeatures12 = kFeatures12;
        req.vkNumFeatures13 = 0;
        req.vkFeatures13 = nullptr;
        req.vkNumOpticalFlowQueuesRequired = 0;
    }
    return Result::eOk;
}

SL_NGX_EXPORT Result slGetFeatureVersion(sl::Feature feature, sl::FeatureVersion& version)
{
    if (!ours(feature))
        return Result::eErrorFeatureMissing;
    version.versionSL = sl::Version(SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH);
    version.versionNGX = sl::Version(0, 0, 0);
    return Result::eOk;
}

SL_NGX_EXPORT Result slGetFeatureFunction(sl::Feature feature, const char* name, void*& function)
{
    function = nullptr;
    if (!name)
        return Result::eErrorInvalidParameter;
    struct Entry
    {
        sl::Feature feature;
        const char* name;
        void* fn;
    };
    static const Entry kTable[] = {
        {sl::kFeatureDLSS, "slDLSSGetOptimalSettings", reinterpret_cast<void*>(&dlssGetOptimalSettings)},
        {sl::kFeatureDLSS, "slDLSSGetState", reinterpret_cast<void*>(&dlssGetState)},
        {sl::kFeatureDLSS, "slDLSSSetOptions", reinterpret_cast<void*>(&dlssSetOptions)},
        {sl::kFeatureDLSS_RR, "slDLSSDGetOptimalSettings", reinterpret_cast<void*>(&dlssdGetOptimalSettings)},
        {sl::kFeatureDLSS_RR, "slDLSSDGetState", reinterpret_cast<void*>(&dlssdGetState)},
        {sl::kFeatureDLSS_RR, "slDLSSDSetOptions", reinterpret_cast<void*>(&dlssdSetOptions)},
    };
    for (auto& e : kTable)
    {
        if (e.feature == feature && std::strcmp(e.name, name) == 0)
        {
            function = e.fn;
            return Result::eOk;
        }
    }
    return ours(feature) ? Result::eErrorMissingOrInvalidAPI : Result::eErrorFeatureMissing;
}

SL_NGX_EXPORT Result slGetNewFrameToken(sl::FrameToken*& token, const uint32_t* frameIndex)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    Token& t = g.tokens[g.nextToken];
    g.nextToken = (g.nextToken + 1) % kTokenRing;
    t.index = frameIndex ? *frameIndex : ++g.frameCounter;
    if (frameIndex)
        g.frameCounter = *frameIndex;
    g.issued++;
    releaseDue(false);
    token = &t;
    return Result::eOk;
}

SL_NGX_EXPORT Result slSetConstants(const sl::Constants& values, const sl::FrameToken& frame,
    const sl::ViewportHandle& viewport)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!sameType(values.structType, sl::Constants::s_structType) || values.structVersion < sl::kStructVersion2)
        return Result::eErrorInvalidParameter;
    auto& vp = g.viewports[(uint32_t)viewport];
    auto& f = vp.frames[(uint32_t)frame];
    f.constants = values;
    f.constants.next = nullptr;
    f.hasConstants = true;
    while (vp.frames.size() > kFramesKept)
        vp.frames.erase(vp.frames.begin());
    return Result::eOk;
}

SL_NGX_EXPORT Result slSetTagForFrame(const sl::FrameToken& frame, const sl::ViewportHandle& viewport,
    const sl::ResourceTag* resources, uint32_t numResources, sl::CommandBuffer*)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (numResources && !resources)
        return Result::eErrorInvalidParameter;
    auto& vp = g.viewports[(uint32_t)viewport];
    auto& f = vp.frames[(uint32_t)frame];
    for (uint32_t i = 0; i < numResources; i++)
    {
        const auto& t = resources[i];
        if (!t.resource)
        {
            f.tags.erase(t.type);
            continue;
        }
        // nothing is copied here: a tag must stay valid until the evaluate
        if (t.lifecycle == sl::ResourceLifecycle::eOnlyValidNow)
        {
            LOG_ERROR("sl-ngx: tag %u is eOnlyValidNow; this library does not copy resources", (unsigned)t.type);
            return Result::eErrorInvalidParameter;
        }
        if (t.resource->type != sl::ResourceType::eTex2d || !t.resource->native || !t.resource->view ||
            !t.resource->width || !t.resource->height || !t.resource->nativeFormat)
        {
            LOG_ERROR("sl-ngx: tag %u needs a Vulkan image with its view, size and format", (unsigned)t.type);
            return Result::eErrorInvalidParameter;
        }
        Tag tag;
        tag.resource = *t.resource;
        tag.resource.next = nullptr;
        tag.extent = t.extent;
        f.tags[t.type] = tag;
    }
    while (vp.frames.size() > kFramesKept)
        vp.frames.erase(vp.frames.begin());
    return Result::eOk;
}

SL_NGX_EXPORT Result slAllocateResources(sl::CommandBuffer* cmdBuffer, sl::Feature feature,
    const sl::ViewportHandle& viewport)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!ours(feature))
        return Result::eErrorFeatureMissing;
    if (!g.started)
        return Result::eErrorDeviceNotCreated;
    if (!cmdBuffer)
        return Result::eErrorInvalidParameter;
    // frame generation is made at its first evaluate, which knows the sizes
    if (feature == sl::kFeatureDLSS_G)
        return Result::eOk;
    auto& vp = g.viewports[(uint32_t)viewport];
    const sl::Constants* c = latestConstants(vp);
    if (!c)
        return Result::eErrorMissingConstants;
    Shape s;
    Result r = shapeFor(feature, vp, *c, nullptr, nullptr, false, s);
    if (r != Result::eOk)
        return r;
    if (vp.handle && vp.shape == s)
        return Result::eOk;
    return create(reinterpret_cast<VkCommandBuffer>(cmdBuffer), vp, s);
}

// The caller makes sure the GPU is done with the viewport's last evaluate.
SL_NGX_EXPORT Result slFreeResources(sl::Feature feature, const sl::ViewportHandle& viewport)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!ours(feature))
        return Result::eErrorFeatureMissing;
    auto it = g.viewports.find((uint32_t)viewport);
    if (it == g.viewports.end())
        return Result::eOk;
    if (feature == sl::kFeatureDLSS_G)
    {
        releaseNow(it->second.fgHandle);
        it->second.fgHandle = nullptr;
        it->second.fgFrameId = 0;
        if (!it->second.handle && !it->second.hasDlss && !it->second.hasDlssd)
            g.viewports.erase(it);
        return Result::eOk;
    }
    if (it->second.shape.feature == feature)
    {
        releaseNow(it->second.handle);
        it->second.handle = nullptr;
        it->second.shape = Shape{};
    }
    if (feature == sl::kFeatureDLSS)
        it->second.hasDlss = false;
    else
        it->second.hasDlssd = false;
    if (!it->second.handle && !it->second.fgHandle && !it->second.hasDlss && !it->second.hasDlssd)
        g.viewports.erase(it);
    return Result::eOk;
}

SL_NGX_EXPORT Result slEvaluateFeature(sl::Feature feature, const sl::FrameToken& frame,
    const sl::BaseStructure** inputs, uint32_t numInputs, sl::CommandBuffer* cmdBuffer)
{
    std::lock_guard<std::mutex> l(g.lock);
    if (!g.initialized)
        return Result::eErrorInitNotCalled;
    if (!ours(feature))
        return Result::eErrorFeatureMissing;
    if (!g.started)
        return Result::eErrorDeviceNotCreated;
    if (!cmdBuffer || (numInputs && !inputs))
        return Result::eErrorInvalidParameter;

    // the viewport, and any tags handed with the evaluate itself
    const sl::ViewportHandle* handle = nullptr;
    std::map<sl::BufferType, Tag> local;
    for (uint32_t i = 0; i < numInputs; i++)
    {
        for (const sl::BaseStructure* s = inputs[i]; s; s = s->next)
        {
            if (sameType(s->structType, sl::ViewportHandle::s_structType))
                handle = static_cast<const sl::ViewportHandle*>(s);
            else if (sameType(s->structType, sl::ResourceTag::s_structType))
            {
                auto t = static_cast<const sl::ResourceTag*>(s);
                if (t->resource)
                {
                    Tag tag;
                    tag.resource = *t->resource;
                    tag.resource.next = nullptr;
                    tag.extent = t->extent;
                    local[t->type] = tag;
                }
            }
        }
    }
    if (!handle)
        return Result::eErrorMissingInputParameter;
    auto vit = g.viewports.find((uint32_t)*handle);
    if (vit == g.viewports.end())
        return Result::eErrorMissingInputParameter;
    Viewport& vp = vit->second;
    auto fit = vp.frames.find((uint32_t)frame);
    if (fit == vp.frames.end() || !fit->second.hasConstants)
        return Result::eErrorMissingConstants;
    const sl::Constants& c = fit->second.constants;
    std::map<sl::BufferType, Tag> tags = fit->second.tags;
    for (auto& [type, tag] : local)
        tags[type] = tag;

    if (feature == sl::kFeatureDLSS_G)
        return evaluateFg(reinterpret_cast<VkCommandBuffer>(cmdBuffer), vp, c, tags, local);

    const bool rr = feature == sl::kFeatureDLSS_RR;
    const Tag* colorIn = findTag(tags, sl::kBufferTypeScalingInputColor);
    const Tag* colorOut = findTag(tags, sl::kBufferTypeScalingOutputColor);
    const Tag* depth = findTag(tags, sl::kBufferTypeDepth);
    const Tag* mvec = findTag(tags, sl::kBufferTypeMotionVectors);
    const Tag* exposure = findTag(tags, sl::kBufferTypeExposure);
    if (!colorIn || !colorOut || !depth || !mvec)
    {
        LOG_ERROR("sl-ngx: evaluate needs colour in and out, depth and motion vectors tagged");
        return Result::eErrorMissingInputParameter;
    }
    const Tag *albedo = nullptr, *specular = nullptr, *normals = nullptr, *roughness = nullptr;
    if (rr)
    {
        albedo = findTag(tags, sl::kBufferTypeAlbedo);
        specular = findTag(tags, sl::kBufferTypeSpecularAlbedo);
        const bool packed = vp.hasDlssd && vp.dlssd.normalRoughnessMode == sl::DLSSDNormalRoughnessMode::ePacked;
        normals = findTag(tags, packed ? sl::kBufferTypeNormalRoughness : sl::kBufferTypeNormals);
        roughness = packed ? nullptr : findTag(tags, sl::kBufferTypeRoughness);
        if (!albedo || !specular || !normals || (!packed && !roughness))
        {
            LOG_ERROR("sl-ngx: Ray Reconstruction needs albedo, specular albedo and normals (and roughness) tagged");
            return Result::eErrorMissingInputParameter;
        }
    }

    Shape s;
    Result r = shapeFor(feature, vp, c, colorIn, mvec, exposure != nullptr, s);
    if (r != Result::eOk)
        return r;
    auto cmd = reinterpret_cast<VkCommandBuffer>(cmdBuffer);
    if (!vp.handle || !(vp.shape == s))
    {
        r = create(cmd, vp, s);
        if (r != Result::eOk)
            return r;
    }
    vp.lastUsed = g.issued;

    const auto inExt = extentOf(*colorIn), outExt = extentOf(*colorOut), depthExt = extentOf(*depth),
               mvExt = extentOf(*mvec);
    NVSDK_NGX_Resource_VK color = ngxImage(colorIn->resource, false);
    NVSDK_NGX_Resource_VK output = ngxImage(colorOut->resource, true);
    NVSDK_NGX_Resource_VK depthRes = ngxImage(depth->resource, false);
    NVSDK_NGX_Resource_VK motion = ngxImage(mvec->resource, false);
    NVSDK_NGX_Resource_VK exposureRes{};
    if (exposure)
        exposureRes = ngxImage(exposure->resource, false);
    const float mvx = c.mvecScale.x * (float)inExt.width;
    const float mvy = c.mvecScale.y * (float)inExt.height;
    const int reset = c.reset == sl::Boolean::eTrue ? 1 : 0;
    if (vp.logNextEvaluate)
    {
        vp.logNextEvaluate = false;
        LOG_INFO("sl-ngx: first evaluate: jitter %.4f %.4f, motion scale %.1f %.1f, reset %d, render %ux%u at %u,%u, "
                 "output %ux%u, exposure %s",
            c.jitterOffset.x, c.jitterOffset.y, mvx, mvy, reset, inExt.width, inExt.height, inExt.left, inExt.top,
            outExt.width, outExt.height, exposure ? "tagged" : "auto");
    }

    NVSDK_NGX_Result nr;
    if (rr)
    {
        NVSDK_NGX_Resource_VK albedoRes = ngxImage(albedo->resource, false);
        NVSDK_NGX_Resource_VK specularRes = ngxImage(specular->resource, false);
        NVSDK_NGX_Resource_VK normalsRes = ngxImage(normals->resource, false);
        NVSDK_NGX_Resource_VK roughnessRes{};
        if (roughness)
            roughnessRes = ngxImage(roughness->resource, false);
        // NGX reads both matrices during the call
        sl::float4x4 worldToView = vp.dlssd.worldToCameraView;
        sl::float4x4 viewToClip = c.cameraViewToClip;
        const auto aExt = extentOf(*albedo), sExt = extentOf(*specular), nExt = extentOf(*normals);
        NVSDK_NGX_VK_DLSSD_Eval_Params p{};
        p.pInDiffuseAlbedo = &albedoRes;
        p.pInSpecularAlbedo = &specularRes;
        p.pInNormals = &normalsRes;
        p.pInRoughness = roughness ? &roughnessRes : nullptr;
        p.pInColor = &color;
        p.pInOutput = &output;
        p.pInDepth = &depthRes;
        p.pInMotionVectors = &motion;
        p.pInExposureTexture = exposure ? &exposureRes : nullptr;
        p.InJitterOffsetX = c.jitterOffset.x;
        p.InJitterOffsetY = c.jitterOffset.y;
        p.InRenderSubrectDimensions = {inExt.width, inExt.height};
        p.InReset = reset;
        p.InMVScaleX = mvx;
        p.InMVScaleY = mvy;
        p.InColorSubrectBase = {inExt.left, inExt.top};
        p.InDepthSubrectBase = {depthExt.left, depthExt.top};
        p.InMVSubrectBase = {mvExt.left, mvExt.top};
        p.InOutputSubrectBase = {outExt.left, outExt.top};
        p.InDiffuseAlbedoSubrectBase = {aExt.left, aExt.top};
        p.InSpecularAlbedoSubrectBase = {sExt.left, sExt.top};
        p.InNormalsSubrectBase = {nExt.left, nExt.top};
        if (roughness)
        {
            auto rExt = extentOf(*roughness);
            p.InRoughnessSubrectBase = {rExt.left, rExt.top};
        }
        p.InPreExposure = vp.dlssd.preExposure;
        p.InExposureScale = vp.dlssd.exposureScale;
        p.InIndicatorInvertXAxis = vp.dlssd.indicatorInvertAxisX == sl::Boolean::eTrue;
        p.InIndicatorInvertYAxis = vp.dlssd.indicatorInvertAxisY == sl::Boolean::eTrue;
        p.pInWorldToViewMatrix = &worldToView.row[0].x;
        p.pInViewToClipMatrix = &viewToClip.row[0].x;
        nr = NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, vp.handle, g.params, &p);
        // the shared parameters must not keep pointers to these locals
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX, nullptr);
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX, nullptr);
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, NVSDK_NGX_Parameter_DiffuseAlbedo, nullptr);
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, NVSDK_NGX_Parameter_SpecularAlbedo, nullptr);
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, NVSDK_NGX_Parameter_GBuffer_Normals, nullptr);
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, NVSDK_NGX_Parameter_GBuffer_Roughness, nullptr);
    }
    else
    {
        NVSDK_NGX_VK_DLSS_Eval_Params p{};
        p.Feature.pInColor = &color;
        p.Feature.pInOutput = &output;
        p.pInDepth = &depthRes;
        p.pInMotionVectors = &motion;
        p.pInExposureTexture = exposure ? &exposureRes : nullptr;
        p.InJitterOffsetX = c.jitterOffset.x;
        p.InJitterOffsetY = c.jitterOffset.y;
        p.InRenderSubrectDimensions = {inExt.width, inExt.height};
        p.InReset = reset;
        p.InMVScaleX = mvx;
        p.InMVScaleY = mvy;
        p.InColorSubrectBase = {inExt.left, inExt.top};
        p.InDepthSubrectBase = {depthExt.left, depthExt.top};
        p.InMVSubrectBase = {mvExt.left, mvExt.top};
        p.InOutputSubrectBase = {outExt.left, outExt.top};
        p.InPreExposure = vp.dlss.preExposure;
        p.InExposureScale = vp.dlss.exposureScale;
        p.InIndicatorInvertXAxis = vp.dlss.indicatorInvertAxisX == sl::Boolean::eTrue;
        p.InIndicatorInvertYAxis = vp.dlss.indicatorInvertAxisY == sl::Boolean::eTrue;
        nr = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, vp.handle, g.params, &p);
    }
    for (const char* name : {NVSDK_NGX_Parameter_Color, NVSDK_NGX_Parameter_Output, NVSDK_NGX_Parameter_Depth,
             NVSDK_NGX_Parameter_MotionVectors, NVSDK_NGX_Parameter_ExposureTexture})
        NVSDK_NGX_Parameter_SetVoidPointer(g.params, name, nullptr);
    if (NVSDK_NGX_FAILED(nr))
    {
        LOG_ERROR("sl-ngx: NGX evaluate failed: %#x", (unsigned)nr);
        return Result::eErrorNGXFailed;
    }
    return Result::eOk;
}

} // extern "C"
