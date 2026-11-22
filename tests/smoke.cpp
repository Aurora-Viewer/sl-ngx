// Smoke test: drives libsl_ngx.so through Streamline's public interface
// only, on a real NVIDIA Vulkan device. Upscales a flat colour and checks
// the output took it on.
//
//   tests/smoke <model folder> [rr|both]
// "both" runs Ray Reconstruction, then DLSS, on one device. Each feature
// runs at Quality, then Performance (a new feature inside the evaluate).
// Exit 0: every run wrote the input colour. Exit 2: not supported here.
#include <vulkan/vulkan.h>

#include "sl/sl.h"
#include "sl/sl_dlss.h"
#include "sl/sl_dlss_d.h"
#include "sl/sl_helpers_vk.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK(x)                                                                                                   \
    do                                                                                                          \
    {                                                                                                           \
        VkResult r_ = (x);                                                                                      \
        if (r_ != VK_SUCCESS)                                                                                   \
        {                                                                                                       \
            std::fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);                               \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    } while (0)
#define SL(x)                                                                                                   \
    do                                                                                                          \
    {                                                                                                           \
        sl::Result r_ = (x);                                                                                    \
        if (r_ != sl::Result::eOk)                                                                              \
        {                                                                                                       \
            std::fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, (int)r_);                          \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    } while (0)

static VkDevice dev;
static VkPhysicalDevice pd;

static uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    std::fprintf(stderr, "no memory type\n");
    std::exit(1);
}

struct Image
{
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    VkFormat format;
    uint32_t w, h;
    VkImageUsageFlags usage;
};

static Image makeImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    Image im{};
    im.format = format;
    im.w = w;
    im.h = h;
    im.usage = usage;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    VK(vkCreateImage(dev, &ci, nullptr, &im.image));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, im.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(dev, &ai, nullptr, &im.memory));
    VK(vkBindImageMemory(dev, im.image, im.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    VK(vkCreateImageView(dev, &vi, nullptr, &im.view));
    return im;
}

static void barrier(VkCommandBuffer cmd, const Image& im, VkImageAspectFlags aspect, VkImageLayout from,
    VkImageLayout to)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
        0, nullptr, 1, &b);
}

static sl::Resource resourceOf(const Image& im, VkImageLayout layout)
{
    sl::Resource r(sl::ResourceType::eTex2d, (void*)im.image, (void*)im.memory, (void*)im.view, (uint32_t)layout);
    r.width = im.w;
    r.height = im.h;
    r.nativeFormat = (uint32_t)im.format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.usage = (uint32_t)im.usage;
    return r;
}

static float half(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? INFINITY : std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}

static void logMessage(sl::LogType type, const char* msg)
{
    std::fprintf(stderr, "[sl %d] %s\n", (int)type, msg);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <model folder> [rr|both]\n", argv[0]);
        return 1;
    }
    std::vector<sl::Feature> features = {sl::kFeatureDLSS};
    if (argc > 2 && std::strcmp(argv[2], "rr") == 0)
        features = {sl::kFeatureDLSS_RR};
    else if (argc > 2 && std::strcmp(argv[2], "both") == 0)
        features = {sl::kFeatureDLSS_RR, sl::kFeatureDLSS};
    auto addAll = [](std::vector<const char*>& to, const char* const* names, uint32_t n) {
        for (uint32_t i = 0; i < n; i++)
        {
            bool have = false;
            for (auto t : to)
                have = have || std::strcmp(t, names[i]) == 0;
            if (!have)
                to.push_back(strdup(names[i]));
        }
    };
    std::string folder = argv[1];
    std::wstring wfolder(folder.begin(), folder.end());
    const wchar_t* paths[] = {wfolder.c_str()};

    sl::Preferences pref{};
    pref.logLevel = sl::LogLevel::eVerbose;
    pref.logMessageCallback = logMessage;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = L"/tmp";
    pref.flags = sl::PreferenceFlags::eUseManualHooking;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0";
    pref.projectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";
    pref.renderAPI = sl::RenderAPI::eVulkan;
    SL(slInit(pref, sl::kSDKVersion));

    // the instance, with what the feature names
    sl::FeatureRequirements req{};
    std::vector<const char*> instExts;
    for (auto f : features)
    {
        SL(slGetFeatureRequirements(f, req));
        addAll(instExts, req.vkInstanceExtensions, req.vkNumInstanceExtensions);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instExts.size();
    ici.ppEnabledExtensionNames = instExts.data();
    VkInstance instance;
    VK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(instance, &n, pds.data());
    pd = VK_NULL_HANDLE;
    for (auto p : pds)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(p, &props);
        if (props.vendorID == 0x10de)
        {
            pd = p;
            std::printf("device: %s\n", props.deviceName);
            break;
        }
    }
    if (!pd)
    {
        std::fprintf(stderr, "no NVIDIA device\n");
        return 2;
    }

    // the device, with what NGX names for it
    sl::VulkanInfo vk{};
    vk.instance = instance;
    vk.physicalDevice = pd;
    SL(slSetVulkanInfo(vk));
    std::vector<const char*> devExts;
    for (auto f : features)
    {
        SL(slGetFeatureRequirements(f, req));
        addAll(devExts, req.vkDeviceExtensions, req.vkNumDeviceExtensions);
    }
    for (auto e : devExts)
        std::printf("device extension: %s\n", e);
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());
    uint32_t family = 0;
    while (family < qfCount && !(qfs[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        family++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)devExts.size();
    dci.ppEnabledExtensionNames = devExts.data();
    VK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);
    vk.device = dev;
    vk.graphicsQueueFamily = family;
    SL(slSetVulkanInfo(vk));

    int bad = 0;
    for (const sl::Feature feature : features)
    {
    const bool rr = feature == sl::kFeatureDLSS_RR;
    std::printf("--- %s\n", rr ? "Ray Reconstruction" : "DLSS");
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = pd;
    sl::Result supported = slIsFeatureSupported(feature, adapter);
    std::printf("supported: %d\n", (int)supported);
    if (supported != sl::Result::eOk)
    {
        slShutdown();
        return 2;
    }

    const uint32_t outW = 1920, outH = 1080;
    sl::ViewportHandle viewport(7);
    uint32_t renderW, renderH;
    if (rr)
    {
        sl::DLSSDOptions o{};
        o.mode = sl::DLSSMode::eMaxQuality;
        o.outputWidth = outW;
        o.outputHeight = outH;
        o.colorBuffersHDR = sl::Boolean::eTrue;
        o.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
        sl::DLSSDOptimalSettings s{};
        SL(slDLSSDGetOptimalSettings(o, s));
        SL(slDLSSDSetOptions(viewport, o));
        renderW = s.optimalRenderWidth;
        renderH = s.optimalRenderHeight;
    }
    else
    {
        sl::DLSSOptions o{};
        o.mode = sl::DLSSMode::eMaxQuality;
        o.outputWidth = outW;
        o.outputHeight = outH;
        o.colorBuffersHDR = sl::Boolean::eTrue;
        sl::DLSSOptimalSettings s{};
        SL(slDLSSGetOptimalSettings(o, s));
        SL(slDLSSSetOptions(viewport, o));
        renderW = s.optimalRenderWidth;
        renderH = s.optimalRenderHeight;
    }
    std::printf("render %ux%u -> %ux%u\n", renderW, renderH, outW, outH);

    const VkImageUsageFlags in = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    Image color = makeImage(renderW, renderH, VK_FORMAT_R16G16B16A16_SFLOAT, in, VK_IMAGE_ASPECT_COLOR_BIT);
    Image depth = makeImage(renderW, renderH, VK_FORMAT_D32_SFLOAT, in, VK_IMAGE_ASPECT_DEPTH_BIT);
    Image motion = makeImage(renderW, renderH, VK_FORMAT_R16G16_SFLOAT, in, VK_IMAGE_ASPECT_COLOR_BIT);
    Image albedo = makeImage(renderW, renderH, VK_FORMAT_R8G8B8A8_UNORM, in, VK_IMAGE_ASPECT_COLOR_BIT);
    Image specular = makeImage(renderW, renderH, VK_FORMAT_R8G8B8A8_UNORM, in, VK_IMAGE_ASPECT_COLOR_BIT);
    Image normals = makeImage(renderW, renderH, VK_FORMAT_R16G16B16A16_SFLOAT, in, VK_IMAGE_ASPECT_COLOR_BIT);
    Image output = makeImage(outW, outH, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = (VkDeviceSize)outW * outH * 8;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback;
    VK(vkCreateBuffer(dev, &bci, nullptr, &readback));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, readback, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex =
        memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory readbackMem;
    VK(vkAllocateMemory(dev, &ai, nullptr, &readbackMem));
    VK(vkBindBufferMemory(dev, readback, readbackMem, 0));

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family;
    VkCommandPool pool;
    VK(vkCreateCommandPool(dev, &pci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK(vkCreateFence(dev, &fci, nullptr, &fence));
    auto submit = [&] {
        VK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK(vkQueueSubmit(queue, 1, &si, fence));
        VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
        VK(vkResetFences(dev, 1, &fence));
    };
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    // inputs: a flat colour, far depth, no motion; output cleared to zero
    const float want[3] = {0.25f, 0.5f, 0.75f};
    VK(vkBeginCommandBuffer(cmd, &begin));
    const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT;
    auto clear = [&](Image& im, VkClearColorValue v) {
        barrier(cmd, im, C, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange range{C, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    };
    clear(color, {{want[0], want[1], want[2], 1.0f}});
    clear(motion, {{0, 0, 0, 0}});
    clear(albedo, {{1, 1, 1, 1}});
    clear(specular, {{0.04f, 0.04f, 0.04f, 1}});
    clear(normals, {{0, 0, 1, 0.5f}});
    clear(output, {{0, 0, 0, 0}});
    barrier(cmd, depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearDepthStencilValue far{1.0f, 0};
    VkImageSubresourceRange drange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vkCmdClearDepthStencilImage(cmd, depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &far, 1, &drange);
    for (Image* im : {&color, &motion, &albedo, &specular, &normals})
        barrier(cmd, *im, C, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    barrier(cmd, depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    barrier(cmd, output, C, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    submit();

    const auto RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sl::Resource rColor = resourceOf(color, RO), rDepth = resourceOf(depth, RO), rMotion = resourceOf(motion, RO),
                 rOut = resourceOf(output, VK_IMAGE_LAYOUT_GENERAL), rAlbedo = resourceOf(albedo, RO),
                 rSpec = resourceOf(specular, RO), rNormals = resourceOf(normals, RO);
    const auto L = sl::ResourceLifecycle::eValidUntilEvaluate;
    sl::Extent inExt{0, 0, renderW, renderH}, outExt{0, 0, outW, outH};
    std::vector<sl::ResourceTag> tags = {
        sl::ResourceTag(&rColor, sl::kBufferTypeScalingInputColor, L, &inExt),
        sl::ResourceTag(&rDepth, sl::kBufferTypeDepth, L, &inExt),
        sl::ResourceTag(&rMotion, sl::kBufferTypeMotionVectors, L, &inExt),
        sl::ResourceTag(&rOut, sl::kBufferTypeScalingOutputColor, L, &outExt),
    };
    if (rr)
    {
        tags.push_back(sl::ResourceTag(&rAlbedo, sl::kBufferTypeAlbedo, L, &inExt));
        tags.push_back(sl::ResourceTag(&rSpec, sl::kBufferTypeSpecularAlbedo, L, &inExt));
        tags.push_back(sl::ResourceTag(&rNormals, sl::kBufferTypeNormalRoughness, L, &inExt));
    }

    sl::Constants c{};
    sl::float4x4 identity;
    identity.setRow(0, {1, 0, 0, 0});
    identity.setRow(1, {0, 1, 0, 0});
    identity.setRow(2, {0, 0, 1, 0});
    identity.setRow(3, {0, 0, 0, 1});
    c.cameraViewToClip = c.clipToCameraView = c.clipToLensClip = c.clipToPrevClip = c.prevClipToClip = identity;
    c.jitterOffset = {0, 0};
    c.mvecScale = {1, 1};
    c.cameraPinholeOffset = {0, 0};
    c.cameraPos = {0, 0, 0};
    c.cameraUp = {0, 1, 0};
    c.cameraRight = {1, 0, 0};
    c.cameraFwd = {0, 0, -1};
    c.cameraNear = 0.1f;
    c.cameraFar = 1000.0f;
    c.cameraFOV = 1.0f;
    c.cameraAspectRatio = (float)outW / outH;
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    if (rr)
    {
        sl::DLSSDOptions o{};
        o.mode = sl::DLSSMode::eMaxQuality;
        o.outputWidth = outW;
        o.outputHeight = outH;
        o.colorBuffersHDR = sl::Boolean::eTrue;
        o.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
        o.worldToCameraView = o.cameraViewToWorld = identity;
        SL(slDLSSDSetOptions(viewport, o));
    }

    for (uint32_t frame = 0; frame < 8; frame++)
    {
        if (frame == 4)
        {
            // Performance: a smaller render size inside the same images, so
            // the evaluate itself has to make a new feature
            uint32_t w, h;
            if (rr)
            {
                sl::DLSSDOptions o{};
                o.mode = sl::DLSSMode::eMaxPerformance;
                o.outputWidth = outW;
                o.outputHeight = outH;
                o.colorBuffersHDR = sl::Boolean::eTrue;
                o.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
                o.worldToCameraView = o.cameraViewToWorld = identity;
                sl::DLSSDOptimalSettings st{};
                SL(slDLSSDGetOptimalSettings(o, st));
                SL(slDLSSDSetOptions(viewport, o));
                w = st.optimalRenderWidth;
                h = st.optimalRenderHeight;
            }
            else
            {
                sl::DLSSOptions o{};
                o.mode = sl::DLSSMode::eMaxPerformance;
                o.outputWidth = outW;
                o.outputHeight = outH;
                o.colorBuffersHDR = sl::Boolean::eTrue;
                sl::DLSSOptimalSettings st{};
                SL(slDLSSGetOptimalSettings(o, st));
                SL(slDLSSSetOptions(viewport, o));
                w = st.optimalRenderWidth;
                h = st.optimalRenderHeight;
            }
            std::printf("performance: render %ux%u\n", w, h);
            inExt = {0, 0, w, h};
            for (auto& t : tags)
                if (t.type != sl::kBufferTypeScalingOutputColor)
                    t.extent = inExt;
        }
        sl::FrameToken* token = nullptr;
        SL(slGetNewFrameToken(token, nullptr));
        c.reset = frame == 0 ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        SL(slSetConstants(c, *token, viewport));
        VK(vkBeginCommandBuffer(cmd, &begin));
        if (frame == 4)
        {
            // so only the new feature's evaluates can put the colour back
            barrier(cmd, output, C, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue zero{{0, 0, 0, 0}};
            VkImageSubresourceRange range{C, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, output.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            barrier(cmd, output, C, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        SL(slSetTagForFrame(*token, viewport, tags.data(), (uint32_t)tags.size(), (sl::CommandBuffer*)cmd));
        if (frame == 0)
            SL(slAllocateResources((sl::CommandBuffer*)cmd, feature, viewport));
        const sl::BaseStructure* inputs[] = {&viewport};
        SL(slEvaluateFeature(feature, *token, inputs, 1, (sl::CommandBuffer*)cmd));
        submit();
    }

    VK(vkBeginCommandBuffer(cmd, &begin));
    barrier(cmd, output, C, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {C, 0, 0, 1};
    copy.imageExtent = {outW, outH, 1};
    vkCmdCopyImageToBuffer(cmd, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
    submit();

    uint64_t vram = 0;
    if (rr)
    {
        sl::DLSSDState st{};
        SL(slDLSSDGetState(viewport, st));
        vram = st.estimatedVRAMUsageInBytes;
    }
    else
    {
        sl::DLSSState st{};
        SL(slDLSSGetState(viewport, st));
        vram = st.estimatedVRAMUsageInBytes;
    }
    std::printf("estimated VRAM: %llu bytes\n", (unsigned long long)vram);

    uint16_t* px = nullptr;
    VK(vkMapMemory(dev, readbackMem, 0, VK_WHOLE_SIZE, 0, (void**)&px));
    const uint32_t samples[][2] = {{outW / 2, outH / 2}, {10, 10}, {outW - 10, outH - 10}, {outW / 3, 2 * outH / 3}};
    for (auto& s : samples)
    {
        const uint16_t* p = px + ((size_t)s[1] * outW + s[0]) * 4;
        float r = half(p[0]), g = half(p[1]), b = half(p[2]);
        std::printf("pixel %u,%u: %.3f %.3f %.3f\n", s[0], s[1], r, g, b);
        // Ray Reconstruction re-lights from the guides, so only DLSS is held to the input colour
        const float tol = rr ? 0.6f : 0.05f;
        if (std::fabs(r - want[0]) > tol || std::fabs(g - want[1]) > tol || std::fabs(b - want[2]) > tol ||
            (r == 0 && g == 0 && b == 0))
            bad++;
    }
    vkUnmapMemory(dev, readbackMem);

    SL(slFreeResources(feature, viewport));
    }
    SL(slShutdown());
    std::printf(bad ? "FAIL: output did not take the input colour\n" : "PASS\n");
    return bad ? 1 : 0;
}
