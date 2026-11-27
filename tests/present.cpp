// Presenter test: drives the presenter (include/sl_ngx/present.h) and
// Reflex through libsl_ngx.so on a real NVIDIA Vulkan device and a window,
// under the validation layer (synchronisation and thread safety) when it is
// installed. Each rendered frame is two flat colours, the generated one and
// the real one, cleared on the application's queue.
//
//   tests/present [frames]
// Runs: generated and real frames; real frames alone; pause around a
// vkDeviceWaitIdle; a resize; a frame whose `ready` never comes; destroy
// with a frame queued; a presenter without VK_NV_low_latency2 (no latency
// reports); slShutdown with a presenter alive.
// Exit 0: every check held and the validation layer reported no error.
// Exit 2: no NVIDIA device or not supported here.
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "sl/sl.h"
#include "sl/sl_helpers_vk.h"
#include "sl/sl_pcl.h"
#include "sl/sl_reflex.h"
#include "sl_ngx/present.h"

#include <atomic>
#include <chrono>
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

namespace
{

int failures = 0;
std::atomic<int> validationErrors{0};

void check(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok)
        failures++;
}

void logMessage(sl::LogType type, const char* msg)
{
    if (type != sl::LogType::eInfo)
        std::printf("  sl: %s\n", msg);
}

VKAPI_ATTR VkBool32 VKAPI_CALL onValidation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        validationErrors++;
        std::printf("  validation: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

bool hasLayer(const char* name)
{
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> l(n);
    vkEnumerateInstanceLayerProperties(&n, l.data());
    for (auto& p : l)
        if (!std::strcmp(p.layerName, name))
            return true;
    return false;
}

struct Gpu
{
    VkInstance instance;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue queue;
    uint32_t presenterBits = 0;
    VkCommandPool pool;
};

struct FrameSet
{
    VkImage image[2];
    VkDeviceMemory memory[2];
    VkCommandBuffer cb;
    VkFence fence;
};

constexpr uint32_t W = 640, H = 360, kSets = 3;
constexpr VkImageLayout kHeld = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // what the presenter is told

uint32_t memoryType(const Gpu& g, uint32_t bits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g.pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    std::fprintf(stderr, "no memory type\n");
    std::exit(1);
}

void makeSet(const Gpu& g, FrameSet& s)
{
    for (int i = 0; i < 2; i++)
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.extent = {W, H, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VK(vkCreateImage(g.dev, &ci, nullptr, &s.image[i]));
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(g.dev, s.image[i], &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memoryType(g, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK(vkAllocateMemory(g.dev, &ai, nullptr, &s.memory[i]));
        VK(vkBindImageMemory(g.dev, s.image[i], s.memory[i], 0));
    }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK(vkAllocateCommandBuffers(g.dev, &ai, &s.cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK(vkCreateFence(g.dev, &fi, nullptr, &s.fence));
}

void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &b);
}

void waitValue(const Gpu& g, VkSemaphore s, uint64_t v)
{
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &s;
    wi.pValues = &v;
    VK(vkWaitSemaphores(g.dev, &wi, 5000000000ull));
}

uint64_t value(const Gpu& g, VkSemaphore s)
{
    uint64_t v = 0;
    VK(vkGetSemaphoreCounterValue(g.dev, s, &v));
    return v;
}

VkSemaphore timeline(const Gpu& g)
{
    VkSemaphoreTypeCreateInfo t{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    t.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &t;
    VkSemaphore s;
    VK(vkCreateSemaphore(g.dev, &ci, nullptr, &s));
    return s;
}

struct Api
{
    PFun_slReflexSetOptions* setOptions;
    PFun_slReflexSleep* sleep;
    PFun_slReflexGetState* getState;
    PFun_slPCLSetMarker* marker;
};

// One application frame: Reflex sleep and markers, the two images cleared
// on the application's queue (signalling `ready` = n), handed over.
struct Run
{
    const Gpu& g;
    const Api& api;
    FrameSet* sets;
    VkSemaphore ready, released;
    uint64_t n = 0;     // frames handed over or released by the caller so far
    uint64_t timeouts = 0;

    VkResult frame(SlngxPresenter* p, bool generated, VkSemaphore readyOverride = VK_NULL_HANDLE)
    {
        n++;
        // the set comes back once the presenter released the frame three before
        if (n > kSets)
            waitValue(g, released, n - kSets);
        FrameSet& s = sets[n % kSets];
        // and once our own clear of it is done (a frame released unread may be released before)
        VK(vkWaitForFences(g.dev, 1, &s.fence, VK_TRUE, 5000000000ull));
        VK(vkResetFences(g.dev, 1, &s.fence));
        sl::FrameToken* token;
        SL(slGetNewFrameToken(token, nullptr));
        SL(api.sleep(*token));
        SL(api.marker(sl::PCLMarker::eSimulationStart, *token));
        SL(api.marker(sl::PCLMarker::eSimulationEnd, *token));
        SL(api.marker(sl::PCLMarker::eRenderSubmitStart, *token));
        vkResetCommandBuffer(s.cb, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK(vkBeginCommandBuffer(s.cb, &bi));
        for (int i = 0; i < 2; i++)
        {
            barrier(s.cb, s.image[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            float t = (float)(n % 120) / 120.0f;
            VkClearColorValue c{{i ? t : 1.0f - t, i ? 0.2f : 0.6f, i ? 1.0f - t : t, 1.0f}};
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(s.cb, s.image[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
            barrier(s.cb, s.image[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kHeld);
        }
        VK(vkEndCommandBuffer(s.cb));
        VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ts.signalSemaphoreValueCount = readyOverride ? 0 : 1;
        ts.pSignalSemaphoreValues = &n;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &ts;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s.cb;
        si.signalSemaphoreCount = readyOverride ? 0 : 1;
        si.pSignalSemaphores = &ready;
        VK(vkQueueSubmit(g.queue, 1, &si, s.fence));
        SL(api.marker(sl::PCLMarker::eRenderSubmitEnd, *token));
        SL(api.marker(sl::PCLMarker::ePresentStart, *token));
        SlngxPresentFrame f{SLNGX_PRESENTER_VERSION};
        f.frame = (uint32_t)*token;
        if (generated)
            f.generated = {s.image[0], kHeld, W, H};
        f.real = {s.image[1], kHeld, W, H};
        f.ready = readyOverride ? readyOverride : ready;
        f.readyValue = readyOverride ? 1 : n;
        f.released = released;
        f.releasedValue = n;
        VkResult r = slngxPresent(p, &f, 2000000000ull);
        SL(api.marker(sl::PCLMarker::ePresentEnd, *token));
        if (r != VK_SUCCESS)
        {
            // not taken: the release is ours, after the frame before's
            timeouts++;
            waitValue(g, released, n - 1);
            vkQueueWaitIdle(g.queue);
            VkSemaphoreSignalInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            sig.semaphore = released;
            sig.value = n;
            VK(vkSignalSemaphore(g.dev, &sig));
        }
        return r;
    }
};

SlngxPresenterStats stats(SlngxPresenter* p)
{
    SlngxPresenterStats s{SLNGX_PRESENTER_VERSION};
    VK(slngxPresenterGetStats(p, &s));
    return s;
}

void print(const char* when, const SlngxPresenterStats& s)
{
    std::printf("  %s: accepted %llu skipped %llu presented %llu+%llu shown %llu+%llu timing %u lowLatency %u reports "
                "%u swapchains %u %ux%u copy %.3f ms interval %.2f ms half %.2f ms error %d\n",
        when, (unsigned long long)s.framesAccepted, (unsigned long long)s.framesSkipped,
        (unsigned long long)s.generatedPresented, (unsigned long long)s.realPresented,
        (unsigned long long)s.generatedShown, (unsigned long long)s.realShown, s.presentTiming, s.lowLatency,
        s.latencyReports, s.swapchainsMade, s.width, s.height, s.copyGpuMs, s.intervalMs, s.presentHalfMs,
        s.lastError);
}

SlngxPresenter* makePresenter(const Gpu& g, VkSurfaceKHR surface, uint32_t bits)
{
    SlngxPresenterDesc d{SLNGX_PRESENTER_VERSION};
    d.surface = surface;
    d.queueFamily = 0;
    d.queueIndex = 1;
    d.deviceBits = bits;
    d.width = W;
    d.height = H;
    d.format = VK_FORMAT_B8G8R8A8_UNORM;
    d.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    SlngxPresenter* p = nullptr;
    VK(slngxPresenterCreate(&d, &p));
    return p;
}

} // namespace

int main(int argc, char** argv)
{
    const int frames = argc > 1 ? std::atoi(argv[1]) : 240;
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 2;
    }
    SDL_Window* win = SDL_CreateWindow("sl-ngx presenter test [AUTOMATION - DO NOT TOUCH]", W, H,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win)
    {
        std::fprintf(stderr, "window: %s\n", SDL_GetError());
        return 2;
    }

    sl::Preferences pref{};
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = logMessage;
    pref.pathToLogsAndData = L"/tmp";
    pref.flags = sl::PreferenceFlags::eUseManualHooking;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0";
    pref.projectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";
    pref.renderAPI = sl::RenderAPI::eVulkan;
    SL(slInit(pref, sl::kSDKVersion));

    // the instance: SDL's, the presenter's (from the requirements, probed
    // on a first instance) and validation
    Gpu g{};
    std::vector<std::string> iext;
    Uint32 sdlN = 0;
    const char* const* sdlE = SDL_Vulkan_GetInstanceExtensions(&sdlN);
    for (Uint32 i = 0; i < sdlN; i++)
        iext.emplace_back(sdlE[i]);
    const bool validation = hasLayer("VK_LAYER_KHRONOS_validation");
    if (validation)
        iext.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    auto makeInstance = [&](bool withLayer) {
        std::vector<const char*> names;
        for (auto& e : iext)
            names.push_back(e.c_str());
        const char* layer = "VK_LAYER_KHRONOS_validation";
        VkBool32 on = VK_TRUE;
        VkLayerSettingEXT settings[] = {
            {layer, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on},
            {layer, "thread_safety", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on},
        };
        VkLayerSettingsCreateInfoEXT ls{VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT};
        ls.settingCount = 2;
        ls.pSettings = settings;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pNext = withLayer ? &ls : nullptr;
        ci.pApplicationInfo = &app;
        ci.enabledLayerCount = withLayer ? 1 : 0;
        ci.ppEnabledLayerNames = &layer;
        ci.enabledExtensionCount = (uint32_t)names.size();
        ci.ppEnabledExtensionNames = names.data();
        VkInstance inst;
        VK(vkCreateInstance(&ci, nullptr, &inst));
        return inst;
    };
    auto nvidia = [](VkInstance inst) {
        uint32_t n = 8;
        VkPhysicalDevice pds[8];
        vkEnumeratePhysicalDevices(inst, &n, pds);
        for (uint32_t i = 0; i < n; i++)
        {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(pds[i], &p);
            if (p.vendorID == 0x10de)
                return pds[i];
        }
        return (VkPhysicalDevice)VK_NULL_HANDLE;
    };
    VkInstance probe = makeInstance(false);
    VkPhysicalDevice probePd = nvidia(probe);
    if (!probePd)
    {
        std::fprintf(stderr, "no NVIDIA device\n");
        return 2;
    }
    sl::VulkanInfo vk{};
    vk.instance = probe;
    vk.physicalDevice = probePd;
    SL(slSetVulkanInfo(vk));
    sl::FeatureRequirements req{};
    SL(slGetFeatureRequirements(sl::kFeatureDLSS_G, req));
    std::vector<std::string> dext{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    for (uint32_t i = 0; i < req.vkNumInstanceExtensions; i++)
        iext.emplace_back(req.vkInstanceExtensions[i]);
    for (uint32_t i = 0; i < req.vkNumDeviceExtensions; i++)
        dext.emplace_back(req.vkDeviceExtensions[i]);
    // bufferDeviceAddress is core in 1.2 (as the KHR extension): the EXT one may not sit beside it
    for (auto it = dext.begin(); it != dext.end(); ++it)
        if (*it == VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
        {
            dext.erase(it);
            break;
        }
    check(req.vkNumGraphicsQueuesRequired == 1, "frame generation asks for one more graphics queue");
    vkDestroyInstance(probe, nullptr);

    g.instance = makeInstance(validation);
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validation)
    {
        auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g.instance,
            "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        mi.pfnUserCallback = onValidation;
        VK(create(g.instance, &mi, nullptr, &messenger));
    }
    std::printf("validation layer: %s\n", validation ? "on (sync, thread safety)" : "not installed");
    g.pd = nvidia(g.instance);
    VkSurfaceKHR surface;
    if (!SDL_Vulkan_CreateSurface(win, g.instance, nullptr, &surface))
    {
        std::fprintf(stderr, "surface: %s\n", SDL_GetError());
        return 2;
    }
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, qf.data());
    if (qf[0].queueCount < 2)
    {
        std::fprintf(stderr, "family 0 has one queue\n");
        return 2;
    }

    // the device: what the requirements named, each with its feature
    auto has = [&](const char* name) {
        for (auto& e : dext)
            if (e == name)
                return true;
        return false;
    };
    VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR fm1{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR};
    VkPhysicalDevicePresentTimingFeaturesEXT ft{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};
    VkPhysicalDevicePresentWait2FeaturesKHR fw2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR};
    VkPhysicalDevicePresentId2FeaturesKHR fi2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR};
    VkPhysicalDevicePresentIdFeaturesKHR fi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = f12.bufferDeviceAddress = f12.descriptorIndexing = VK_TRUE;
    void* chain = nullptr;
    auto link = [&](auto& s, const char* ext) {
        if (!has(ext))
            return false;
        s.pNext = chain;
        chain = &s;
        return true;
    };
    if (link(fm1, "VK_KHR_swapchain_maintenance1"))
        fm1.swapchainMaintenance1 = VK_TRUE;
    if (link(ft, "VK_EXT_present_timing"))
        ft.presentTiming = VK_TRUE, g.presenterBits |= SLNGX_DEVICE_PRESENT_TIMING;
    if (link(fw2, "VK_KHR_present_wait2"))
        fw2.presentWait2 = VK_TRUE, g.presenterBits |= SLNGX_DEVICE_PRESENT_WAIT2;
    if (link(fi2, "VK_KHR_present_id2"))
        fi2.presentId2 = VK_TRUE, g.presenterBits |= SLNGX_DEVICE_PRESENT_ID2;
    if (link(fi, VK_KHR_PRESENT_ID_EXTENSION_NAME))
        fi.presentId = VK_TRUE;
    if (has(VK_NV_LOW_LATENCY_2_EXTENSION_NAME) && has(VK_KHR_PRESENT_ID_EXTENSION_NAME))
        g.presenterBits |= SLNGX_DEVICE_LOW_LATENCY2;
    f12.pNext = chain;
    std::vector<const char*> dnames;
    for (auto& e : dext)
        dnames.push_back(e.c_str());
    float prio[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;
    qci.queueCount = 2;
    qci.pQueuePriorities = prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dnames.size();
    dci.ppEnabledExtensionNames = dnames.data();
    VK(vkCreateDevice(g.pd, &dci, nullptr, &g.dev));
    vkGetDeviceQueue(g.dev, 0, 0, &g.queue);
    std::printf("device extensions:");
    for (auto& e : dext)
        std::printf(" %s", e.c_str());
    std::printf("\npresenter bits %#x\n", g.presenterBits);

    vk.instance = g.instance;
    vk.physicalDevice = g.pd;
    vk.device = g.dev;
    SL(slSetVulkanInfo(vk));
    sl::AdapterInfo adapter{};
    if (slIsFeatureSupported(sl::kFeatureReflex, adapter) != sl::Result::eOk)
    {
        std::fprintf(stderr, "Reflex not supported here\n");
        return 2;
    }
    Api api{};
    SL(slGetFeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", (void*&)api.setOptions));
    SL(slGetFeatureFunction(sl::kFeatureReflex, "slReflexSleep", (void*&)api.sleep));
    SL(slGetFeatureFunction(sl::kFeatureReflex, "slReflexGetState", (void*&)api.getState));
    SL(slGetFeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", (void*&)api.marker));

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(g.dev, &pci, nullptr, &g.pool));
    FrameSet sets[kSets];
    for (auto& s : sets)
        makeSet(g, s);
    Run run{g, api, sets, timeline(g), timeline(g)};

    // Reflex off: the sleep must not wait
    {
        sl::ReflexOptions o{};
        SL(api.setOptions(o));
        SlngxPresenter* p = makePresenter(g, surface, g.presenterBits);
        sl::FrameToken* token;
        SL(slGetNewFrameToken(token, nullptr));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; i++)
            SL(api.sleep(*token));
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  10 sleeps with Reflex off: %.2f ms\n", ms);
        check(ms < 5, "Reflex off: slReflexSleep returns at once");
        slngxPresenterDestroy(p);
    }

    sl::ReflexOptions o{};
    o.mode = sl::ReflexMode::eLowLatency;
    SL(api.setOptions(o));
    SlngxPresenter* p = makePresenter(g, surface, g.presenterBits);

    // 1. generated and real
    for (int i = 0; i < frames; i++)
        run.frame(p, true);
    VK(slngxPresenterPause(p)); // every frame handed over presented (or skipped)
    auto s1 = stats(p);
    slngxPresenterResume(p);
    print("generated + real", s1);
    // 2. real alone
    for (int i = 0; i < 30; i++)
        run.frame(p, false);
    VK(slngxPresenterPause(p));
    auto s2 = stats(p);
    slngxPresenterResume(p);
    print("real alone", s2);
    check(s2.generatedPresented == s1.generatedPresented, "a frame without a generated image presents once");
    // 3. pause, a device-wide wait, resume
    VK(slngxPresenterPause(p));
    check(value(g, run.released) == run.n, "pause: every frame handed over is released");
    VK(vkDeviceWaitIdle(g.dev));
    slngxPresenterResume(p);
    for (int i = 0; i < 30; i++)
        run.frame(p, true);
    // 4. resize
    SDL_SetWindowSize(win, W * 3 / 2, H * 3 / 2);
    SDL_PumpEvents();
    slngxPresenterResize(p, W * 3 / 2, H * 3 / 2);
    for (int i = 0; i < 60; i++)
    {
        SDL_PumpEvents();
        run.frame(p, true);
    }
    auto s4 = stats(p);
    print("after resize", s4);
    check(s4.swapchainsMade >= 2, "resize: the swapchain is remade");
    // 5. a frame whose ready never comes, while the frame before is on the GPU
    VkSemaphore never = timeline(g);
    const uint64_t skippedBefore = stats(p).framesSkipped;
    run.frame(p, true);
    run.frame(p, true, never);
    for (int i = 0; i < 10; i++)
        run.frame(p, true);
    waitValue(g, run.released, run.n);
    auto s5 = stats(p);
    print("after a frame never ready", s5);
    check(s5.framesSkipped == skippedBefore + 1, "a frame never ready is skipped, and released");
    // the totals
    VK(slngxPresenterPause(p));
    auto all = stats(p);
    print("totals", all);
    check(all.framesAccepted + run.timeouts == run.n, "every frame was taken or released by the caller");
    check(all.framesAccepted == all.realPresented + all.framesSkipped, "every frame taken was presented or skipped");
    check(all.lastError == VK_SUCCESS, "no present failed");
    check(all.lowLatency == 1, "VK_NV_low_latency2 is on the swapchain");
    check(all.latencyReports > 0, "Reflex returns latency reports");
    check(all.copyGpuMs > 0, "the copies' GPU time is measured");
    {
        sl::ReflexState rs{};
        SL(api.getState(rs));
        uint32_t filled = 0;
        for (auto& r : rs.frameReport)
            filled += r.gpuRenderEndTime != 0;
        std::printf("  slReflexGetState: available %d, %u reports with GPU times\n", rs.lowLatencyAvailable, filled);
        check(rs.lowLatencyAvailable && filled > 0, "slReflexGetState fills frame reports");
    }
    slngxPresenterResume(p);
    // 6. destroy with frames queued
    run.frame(p, true);
    run.frame(p, true);
    slngxPresenterDestroy(p);
    check(value(g, run.released) == run.n, "destroy: every frame handed over is released");

    // 7. without VK_NV_low_latency2: no latency reports
    p = makePresenter(g, surface, g.presenterBits & ~SLNGX_DEVICE_LOW_LATENCY2);
    for (int i = 0; i < 30; i++)
        run.frame(p, true);
    VK(slngxPresenterPause(p));
    auto s7 = stats(p);
    print("without low latency", s7);
    check(s7.lowLatency == 0 && s7.latencyReports == 0, "without VK_NV_low_latency2: no reports");
    // the last one may be skipped: Pause releases a frame still queued
    check(s7.realPresented + s7.framesSkipped == 30 && s7.realPresented >= 29,
        "without VK_NV_low_latency2: every frame presented");
    slngxPresenterResume(p);

    // 8. slShutdown with a presenter alive: the handle is ended under us
    run.frame(p, true);
    SL(slShutdown());
    check(value(g, run.released) == run.n, "slShutdown: every frame handed over is released");
    SlngxPresenterStats dead{SLNGX_PRESENTER_VERSION};
    check(slngxPresenterGetStats(p, &dead) == VK_ERROR_DEVICE_LOST, "an ended handle answers VK_ERROR_DEVICE_LOST");
    slngxPresenterDestroy(p); // a no-op now

    VK(vkDeviceWaitIdle(g.dev));
    for (auto& s : sets)
        vkDestroyFence(g.dev, s.fence, nullptr);
    for (auto& s : sets)
        for (int i = 0; i < 2; i++)
        {
            vkDestroyImage(g.dev, s.image[i], nullptr);
            vkFreeMemory(g.dev, s.memory[i], nullptr);
        }
    vkDestroySemaphore(g.dev, never, nullptr);
    vkDestroySemaphore(g.dev, run.ready, nullptr);
    vkDestroySemaphore(g.dev, run.released, nullptr);
    vkDestroyCommandPool(g.dev, g.pool, nullptr);
    vkDestroyDevice(g.dev, nullptr);
    vkDestroySurfaceKHR(g.instance, surface, nullptr);
    if (messenger)
        ((PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g.instance, "vkDestroyDebugUtilsMessengerEXT"))(
            g.instance, messenger, nullptr);
    vkDestroyInstance(g.instance, nullptr);
    SDL_DestroyWindow(win);
    SDL_Quit();
    check(validationErrors == 0, "the validation layer reported no error");
    std::printf("%s: %d check(s) failed\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
