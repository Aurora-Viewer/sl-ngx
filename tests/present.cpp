// Presenter test: drives the presenter (include/sl_ngx/present.h) and
// Reflex through libsl_ngx.so on a real NVIDIA Vulkan device and a window,
// under the validation layer (synchronisation and thread safety) when it is
// installed. Each rendered frame is two flat colours, the generated one and
// the real one, cleared on the application's queue.
//
//   tests/present [frames]
//   tests/present swap
// Runs: Reflex off; the viewer's configuration (VK_NV_low_latency2 alone,
// images held in TRANSFER_SRC_OPTIMAL, real frames first, a frame number
// that repeats); a Reflex frame limit; the error codes of create and of
// frames that break the contract; generated and real frames; real
// frames alone; pause around a vkDeviceWaitIdle, and what a paused
// presenter answers; pause straight after a frame; a resize; a frame whose
// `ready` never comes; an unbounded hand-over wait; a hand-over with no
// wait while one waits; destroy with a frame queued; the shown counts
// without a next frame (where the window is on screen); a presenter
// without VK_NV_low_latency2 (no latency reports); slShutdown with a
// presenter alive.
// "swap" runs only this: a presenter on a device NGX did not start on (it
// is given a project id NGX refuses), ended by slSetVulkanInfo with another
// device (the devices are left as they are, so the validation layer's leak
// report is not reached).
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
#include <thread>
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
    VkImageLayout held = kHeld; // the layout the images are handed over in
    int64_t fixedFrame = -1;    // >= 0: the frame number given instead of the token's

    VkResult frame(SlngxPresenter* p, bool generated, VkSemaphore readyOverride = VK_NULL_HANDLE,
        uint64_t timeoutNs = 2000000000ull)
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
            barrier(s.cb, s.image[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, held);
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
        f.frame = fixedFrame >= 0 ? (uint32_t)fixedFrame : (uint32_t)*token;
        if (generated)
            f.generated = {s.image[0], held, W, H};
        f.real = {s.image[1], held, W, H};
        f.ready = readyOverride ? readyOverride : ready;
        f.readyValue = readyOverride ? 1 : n;
        f.released = released;
        f.releasedValue = n;
        VkResult r = slngxPresent(p, &f, timeoutNs);
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
    const bool swapMode = argc > 1 && !std::strcmp(argv[1], "swap");
    const int frames = argc > 1 && !swapMode ? std::atoi(argv[1]) : 240;
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
    // the layer's settings (sync and thread-safety validation) need its VK_EXT_layer_settings
    bool layerSettings = false;
    if (validation)
    {
        iext.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &n, nullptr);
        std::vector<VkExtensionProperties> e(n);
        vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &n, e.data());
        for (auto& x : e)
            layerSettings |= !std::strcmp(x.extensionName, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    }
    VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    mi.pfnUserCallback = onValidation;
    auto makeInstance = [&](bool withLayer) {
        std::vector<const char*> names;
        for (auto& e : iext)
            names.push_back(e.c_str());
        if (withLayer && layerSettings)
            names.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
        const char* layer = "VK_LAYER_KHRONOS_validation";
        VkBool32 on = VK_TRUE;
        VkLayerSettingEXT settings[] = {
            {layer, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on},
            {layer, "thread_safety", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on},
        };
        VkLayerSettingsCreateInfoEXT ls{VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT};
        ls.settingCount = 2;
        ls.pSettings = settings;
        // the messenger from the start, so that instance creation's own errors count
        mi.pNext = withLayer && layerSettings ? &ls : nullptr;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pNext = withLayer ? (const void*)&mi : nullptr;
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
        mi.pNext = nullptr;
        VK(create(g.instance, &mi, nullptr, &messenger));
    }
    std::printf("validation layer: %s\n",
        !validation ? "not installed" : layerSettings ? "on (sync, thread safety)" : "on (no VK_EXT_layer_settings: default checks)");
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

    if (swapMode)
    {
        // NGX starts on any device it is given (even one with the swapchain
        // alone, or another vendor's): a project id it refuses keeps it from
        // starting, the case of a device NGX did not start on
        slShutdown();
        pref.projectId = "not-a-project-id";
        SL(slInit(pref, sl::kSDKVersion));
        auto bareDevice = [&] {
            VkPhysicalDeviceVulkan12Features f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            f.timelineSemaphore = VK_TRUE;
            const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
            float prio[2] = {1.0f, 1.0f};
            VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            q.queueFamilyIndex = 0;
            q.queueCount = 2;
            q.pQueuePriorities = prio;
            VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            ci.pNext = &f;
            ci.queueCreateInfoCount = 1;
            ci.pQueueCreateInfos = &q;
            ci.enabledExtensionCount = 1;
            ci.ppEnabledExtensionNames = &ext;
            VkDevice d;
            VK(vkCreateDevice(g.pd, &ci, nullptr, &d));
            return d;
        };
        vk.instance = g.instance;
        vk.physicalDevice = g.pd;
        vk.device = bareDevice();
        const sl::Result started = slSetVulkanInfo(vk);
        std::printf("  slSetVulkanInfo with a project id NGX refuses: %d\n", (int)started);
        if (started == sl::Result::eOk)
        {
            std::fprintf(stderr, "NGX started anyway: the case cannot be made here\n");
            return 2;
        }
        SlngxPresenter* a = makePresenter(g, surface, 0);
        vk.device = bareDevice();
        slSetVulkanInfo(vk);
        SlngxPresenterStats st{SLNGX_PRESENTER_VERSION};
        check(slngxPresenterGetStats(a, &st) == VK_ERROR_DEVICE_LOST,
            "a new device ends the presenter where NGX did not start on the old one");
        slngxPresenterDestroy(a);
        check(validationErrors == 0, "the validation layer reported no error");
        std::printf("%s: %d check(s) failed\n", failures ? "FAILED" : "PASSED", failures);
        return failures ? 1 : 0;
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
        fm1.swapchainMaintenance1 = VK_TRUE, g.presenterBits |= SLNGX_DEVICE_SWAPCHAIN_MAINTENANCE1;
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

    // the viewer's configuration: VK_NV_low_latency2 alone, the images held
    // in TRANSFER_SRC_OPTIMAL, a fresh presenter whose first frames come
    // without a generated image, then a frame number that repeats (a frame
    // token the application could not get)
    if (g.presenterBits & SLNGX_DEVICE_LOW_LATENCY2)
    {
        const int errorsBefore = validationErrors;
        SlngxPresenter* v = makePresenter(g, surface, SLNGX_DEVICE_LOW_LATENCY2);
        run.held = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        for (int i = 0; i < 6; i++)
            run.frame(v, false);
        for (int i = 0; i < 30; i++)
            run.frame(v, true);
        run.fixedFrame = 0;
        for (int i = 0; i < 10; i++)
            run.frame(v, true);
        run.fixedFrame = -1;
        for (int i = 0; i < 10; i++)
            run.frame(v, true);
        VK(slngxPresenterPause(v));
        auto sv = stats(v);
        print("viewer configuration", sv);
        // the pause may release the last frame unshown
        check(sv.realPresented + sv.framesSkipped == 56 && sv.realPresented >= 55 &&
                sv.generatedPresented <= 50 && sv.generatedPresented + 6 >= sv.realPresented,
            "viewer configuration: every frame presented, the generated ones too");
        check(validationErrors == errorsBefore, "viewer configuration: no validation error (repeated frame numbers too)");
        slngxPresenterResume(v);
        slngxPresenterDestroy(v);
        run.held = kHeld;
    }

    // a Reflex frame limit is per rendered frame: two presents each
    {
        sl::ReflexOptions lim = o;
        // 10 rendered frames a second: slower than a FIFO pair even where the
        // compositor halves the rate (a workspace not on screen)
        lim.frameLimitUs = 100000;
        SL(api.setOptions(lim));
        SlngxPresenter* q = makePresenter(g, surface, g.presenterBits);
        for (int i = 0; i < 30; i++)
            run.frame(q, true);
        VK(slngxPresenterPause(q));
        auto sq = stats(q);
        print("frame limit 100 ms", sq);
        check(sq.intervalMs > 85 && sq.intervalMs < 130, "a 100 ms frame limit gives 100 ms rendered frames");
        check(sq.generatedPresented + 1 >= sq.realPresented, "a frame limit drops no generated frame");
        slngxPresenterResume(q);
        slngxPresenterDestroy(q);
        SL(api.setOptions(o));
    }

    // the contract's error codes
    {
        SlngxPresenterDesc d{SLNGX_PRESENTER_VERSION + 1};
        d.surface = surface;
        d.queueIndex = 1;
        d.width = W, d.height = H;
        d.format = VK_FORMAT_B8G8R8A8_UNORM;
        d.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        SlngxPresenter* x = nullptr;
        check(slngxPresenterCreate(&d, &x) == VK_ERROR_INCOMPATIBLE_DRIVER && !x,
            "create: another version answers VK_ERROR_INCOMPATIBLE_DRIVER");
        d.version = SLNGX_PRESENTER_VERSION;
        d.format = VK_FORMAT_R8_UNORM;
        check(slngxPresenterCreate(&d, &x) == VK_ERROR_FORMAT_NOT_SUPPORTED && !x,
            "create: a format the surface lacks answers VK_ERROR_FORMAT_NOT_SUPPORTED");
        d.format = VK_FORMAT_B8G8R8A8_UNORM;
        d.queueIndex = 99;
        check(slngxPresenterCreate(&d, &x) == VK_ERROR_INITIALIZATION_FAILED && !x,
            "create: no such queue answers VK_ERROR_INITIALIZATION_FAILED");
    }

    SlngxPresenter* p = makePresenter(g, surface, g.presenterBits);
    const uint64_t nAtP = run.n, timeoutsAtP = run.timeouts;
    {
        SlngxPresenterDesc d{SLNGX_PRESENTER_VERSION};
        d.surface = surface;
        d.queueIndex = 1;
        d.width = W, d.height = H;
        d.format = VK_FORMAT_B8G8R8A8_UNORM;
        d.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        SlngxPresenter* x = nullptr;
        check(slngxPresenterCreate(&d, &x) == VK_ERROR_INITIALIZATION_FAILED && !x,
            "create: a second presenter answers VK_ERROR_INITIALIZATION_FAILED");
        // refused before any Vulkan call: the image handle is never used
        SlngxPresentFrame f{SLNGX_PRESENTER_VERSION};
        f.real.image = reinterpret_cast<VkImage>(uintptr_t(1));
        f.real.layout = kHeld;
        f.real.width = 0, f.real.height = H;
        f.ready = run.ready;
        check(slngxPresent(p, &f, 0) == VK_ERROR_INITIALIZATION_FAILED, "a frame of size zero is refused");
        f.real.width = W;
        f.real.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(slngxPresent(p, &f, 0) == VK_ERROR_INITIALIZATION_FAILED, "a frame in an undefined layout is refused");
    }

    // 1. generated and real
    for (int i = 0; i < frames; i++)
        run.frame(p, true);
    VK(slngxPresenterPause(p)); // every frame handed over presented (or skipped)
    auto s1 = stats(p);
    slngxPresenterResume(p);
    print("generated + real", s1);
    check(s1.generatedPresented == s1.realPresented && s1.realPresented + s1.framesSkipped == (uint64_t)frames,
        "generated + real: both presented for every frame");
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
    {
        // paused: a valid frame is not taken (refused before any Vulkan
        // call, so the release stays ours, unsignalled); a PREINITIALIZED
        // image is not valid (a barrier cannot bring it back to that layout)
        FrameSet& s = sets[(run.n + 1) % kSets];
        SlngxPresentFrame f{SLNGX_PRESENTER_VERSION};
        f.frame = 1000000;
        f.real = {s.image[1], kHeld, W, H};
        f.ready = run.ready;
        f.readyValue = run.n;
        f.released = run.released;
        f.releasedValue = run.n + 1;
        check(slngxPresent(p, &f, 0) == VK_NOT_READY, "paused: a frame answers VK_NOT_READY");
        f.real.layout = VK_IMAGE_LAYOUT_PREINITIALIZED;
        check(slngxPresent(p, &f, 0) == VK_ERROR_INITIALIZATION_FAILED, "a frame in the preinitialized layout is refused");
        check(value(g, run.released) == run.n, "paused: a refused frame is not released");
    }
    VK(vkDeviceWaitIdle(g.dev));
    slngxPresenterResume(p);
    // 3b. pause straight after a frame: the presenter's thread is done with
    // a frame once it has submitted its copy, which may still run
    {
        int missed = 0;
        for (int i = 0; i < 100; i++)
        {
            run.frame(p, true);
            VK(slngxPresenterPause(p));
            missed += value(g, run.released) != run.n;
            slngxPresenterResume(p);
        }
        std::printf("  pause straight after a frame: %d of 100 not released at its return\n", missed);
        check(missed == 0, "pause straight after a frame: the frame is released when it returns");
    }
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
    // an unbounded wait to hand over (UINT64_MAX) waits, with a frame queued behind one never ready
    {
        const uint64_t timeoutsBefore = run.timeouts;
        run.frame(p, true, never);
        run.frame(p, true);
        VkResult r = run.frame(p, true, VK_NULL_HANDLE, UINT64_MAX);
        check(r == VK_SUCCESS && run.timeouts == timeoutsBefore, "a UINT64_MAX hand-over waits until taken");
        waitValue(g, run.released, run.n);
    }
    // the totals
    VK(slngxPresenterPause(p));
    auto all = stats(p);
    print("totals", all);
    check(all.framesAccepted + (run.timeouts - timeoutsAtP) == run.n - nAtP,
        "every frame was taken or released by the caller");
    check(run.timeouts == 0, "no hand-over timed out");
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
    // a hand-over that may not wait, while a frame waits for one never ready
    {
        run.frame(p, true, never);
        run.frame(p, true);
        const uint64_t timeoutsBefore = run.timeouts;
        VkResult r = run.frame(p, true, VK_NULL_HANDLE, 0);
        check(r == VK_TIMEOUT && run.timeouts == timeoutsBefore + 1,
            "a hand-over with timeout 0 while a frame waits answers VK_TIMEOUT");
        waitValue(g, run.released, run.n);
        check(run.frame(p, true) == VK_SUCCESS && stats(p).lastError == VK_SUCCESS,
            "after a VK_TIMEOUT the next frame is taken");
    }
    // 6. destroy with frames queued
    run.frame(p, true);
    run.frame(p, true);
    slngxPresenterDestroy(p);
    check(value(g, run.released) == run.n, "destroy: every frame handed over is released");

    // the shown counts come in with no frame after them (GetStats collects
    // them); the display shows the frames only where the window is on screen
    if (!all.presentTiming)
        std::printf("  shown counts: skipped, no present timing on the swapchain\n");
    else if (all.generatedShown + all.realShown == 0)
        std::printf("  shown counts: skipped, none in the long run (the window is not on screen)\n");
    else
    {
        p = makePresenter(g, surface, g.presenterBits);
        run.frame(p, true);
        VK(slngxPresenterPause(p));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto sc = stats(p);
        print("one frame, then stats", sc);
        check(sc.generatedShown + sc.realShown > 0, "present timing: the shown counts come in without a next frame");
        slngxPresenterResume(p);
        slngxPresenterDestroy(p);
    }

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
