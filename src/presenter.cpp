// SPDX-License-Identifier: MIT
//
// sl-ngx's presenter (include/sl_ngx/present.h) and Reflex on its swapchain.
//
// The presenter owns a swapchain on a surface the application gave it and a
// thread that presents on a queue of its own. Per rendered frame it waits on
// the CPU for the application's `ready` timeline (never on the GPU: a wait
// on another queue's work would hold the presenter's queue, and with it any
// vkDeviceWaitIdle, behind the application), copies the generated frame into
// a swapchain image and presents it, waits half the measured frame interval,
// then copies and presents the rendered frame and signals `released`.
//
// Present ids: the generated frame of frame token f is 2f+1, the rendered
// one 2f+2, and Reflex's markers for frame f use 2f+2, as Streamline's
// DLSS-G does with the frame the application presents.

#include "internal.h"
#include "sl/sl_pcl.h"
#include "sl_ngx/present.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#define SL_NGX_EXPORT __declspec(dllexport)
#else
#define SL_NGX_EXPORT __attribute__((visibility("default")))
#endif

using sl::Result;

namespace
{

constexpr uint32_t kSlots = 3;
constexpr uint64_t kReadyTimeoutNs = 1000000000ull;
constexpr uint64_t kSleepTimeoutNs = 200000000ull;
using Clock = std::chrono::steady_clock;

#define PLOG_INFO(...) slngx::log(sl::LogType::eInfo, __VA_ARGS__)
#define PLOG_WARN(...) slngx::log(sl::LogType::eWarn, __VA_ARGS__)
#define PLOG_ERROR(...) slngx::log(sl::LogType::eError, __VA_ARGS__)

struct Slot
{
    VkCommandBuffer cb[2]{};
    VkSemaphore acquired[2]{};
    VkFence fence = VK_NULL_HANDLE;
    bool timed[2]{};
};

// Reflex's options, kept for every swapchain the presenter makes.
std::mutex gReflexLock;
sl::ReflexOptions gReflex{};

} // namespace

struct SlngxPresenter
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    SlngxPresenterDesc desc{};
    bool lowLatency = false, id2 = false, timing = false;

    PFN_vkSetLatencySleepModeNV setSleepMode = nullptr;
    PFN_vkLatencySleepNV latencySleep = nullptr;
    PFN_vkSetLatencyMarkerNV setMarker = nullptr;
    PFN_vkGetLatencyTimingsNV getTimings = nullptr;
    PFN_vkQueueNotifyOutOfBandNV notifyOob = nullptr;
    PFN_vkSetSwapchainPresentTimingQueueSizeEXT setTimingQueue = nullptr;
    PFN_vkGetPastPresentationTimingEXT pastTiming = nullptr;
    PFN_vkGetSwapchainTimeDomainPropertiesEXT timeDomains = nullptr;

    // The swapchain: made and destroyed on the presenter's thread (or by
    // Destroy once it has stopped); Reflex reads it under swapLock.
    std::mutex swapLock;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> copied;
    uint32_t width = 0, height = 0;
    bool swapTiming = false;
    uint64_t timeDomainId = 0;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    float tsPeriodNs = 0;
    Slot slots[kSlots];
    uint32_t nextSlot = 0;

    VkSemaphore sleepSem = VK_NULL_HANDLE;
    uint64_t sleepValue = 0;

    // handed-over frames and the thread's state
    std::mutex m;
    std::condition_variable cv;
    std::deque<SlngxPresentFrame> queued;
    bool busy = false, paused = false, stop = false, lost = false, remake = false;
    uint32_t wantW = 0, wantH = 0;
    SlngxPresenterStats stats{};
    double intervalAvgNs = 0, copyAvgMs = 0, halfAvgNs = 0;
    Clock::time_point lastReady{};
    bool haveLastReady = false;

    std::thread thread;
};

namespace
{

std::mutex gp;                       // guards gPresenter and gSleepers
std::condition_variable gpCv;
SlngxPresenter* gPresenter = nullptr;
int gSleepers = 0;

// Whether the handle is the running presenter: slShutdown and a new device
// in slSetVulkanInfo end it under the application (call those from the
// thread that calls the presenter).
bool live(SlngxPresenter* p)
{
    std::lock_guard<std::mutex> l(gp);
    return p && p == gPresenter;
}

template <class T> T devFn(VkDevice dev, const char* name)
{
    return reinterpret_cast<T>(vkGetDeviceProcAddr(dev, name));
}

void applySleepMode(SlngxPresenter* p)
{
    if (!p->lowLatency || !p->swapchain || !p->setSleepMode)
        return;
    sl::ReflexOptions o;
    {
        std::lock_guard<std::mutex> l(gReflexLock);
        o = gReflex;
    }
    VkLatencySleepModeInfoNV mode{VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV};
    mode.lowLatencyMode = o.mode != sl::ReflexMode::eOff;
    mode.lowLatencyBoost = o.mode == sl::ReflexMode::eLowLatencyWithBoost;
    mode.minimumIntervalUs = o.frameLimitUs;
    VkResult r = p->setSleepMode(p->dev, p->swapchain, &mode);
    if (r != VK_SUCCESS)
        PLOG_WARN("sl-ngx presenter: vkSetLatencySleepModeNV -> %d", r);
}

void hostRelease(SlngxPresenter* p, const SlngxPresentFrame& f)
{
    if (!f.released)
        return;
    VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    si.semaphore = f.released;
    si.value = f.releasedValue;
    VkResult r = vkSignalSemaphore(p->dev, &si);
    if (r != VK_SUCCESS)
        PLOG_WARN("sl-ngx presenter: releasing frame %u from the host -> %d", f.frame, r);
}

void destroySwapchainParts(SlngxPresenter* p, VkSwapchainKHR sc, std::vector<VkSemaphore>& sems)
{
    for (auto s : sems)
        vkDestroySemaphore(p->dev, s, nullptr);
    sems.clear();
    if (sc)
        vkDestroySwapchainKHR(p->dev, sc, nullptr);
}

// Makes (or remakes) the swapchain. The presenter's queue is idle.
VkResult makeSwapchain(SlngxPresenter* p)
{
    uint32_t wantW, wantH;
    {
        std::lock_guard<std::mutex> l(p->m);
        wantW = p->wantW, wantH = p->wantH;
        p->remake = false;
    }
    VkSurfaceCapabilitiesKHR caps{};
    VkResult r = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p->pd, p->desc.surface, &caps);
    if (r != VK_SUCCESS)
        return r;
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xffffffffu)
    {
        extent.width = std::clamp(wantW, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(wantH, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0)
        return VK_NOT_READY; // minimised: no swapchain until it has a size
    uint32_t count = std::max(p->desc.minImageCount ? p->desc.minImageCount : 3u, caps.minImageCount);
    if (caps.maxImageCount)
        count = std::min(count, caps.maxImageCount);

    VkSwapchainLatencyCreateInfoNV latency{VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV};
    latency.latencyModeEnable = VK_TRUE;
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.pNext = p->lowLatency ? &latency : nullptr;
    if (p->id2)
        ci.flags |= VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
    if (p->desc.deviceBits & SLNGX_DEVICE_PRESENT_WAIT2)
        ci.flags |= VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR;
    if (p->timing)
        ci.flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT;
    ci.surface = p->desc.surface;
    ci.minImageCount = count;
    ci.imageFormat = p->desc.format;
    ci.imageColorSpace = p->desc.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
        ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    // FIFO: the only mode where every present reaches the display in order
    // without tearing; the pacer's half-frame wait spaces the pair
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = p->swapchain;
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    {
        // the old swapchain is taken for itself (oldSwapchain is externally synchronised)
        std::lock_guard<std::mutex> l(p->swapLock);
        r = vkCreateSwapchainKHR(p->dev, &ci, nullptr, &sc);
    }
    if (r != VK_SUCCESS)
        return r;
    uint32_t n = 0;
    vkGetSwapchainImagesKHR(p->dev, sc, &n, nullptr);
    std::vector<VkImage> images(n);
    vkGetSwapchainImagesKHR(p->dev, sc, &n, images.data());
    std::vector<VkSemaphore> copied(n);
    for (auto& s : copied)
    {
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(p->dev, &si, nullptr, &s);
    }
    VkSwapchainKHR old;
    std::vector<VkSemaphore> oldSems;
    {
        std::lock_guard<std::mutex> l(p->swapLock);
        old = p->swapchain;
        oldSems.swap(p->copied);
        p->swapchain = sc;
        p->images = std::move(images);
        p->copied = std::move(copied);
        p->width = extent.width, p->height = extent.height;
        p->swapTiming = false;
        if (p->timing && p->setTimingQueue)
            p->swapTiming = p->setTimingQueue(p->dev, sc, 64) == VK_SUCCESS;
        if (p->swapTiming)
        {
            // a present's timing request names one of the swapchain's time domains
            VkTimeDomainKHR domains[8];
            uint64_t ids[8];
            VkSwapchainTimeDomainPropertiesEXT dp{VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT};
            dp.timeDomainCount = 8;
            dp.pTimeDomains = domains;
            dp.pTimeDomainIds = ids;
            uint64_t counter = 0;
            VkResult d = p->timeDomains(p->dev, sc, &dp, &counter);
            p->swapTiming = (d == VK_SUCCESS || d == VK_INCOMPLETE) && dp.timeDomainCount > 0;
            if (p->swapTiming)
                p->timeDomainId = ids[0];
        }
        applySleepMode(p);
    }
    destroySwapchainParts(p, old, oldSems);
    std::lock_guard<std::mutex> l(p->m);
    p->stats.swapchainsMade++;
    p->stats.width = extent.width, p->stats.height = extent.height;
    p->stats.presentTiming = p->swapTiming;
    PLOG_INFO("sl-ngx presenter: swapchain %ux%u, %u images, Reflex %s, present timing %s", extent.width,
        extent.height, n, p->lowLatency ? "on" : "off", p->swapTiming ? "on" : "off");
    return VK_SUCCESS;
}

void readTimestamps(SlngxPresenter* p, uint32_t slot)
{
    Slot& s = p->slots[slot];
    if (!p->queries || !(s.timed[0] || s.timed[1]))
        return;
    uint64_t v[4][2]{};
    VkResult r = vkGetQueryPoolResults(p->dev, p->queries, slot * 4, 4, sizeof(v), v, sizeof(v[0]),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY)
        return;
    double ms = 0;
    for (int h = 0; h < 2; h++)
        if (s.timed[h] && v[h * 2][1] && v[h * 2 + 1][1] && v[h * 2 + 1][0] >= v[h * 2][0])
            ms += (double)(v[h * 2 + 1][0] - v[h * 2][0]) * p->tsPeriodNs / 1e6;
    std::lock_guard<std::mutex> l(p->m);
    p->copyAvgMs = p->copyAvgMs ? p->copyAvgMs * 0.9 + ms * 0.1 : ms;
    p->stats.copyGpuMs = (float)p->copyAvgMs;
    s.timed[0] = s.timed[1] = false;
}

void countShown(SlngxPresenter* p)
{
    if (!p->swapTiming || !p->pastTiming)
        return;
    VkPastPresentationTimingEXT pt[64];
    VkPresentStageTimeEXT st[64];
    for (int round = 0; round < 4; round++)
    {
        for (int i = 0; i < 64; i++)
        {
            pt[i] = VkPastPresentationTimingEXT{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT};
            pt[i].presentStageCount = 1;
            pt[i].pPresentStages = &st[i];
        }
        VkPastPresentationTimingInfoEXT info{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT};
        info.flags = VK_PAST_PRESENTATION_TIMING_ALLOW_OUT_OF_ORDER_RESULTS_BIT_EXT;
        info.swapchain = p->swapchain;
        VkPastPresentationTimingPropertiesEXT props{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT};
        props.presentationTimingCount = 64;
        props.pPresentationTimings = pt;
        VkResult r = p->pastTiming(p->dev, &info, &props);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE)
            return;
        std::lock_guard<std::mutex> l(p->m);
        for (uint32_t i = 0; i < props.presentationTimingCount; i++)
            if (pt[i].reportComplete && pt[i].presentStageCount && st[i].time)
                (pt[i].presentId & 1 ? p->stats.generatedShown : p->stats.realShown)++;
        if (r == VK_SUCCESS)
            return;
    }
}

void marker(SlngxPresenter* p, VkLatencyMarkerNV m, uint64_t id)
{
    if (!p->lowLatency || !p->setMarker)
        return;
    VkSetLatencyMarkerInfoNV info{VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV};
    info.presentID = id;
    info.marker = m;
    p->setMarker(p->dev, p->swapchain, &info);
}

void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess,
    VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// vkAcquireNextImageKHR and vkQueuePresentKHR take the swapchain for
// themselves (externally synchronised), and Reflex's calls on it come from
// the application's thread: both run under swapLock, the acquire polled so
// that the lock is never held while it waits for an image.
VkResult acquire(SlngxPresenter* p, VkSemaphore semaphore, uint32_t& index)
{
    const auto until = Clock::now() + std::chrono::nanoseconds(kReadyTimeoutNs);
    for (;;)
    {
        VkResult r;
        {
            std::lock_guard<std::mutex> l(p->swapLock);
            r = vkAcquireNextImageKHR(p->dev, p->swapchain, 0, semaphore, VK_NULL_HANDLE, &index);
        }
        if (r != VK_NOT_READY && r != VK_TIMEOUT)
            return r;
        if (Clock::now() >= until)
            return VK_TIMEOUT;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

// Copies one image into a swapchain image and presents it. `last`: the
// frame's final submit, which signals `released` and the slot's fence.
VkResult presentOne(SlngxPresenter* p, uint32_t slot, int h, const SlngxPresentImage& src, uint64_t id,
    uint64_t markerId, const SlngxPresentFrame* last)
{
    Slot& s = p->slots[slot];
    uint32_t index = 0;
    VkResult r = VK_ERROR_OUT_OF_DATE_KHR;
    for (int attempt = 0; attempt < 2 && r == VK_ERROR_OUT_OF_DATE_KHR; attempt++)
    {
        if (attempt || !p->swapchain)
        {
            vkQueueWaitIdle(p->queue);
            VkResult m = makeSwapchain(p);
            if (m != VK_SUCCESS)
                return m;
        }
        r = acquire(p, s.acquired[h], index);
    }
    if (r == VK_SUBOPTIMAL_KHR)
    {
        std::lock_guard<std::mutex> l(p->m);
        p->remake = true;
    }
    else if (r != VK_SUCCESS)
        return r;

    VkCommandBuffer cb = s.cb[h];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    if (p->queries)
    {
        vkCmdResetQueryPool(cb, p->queries, slot * 4 + h * 2, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->queries, slot * 4 + h * 2);
    }
    const bool asIs = src.layout == VK_IMAGE_LAYOUT_GENERAL || src.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout srcLayout = asIs ? src.layout : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier(cb, src.image, src.layout, srcLayout, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImage dst = p->images[index];
    barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {(int32_t)src.width, (int32_t)src.height, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)p->width, (int32_t)p->height, 1};
    const bool same = src.width == p->width && src.height == p->height;
    vkCmdBlitImage(cb, src.image, srcLayout, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
        same ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    if (!asIs)
        barrier(cb, src.image, srcLayout, src.layout, VK_ACCESS_TRANSFER_READ_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    if (p->queries)
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->queries, slot * 4 + h * 2 + 1);
    vkEndCommandBuffer(cb);

    VkSemaphore signals[2] = {p->copied[index], last ? last->released : VK_NULL_HANDLE};
    uint64_t signalValues[2] = {0, last ? last->releasedValue : 0};
    const uint32_t nSignals = last && last->released ? 2 : 1;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.signalSemaphoreValueCount = nSignals;
    ts.pSignalSemaphoreValues = signalValues;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &s.acquired[h];
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = nSignals;
    si.pSignalSemaphores = signals;
    if (last)
        vkResetFences(p->dev, 1, &s.fence);
    r = vkQueueSubmit(p->queue, 1, &si, last ? s.fence : VK_NULL_HANDLE);
    if (r != VK_SUCCESS)
        return r;
    s.timed[h] = p->queries != VK_NULL_HANDLE;

    VkPresentTimingInfoEXT pti{VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT};
    pti.timeDomainId = p->timeDomainId;
    pti.presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT;
    VkPresentTimingsInfoEXT ptis{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT};
    ptis.swapchainCount = 1;
    ptis.pTimingInfos = &pti;
    VkPresentId2KHR pid2{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR};
    pid2.swapchainCount = 1;
    pid2.pPresentIds = &id;
    VkPresentIdKHR pid1{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
    pid1.swapchainCount = 1;
    pid1.pPresentIds = &id;
    const void* chain = p->swapTiming ? &ptis : nullptr;
    if (p->id2)
        pid2.pNext = chain, chain = &pid2;
    else if (p->lowLatency)
        pid1.pNext = chain, chain = &pid1;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.pNext = chain;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &p->copied[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &p->swapchain;
    pi.pImageIndices = &index;
    {
        std::lock_guard<std::mutex> l(p->swapLock);
        marker(p, VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV, markerId);
        r = vkQueuePresentKHR(p->queue, &pi);
        marker(p, VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV, markerId);
    }
    if (r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR)
    {
        std::lock_guard<std::mutex> l(p->m);
        p->remake = true;
        r = VK_SUCCESS; // presented, or dropped by the platform: either way done with
    }
    if (r == VK_SUCCESS)
    {
        std::lock_guard<std::mutex> l(p->m);
        (h ? p->stats.realPresented : p->stats.generatedPresented)++;
    }
    return r;
}

void stopOnError(SlngxPresenter* p, VkResult r, const SlngxPresentFrame& f)
{
    PLOG_ERROR("sl-ngx presenter: frame %u -> %d", f.frame, r);
    std::lock_guard<std::mutex> l(p->m);
    p->stats.lastError = r;
    if (r == VK_ERROR_DEVICE_LOST)
    {
        p->lost = true;
        p->stats.deviceLost = 1;
    }
}

void process(SlngxPresenter* p, const SlngxPresentFrame& f)
{
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &f.ready;
    wi.pValues = &f.readyValue;
    VkResult r = vkWaitSemaphores(p->dev, &wi, kReadyTimeoutNs);
    if (r != VK_SUCCESS)
    {
        if (r == VK_TIMEOUT)
            PLOG_WARN("sl-ngx presenter: frame %u not ready after 1 s, released unshown", f.frame);
        else
            stopOnError(p, r, f);
        if (r != VK_ERROR_DEVICE_LOST)
        {
            // the frame before signals `released` on the GPU: a host signal
            // ahead of it would move the timeline backwards
            vkQueueWaitIdle(p->queue);
            hostRelease(p, f);
        }
        std::lock_guard<std::mutex> l(p->m);
        p->stats.framesSkipped++;
        return;
    }
    const auto ready = Clock::now();
    double halfNs;
    {
        std::lock_guard<std::mutex> l(p->m);
        if (p->haveLastReady)
        {
            double iv = std::chrono::duration<double, std::nano>(ready - p->lastReady).count();
            // a pause or a hitch is not the frame rate
            if (iv < 250e6)
                p->intervalAvgNs = p->intervalAvgNs ? p->intervalAvgNs * 0.9 + iv * 0.1 : iv;
        }
        p->lastReady = ready;
        p->haveLastReady = true;
        p->stats.intervalMs = (float)(p->intervalAvgNs / 1e6);
        halfNs = std::min(p->intervalAvgNs / 2, 50e6);
        p->halfAvgNs = halfNs;
        p->stats.presentHalfMs = (float)(halfNs / 1e6);
    }
    bool remake;
    {
        std::lock_guard<std::mutex> l(p->m);
        remake = p->remake;
    }
    if (remake || !p->swapchain)
    {
        vkQueueWaitIdle(p->queue);
        r = makeSwapchain(p);
        if (r != VK_SUCCESS)
        {
            if (r != VK_NOT_READY)
                stopOnError(p, r, f);
            if (r != VK_ERROR_DEVICE_LOST)
                hostRelease(p, f);
            std::lock_guard<std::mutex> l(p->m);
            p->stats.framesSkipped++;
            p->remake = true;
            return;
        }
    }
    const uint32_t slot = p->nextSlot;
    p->nextSlot = (p->nextSlot + 1) % kSlots;
    vkWaitForFences(p->dev, 1, &p->slots[slot].fence, VK_TRUE, kReadyTimeoutNs);
    readTimestamps(p, slot);

    const uint64_t realId = 2ull * f.frame + 2;
    if (f.generated.image)
    {
        r = presentOne(p, slot, 0, f.generated, realId - 1, realId, nullptr);
        if (r != VK_SUCCESS && r != VK_NOT_READY)
        {
            stopOnError(p, r, f);
            vkQueueWaitIdle(p->queue);
            if (r != VK_ERROR_DEVICE_LOST)
                hostRelease(p, f);
            return;
        }
        std::this_thread::sleep_until(ready + std::chrono::nanoseconds((int64_t)halfNs));
    }
    r = presentOne(p, slot, 1, f.real, realId, realId, &f);
    if (r != VK_SUCCESS)
    {
        if (r != VK_NOT_READY)
            stopOnError(p, r, f);
        vkQueueWaitIdle(p->queue);
        if (r != VK_ERROR_DEVICE_LOST)
            hostRelease(p, f);
        std::lock_guard<std::mutex> l(p->m);
        p->stats.framesSkipped++;
        return;
    }
    countShown(p);
}

void run(SlngxPresenter* p)
{
    for (;;)
    {
        SlngxPresentFrame f;
        {
            std::unique_lock<std::mutex> l(p->m);
            p->cv.wait(l, [&] { return p->stop || (!p->paused && !p->queued.empty()); });
            if (p->stop)
                return;
            f = p->queued.front();
            p->queued.pop_front();
            p->busy = true;
        }
        if (!p->lost)
            process(p, f);
        {
            std::lock_guard<std::mutex> l(p->m);
            p->busy = false;
        }
        p->cv.notify_all();
    }
}

// Releases what is still queued; the presenter's thread is idle or gone.
void drain(SlngxPresenter* p, std::unique_lock<std::mutex>& l)
{
    std::deque<SlngxPresentFrame> left;
    left.swap(p->queued);
    const bool lost = p->lost;
    l.unlock();
    vkQueueWaitIdle(p->queue);
    if (!lost)
        for (auto& f : left)
            hostRelease(p, f);
    l.lock();
    p->stats.framesSkipped += left.size();
}

void destroy(SlngxPresenter* p)
{
    {
        std::lock_guard<std::mutex> l(gp);
        if (gPresenter == p)
            gPresenter = nullptr;
    }
    {
        // a Reflex sleep still waiting on this presenter's semaphore
        std::unique_lock<std::mutex> l(gp);
        gpCv.wait(l, [] { return gSleepers == 0; });
    }
    {
        std::lock_guard<std::mutex> l(p->m);
        p->stop = true;
    }
    p->cv.notify_all();
    if (p->thread.joinable())
        p->thread.join();
    {
        std::unique_lock<std::mutex> l(p->m);
        drain(p, l);
    }
    destroySwapchainParts(p, p->swapchain, p->copied);
    p->swapchain = VK_NULL_HANDLE;
    for (auto& s : p->slots)
    {
        for (auto sem : s.acquired)
            if (sem)
                vkDestroySemaphore(p->dev, sem, nullptr);
        if (s.fence)
            vkDestroyFence(p->dev, s.fence, nullptr);
    }
    if (p->queries)
        vkDestroyQueryPool(p->dev, p->queries, nullptr);
    if (p->pool)
        vkDestroyCommandPool(p->dev, p->pool, nullptr);
    if (p->sleepSem)
        vkDestroySemaphore(p->dev, p->sleepSem, nullptr);
    delete p;
    PLOG_INFO("sl-ngx presenter: ended");
}

} // namespace

namespace slngx
{

sl::Result reflexSetOptions(const sl::ReflexOptions& options)
{
    {
        std::lock_guard<std::mutex> l(gReflexLock);
        gReflex.mode = options.mode;
        gReflex.frameLimitUs = options.frameLimitUs;
        gReflex.useMarkersToOptimize = options.useMarkersToOptimize;
    }
    std::lock_guard<std::mutex> l(gp);
    if (gPresenter)
    {
        std::lock_guard<std::mutex> s(gPresenter->swapLock);
        applySleepMode(gPresenter);
    }
    return Result::eOk;
}

sl::Result reflexSleep(const sl::FrameToken&)
{
    VkDevice dev;
    VkSemaphore sem;
    uint64_t value;
    {
        std::lock_guard<std::mutex> l(gp);
        SlngxPresenter* p = gPresenter;
        if (!p || !p->lowLatency || !p->latencySleep)
            return Result::eOk;
        {
            std::lock_guard<std::mutex> r(gReflexLock);
            if (gReflex.mode == sl::ReflexMode::eOff && !gReflex.frameLimitUs)
                return Result::eOk;
        }
        std::lock_guard<std::mutex> s(p->swapLock);
        if (!p->swapchain)
            return Result::eOk;
        value = ++p->sleepValue;
        VkLatencySleepInfoNV info{VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV};
        info.signalSemaphore = p->sleepSem;
        info.value = value;
        if (p->latencySleep(p->dev, p->swapchain, &info) != VK_SUCCESS)
            return Result::eErrorReflexAPI;
        dev = p->dev;
        sem = p->sleepSem;
        gSleepers++;
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &sem;
    wi.pValues = &value;
    vkWaitSemaphores(dev, &wi, kSleepTimeoutNs);
    {
        std::lock_guard<std::mutex> l(gp);
        gSleepers--;
    }
    gpCv.notify_all();
    return Result::eOk;
}

sl::Result pclSetMarker(sl::PCLMarker m, const sl::FrameToken& frame)
{
    VkLatencyMarkerNV vk;
    switch (m)
    {
    case sl::PCLMarker::eSimulationStart: vk = VK_LATENCY_MARKER_SIMULATION_START_NV; break;
    case sl::PCLMarker::eSimulationEnd: vk = VK_LATENCY_MARKER_SIMULATION_END_NV; break;
    case sl::PCLMarker::eRenderSubmitStart: vk = VK_LATENCY_MARKER_RENDERSUBMIT_START_NV; break;
    case sl::PCLMarker::eRenderSubmitEnd: vk = VK_LATENCY_MARKER_RENDERSUBMIT_END_NV; break;
    case sl::PCLMarker::ePresentStart: vk = VK_LATENCY_MARKER_PRESENT_START_NV; break;
    case sl::PCLMarker::ePresentEnd: vk = VK_LATENCY_MARKER_PRESENT_END_NV; break;
    case sl::PCLMarker::eTriggerFlash: vk = VK_LATENCY_MARKER_TRIGGER_FLASH_NV; break;
    case sl::PCLMarker::eControllerInputSample: vk = VK_LATENCY_MARKER_INPUT_SAMPLE_NV; break;
    case sl::PCLMarker::eOutOfBandRenderSubmitStart: vk = VK_LATENCY_MARKER_OUT_OF_BAND_RENDERSUBMIT_START_NV; break;
    case sl::PCLMarker::eOutOfBandRenderSubmitEnd: vk = VK_LATENCY_MARKER_OUT_OF_BAND_RENDERSUBMIT_END_NV; break;
    case sl::PCLMarker::eOutOfBandPresentStart: vk = VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV; break;
    case sl::PCLMarker::eOutOfBandPresentEnd: vk = VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV; break;
    default: return Result::eOk; // no Vulkan counterpart (the PC Latency ping, ...)
    }
    std::lock_guard<std::mutex> l(gp);
    SlngxPresenter* p = gPresenter;
    if (!p)
        return Result::eOk;
    std::lock_guard<std::mutex> s(p->swapLock);
    if (p->swapchain)
        marker(p, vk, 2ull * (uint32_t)frame + 2);
    return Result::eOk;
}

sl::Result reflexGetState(sl::ReflexState& state)
{
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    const bool device = currentDevice(inst, pd, dev);
    state.lowLatencyAvailable = device && deviceHasExtension(pd, VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
    state.latencyReportAvailable = state.lowLatencyAvailable;
    state.flashIndicatorDriverControlled = false;
    for (auto& r : state.frameReport)
        r = sl::ReflexReport{};
    std::lock_guard<std::mutex> l(gp);
    SlngxPresenter* p = gPresenter;
    if (!p || !p->lowLatency || !p->getTimings)
        return Result::eOk;
    std::lock_guard<std::mutex> s(p->swapLock);
    if (!p->swapchain)
        return Result::eOk;
    VkLatencyTimingsFrameReportNV reports[sl::kReflexFrameReportCount];
    for (auto& r : reports)
        r = VkLatencyTimingsFrameReportNV{VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV};
    VkGetLatencyMarkerInfoNV info{VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV};
    info.timingCount = sl::kReflexFrameReportCount;
    info.pTimings = reports;
    p->getTimings(p->dev, p->swapchain, &info);
    uint64_t prevEnd = 0;
    for (uint32_t i = 0; i < info.timingCount && i < (uint32_t)sl::kReflexFrameReportCount; i++)
    {
        const auto& v = reports[i];
        auto& o = state.frameReport[i];
        o.frameID = v.presentID >= 2 ? v.presentID / 2 - 1 : 0;
        o.inputSampleTime = v.inputSampleTimeUs;
        o.simStartTime = v.simStartTimeUs;
        o.simEndTime = v.simEndTimeUs;
        o.renderSubmitStartTime = v.renderSubmitStartTimeUs;
        o.renderSubmitEndTime = v.renderSubmitEndTimeUs;
        o.presentStartTime = v.presentStartTimeUs;
        o.presentEndTime = v.presentEndTimeUs;
        o.driverStartTime = v.driverStartTimeUs;
        o.driverEndTime = v.driverEndTimeUs;
        o.osRenderQueueStartTime = v.osRenderQueueStartTimeUs;
        o.osRenderQueueEndTime = v.osRenderQueueEndTimeUs;
        o.gpuRenderStartTime = v.gpuRenderStartTimeUs;
        o.gpuRenderEndTime = v.gpuRenderEndTimeUs;
        o.gpuActiveRenderTimeUs = (uint32_t)(v.gpuRenderEndTimeUs - v.gpuRenderStartTimeUs);
        o.gpuFrameTimeUs = prevEnd && v.gpuRenderEndTimeUs > prevEnd ? (uint32_t)(v.gpuRenderEndTimeUs - prevEnd) : 0;
        prevEnd = v.gpuRenderEndTimeUs;
    }
    return Result::eOk;
}

void endPresenter()
{
    SlngxPresenter* p;
    {
        std::lock_guard<std::mutex> l(gp);
        p = gPresenter;
    }
    if (p)
    {
        PLOG_WARN("sl-ngx presenter: still running at the device's end: ending it");
        destroy(p);
    }
}

} // namespace slngx

extern "C" {

SL_NGX_EXPORT VkResult slngxPresenterCreate(const SlngxPresenterDesc* desc, SlngxPresenter** out)
{
    if (!desc || !out || desc->version != SLNGX_PRESENTER_VERSION || !desc->surface)
        return VK_ERROR_INITIALIZATION_FAILED;
    *out = nullptr;
    auto* p = new SlngxPresenter;
    if (!slngx::currentDevice(p->instance, p->pd, p->dev))
    {
        delete p;
        PLOG_ERROR("sl-ngx presenter: no device yet (slSetVulkanInfo)");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    {
        std::lock_guard<std::mutex> l(gp);
        if (gPresenter)
        {
            delete p;
            PLOG_ERROR("sl-ngx presenter: one is running already");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    p->desc = *desc;
    p->wantW = desc->width, p->wantH = desc->height;
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(p->pd, desc->queueFamily, desc->surface, &present);
    if (!present)
    {
        delete p;
        PLOG_ERROR("sl-ngx presenter: queue family %u cannot present to the surface", desc->queueFamily);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(p->pd, desc->format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
    {
        delete p;
        PLOG_ERROR("sl-ngx presenter: format %d takes no blit", desc->format);
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    vkGetDeviceQueue(p->dev, desc->queueFamily, desc->queueIndex, &p->queue);
    p->lowLatency = desc->deviceBits & SLNGX_DEVICE_LOW_LATENCY2;
    p->id2 = desc->deviceBits & SLNGX_DEVICE_PRESENT_ID2;
    p->timing = desc->deviceBits & SLNGX_DEVICE_PRESENT_TIMING;
    if (p->lowLatency)
    {
        p->setSleepMode = devFn<PFN_vkSetLatencySleepModeNV>(p->dev, "vkSetLatencySleepModeNV");
        p->latencySleep = devFn<PFN_vkLatencySleepNV>(p->dev, "vkLatencySleepNV");
        p->setMarker = devFn<PFN_vkSetLatencyMarkerNV>(p->dev, "vkSetLatencyMarkerNV");
        p->getTimings = devFn<PFN_vkGetLatencyTimingsNV>(p->dev, "vkGetLatencyTimingsNV");
        p->notifyOob = devFn<PFN_vkQueueNotifyOutOfBandNV>(p->dev, "vkQueueNotifyOutOfBandNV");
        p->lowLatency = p->setSleepMode && p->latencySleep && p->setMarker && p->getTimings;
    }
    if (p->timing)
    {
        p->setTimingQueue = devFn<PFN_vkSetSwapchainPresentTimingQueueSizeEXT>(p->dev,
            "vkSetSwapchainPresentTimingQueueSizeEXT");
        p->pastTiming = devFn<PFN_vkGetPastPresentationTimingEXT>(p->dev, "vkGetPastPresentationTimingEXT");
        p->timeDomains = devFn<PFN_vkGetSwapchainTimeDomainPropertiesEXT>(p->dev,
            "vkGetSwapchainTimeDomainPropertiesEXT");
        p->timing = p->setTimingQueue && p->pastTiming && p->timeDomains;
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = desc->queueFamily;
    VkResult r = vkCreateCommandPool(p->dev, &pci, nullptr, &p->pool);
    for (auto& s : p->slots)
    {
        if (r != VK_SUCCESS)
            break;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = p->pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 2;
        r = vkAllocateCommandBuffers(p->dev, &ai, s.cb);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (auto& sem : s.acquired)
            if (r == VK_SUCCESS)
                r = vkCreateSemaphore(p->dev, &si, nullptr, &sem);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (r == VK_SUCCESS)
            r = vkCreateFence(p->dev, &fci, nullptr, &s.fence);
    }
    VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo tsi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    tsi.pNext = &tci;
    if (r == VK_SUCCESS)
        r = vkCreateSemaphore(p->dev, &tsi, nullptr, &p->sleepSem);
    // GPU time of the copies, where the queue family keeps timestamps
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(p->pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(p->pd, &nq, qf.data());
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(p->pd, &props);
    if (r == VK_SUCCESS && desc->queueFamily < nq && qf[desc->queueFamily].timestampValidBits)
    {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kSlots * 4;
        if (vkCreateQueryPool(p->dev, &qci, nullptr, &p->queries) != VK_SUCCESS)
            p->queries = VK_NULL_HANDLE;
        p->tsPeriodNs = props.limits.timestampPeriod;
    }
    if (r == VK_SUCCESS)
        r = makeSwapchain(p);
    if (r != VK_SUCCESS)
    {
        PLOG_ERROR("sl-ngx presenter: not made: %d", r);
        destroy(p);
        return r;
    }
    if (p->lowLatency && p->notifyOob)
    {
        VkOutOfBandQueueTypeInfoNV oob{VK_STRUCTURE_TYPE_OUT_OF_BAND_QUEUE_TYPE_INFO_NV};
        oob.queueType = VK_OUT_OF_BAND_QUEUE_TYPE_PRESENT_NV;
        p->notifyOob(p->queue, &oob);
    }
    p->stats.version = SLNGX_PRESENTER_VERSION;
    p->stats.lowLatency = p->lowLatency;
    p->thread = std::thread(run, p);
    {
        std::lock_guard<std::mutex> l(gp);
        gPresenter = p;
    }
    *out = p;
    return VK_SUCCESS;
}

SL_NGX_EXPORT VkResult slngxPresent(SlngxPresenter* p, const SlngxPresentFrame* f, uint64_t timeoutNs)
{
    if (!p || !f || f->version != SLNGX_PRESENTER_VERSION || !f->real.image || !f->ready)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!live(p))
        return VK_ERROR_DEVICE_LOST;
    std::unique_lock<std::mutex> l(p->m);
    if (p->lost)
        return VK_ERROR_DEVICE_LOST;
    if (p->paused)
        return VK_NOT_READY;
    // one frame waits while one is presented: more would only add latency
    if (!p->cv.wait_for(l, std::chrono::nanoseconds(timeoutNs), [&] { return p->queued.empty() || p->paused || p->lost; }))
        return VK_TIMEOUT;
    if (p->lost)
        return VK_ERROR_DEVICE_LOST;
    if (p->paused)
        return VK_NOT_READY;
    p->queued.push_back(*f);
    p->stats.framesAccepted++;
    l.unlock();
    p->cv.notify_all();
    return VK_SUCCESS;
}

SL_NGX_EXPORT void slngxPresenterResize(SlngxPresenter* p, uint32_t width, uint32_t height)
{
    if (!live(p))
        return;
    std::lock_guard<std::mutex> l(p->m);
    if (p->wantW != width || p->wantH != height)
    {
        p->wantW = width, p->wantH = height;
        p->remake = true;
    }
}

SL_NGX_EXPORT VkResult slngxPresenterPause(SlngxPresenter* p)
{
    if (!live(p))
        return VK_ERROR_DEVICE_LOST;
    std::unique_lock<std::mutex> l(p->m);
    p->paused = true;
    p->cv.notify_all();
    p->cv.wait(l, [&] { return !p->busy; });
    drain(p, l);
    return p->lost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

SL_NGX_EXPORT void slngxPresenterResume(SlngxPresenter* p)
{
    if (!live(p))
        return;
    {
        std::lock_guard<std::mutex> l(p->m);
        p->paused = false;
        p->haveLastReady = false;
    }
    p->cv.notify_all();
}

SL_NGX_EXPORT void slngxPresenterDestroy(SlngxPresenter* p)
{
    if (live(p))
        destroy(p);
}

SL_NGX_EXPORT VkResult slngxPresenterGetStats(SlngxPresenter* p, SlngxPresenterStats* out)
{
    if (!p || !out || out->version != SLNGX_PRESENTER_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!live(p))
        return VK_ERROR_DEVICE_LOST;
    uint32_t reports = 0;
    if (p->lowLatency && p->getTimings)
    {
        std::lock_guard<std::mutex> s(p->swapLock);
        if (p->swapchain)
        {
            VkGetLatencyMarkerInfoNV info{VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV};
            p->getTimings(p->dev, p->swapchain, &info); // the count alone
            reports = info.timingCount;
        }
    }
    std::lock_guard<std::mutex> l(p->m);
    *out = p->stats;
    out->latencyReports = reports;
    return VK_SUCCESS;
}

} // extern "C"
