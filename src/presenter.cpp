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
// Present ids: the generated frame of frame number f is 2f+1, the rendered
// one 2f+2, and Reflex's markers for frame f use 2f+2, as Streamline's
// DLSS-G does with the frame the application presents. A swapchain's
// present ids must increase: a frame whose number does not is presented
// without ids (0) and without the presenter's markers.
//
// Built against Vulkan headers without VK_NV_low_latency2, VK_KHR_present_id2
// or VK_EXT_present_timing, the parts that need them are left out and their
// device bits are ignored (SLNGX_HAVE_* below).

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
#include <set>
#include <thread>
#include <vector>

#ifdef _WIN32
#define SL_NGX_EXPORT __declspec(dllexport)
#else
#define SL_NGX_EXPORT __attribute__((visibility("default")))
#endif

#if defined(VK_NV_low_latency2)
#define SLNGX_HAVE_LL2 1
#endif
#if defined(VK_KHR_present_id2)
#define SLNGX_HAVE_ID2 1
#endif
// present timing's reports name their present by its VkPresentId2KHR id
#if defined(VK_EXT_present_timing) && defined(VK_KHR_present_id2)
#define SLNGX_HAVE_TIMING 1
#endif
// present fences (the EXT names, which both header generations have)
#if defined(VK_EXT_swapchain_maintenance1)
#define SLNGX_HAVE_M1 1
#endif

using sl::Result;

namespace
{

constexpr uint32_t kSlots = 3;
constexpr uint64_t kReadyTimeoutNs = 1000000000ull;
constexpr uint64_t kSleepTimeoutNs = 200000000ull;
constexpr uint32_t kTimingQueue = 64;
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
    // with present fences (maint1): what each half's last present signals
    // once the presentation engine is done with its semaphore, and whether
    // that present is outstanding
    VkFence presented[2]{};
    bool presentPending[2]{};
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
    // what the device has, from the device bits
    bool lowLatency = false, id2 = false, timing = false, maint1 = false;

    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR surfaceCaps2 = nullptr;
#if SLNGX_HAVE_LL2
    PFN_vkSetLatencySleepModeNV setSleepMode = nullptr;
    PFN_vkLatencySleepNV latencySleep = nullptr;
    PFN_vkSetLatencyMarkerNV setMarker = nullptr;
    PFN_vkGetLatencyTimingsNV getTimings = nullptr;
    PFN_vkQueueNotifyOutOfBandNV notifyOob = nullptr;
#endif
#if SLNGX_HAVE_TIMING
    PFN_vkSetSwapchainPresentTimingQueueSizeEXT setTimingQueue = nullptr;
    PFN_vkGetPastPresentationTimingEXT pastTiming = nullptr;
    PFN_vkGetSwapchainTimeDomainPropertiesEXT timeDomains = nullptr;
#endif

    // The swapchain: made and destroyed on the presenter's thread (or by
    // Destroy once it has stopped); the application's thread (Reflex,
    // slngxPresenterGetStats) reads it under swapLock. Lock order: gp,
    // swapLock, gReflexLock; m is never held while swapLock or gp is
    // taken (a Reflex sleep takes m under gp).
    std::mutex swapLock;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> copied;
    uint32_t width = 0, height = 0;
    // what the surface allows on this swapchain (presenter's thread;
    // swapTiming written under swapLock, which countShown reads it under)
    bool swapId2 = false, swapTiming = false;
    uint64_t timeDomainId = 0;
    uint32_t timingStage = 0; // VkPresentStageFlagsEXT queried
    // presents whose timing is not reported complete yet: changed under
    // swapLock; the presenter's thread reads it without (only countShown,
    // under swapLock, lowers it)
    std::atomic<uint32_t> timingPending{0};
    uint64_t lastRealId = 0;    // the last rendered frame's present id
    bool idWarned = false, suboptimalRemade = false;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    float tsPeriodNs = 0;
    Slot slots[kSlots];
    uint32_t nextSlot = 0;

    VkSemaphore sleepSem = VK_NULL_HANDLE;
    uint64_t sleepValue = 0; // the last value a Reflex sleep was given (under gp and swapLock)

    // handed-over frames and the thread's state
    std::mutex m;
    std::condition_variable cv;
    std::deque<SlngxPresentFrame> queued;
    bool busy = false, paused = false, stop = false, lost = false, remake = false;
    VkResult failed = VK_SUCCESS;  // an error the presenter cannot go on after
    uint32_t wantW = 0, wantH = 0;
    SlngxPresenterStats stats{};
    double intervalAvgNs = 0, copyAvgMs = 0, halfAvgNs = 0;
    Clock::time_point lastReady{};
    bool haveLastReady = false;

    std::thread thread;
};

namespace
{

std::mutex gp;                       // guards gPresenter, gCreating, gEnded and gSleepers
std::condition_variable gpCv;
SlngxPresenter* gPresenter = nullptr;
bool gCreating = false;
// Presenters ended under the application (slShutdown, a new device): kept
// until slngxPresenterDestroy, so that their address is not reused while
// the application still holds the handle.
std::set<SlngxPresenter*> gEnded;
int gSleepers = 0;

// Whether the handle is the running presenter.
bool live(SlngxPresenter* p)
{
    std::lock_guard<std::mutex> l(gp);
    return p && p == gPresenter;
}

template <class T> T devFn(VkDevice dev, const char* name)
{
    return reinterpret_cast<T>(vkGetDeviceProcAddr(dev, name));
}

// Under swapLock. The driver applies minimumIntervalUs between Reflex
// sleeps, once per rendered frame, not per present (tests/present measures
// it): frameLimitUs passes through unchanged.
void applySleepMode(SlngxPresenter* p)
{
#if SLNGX_HAVE_LL2
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
#else
    (void)p;
#endif
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
        if (s)
            vkDestroySemaphore(p->dev, s, nullptr);
    sems.clear();
    if (sc)
        vkDestroySwapchainKHR(p->dev, sc, nullptr);
}

// With present fences (maint1): waits until the presentation engine is
// done with every present made so far, at most kReadyTimeoutNs. Without
// them VK_KHR_swapchain gives no such signal, and the vkQueueWaitIdle
// before a swapchain is destroyed is all there is (the usual practice).
VkResult waitPresented(SlngxPresenter* p)
{
    VkFence fences[kSlots * 2];
    uint32_t n = 0;
    for (auto& s : p->slots)
        for (int h = 0; h < 2; h++)
            if (s.presentPending[h])
                fences[n++] = s.presented[h];
    if (!n)
        return VK_SUCCESS;
    VkResult r = vkWaitForFences(p->dev, n, fences, VK_TRUE, kReadyTimeoutNs);
    if (r == VK_SUCCESS)
        for (auto& s : p->slots)
            s.presentPending[0] = s.presentPending[1] = false;
    return r;
}

// Destroys a swapchain taken out of p->swapchain (a retired one: it was the
// oldSwapchain of the one made after it), once the presentation engine is
// done with its presents and a Reflex sleep the application issued on it
// is over. Where either is not known to be over, it is left undestroyed:
// a retired swapchain holds no surface.
void retire(SlngxPresenter* p, VkSwapchainKHR sc, std::vector<VkSemaphore>& sems, uint64_t sleepValue)
{
    VkResult r = sc ? waitPresented(p) : VK_SUCCESS;
    if (sc && r == VK_SUCCESS && p->sleepSem && sleepValue)
    {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &p->sleepSem;
        wi.pValues = &sleepValue;
        r = vkWaitSemaphores(p->dev, &wi, kSleepTimeoutNs);
    }
    // a lost device's objects may be destroyed whatever is pending
    if (r != VK_SUCCESS && r != VK_ERROR_DEVICE_LOST)
    {
        PLOG_WARN("sl-ngx presenter: an old swapchain is still in use (%d): left undestroyed", r);
        sems.clear();
        return;
    }
    destroySwapchainParts(p, sc, sems);
}

// Leaves no swapchain: after a failed remake, whose oldSwapchain Vulkan
// retired all the same.
void dropSwapchain(SlngxPresenter* p)
{
    VkSwapchainKHR old;
    std::vector<VkSemaphore> sems;
    uint64_t sleepValue;
    {
        std::lock_guard<std::mutex> l(p->swapLock);
        old = p->swapchain;
        sems.swap(p->copied);
        p->swapchain = VK_NULL_HANDLE;
        p->images.clear();
        sleepValue = p->sleepValue;
        p->swapTiming = false;
        p->timingPending = 0;
    }
    p->swapId2 = false;
    retire(p, old, sems, sleepValue);
}

VkCompositeAlphaFlagBitsKHR compositeAlpha(VkCompositeAlphaFlagsKHR supported)
{
    for (auto a : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
             VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR})
        if (supported & a)
            return a;
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
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

    // what the surface allows of the device's present features (the device
    // extensions need VK_KHR_get_surface_capabilities2 on the instance)
    bool id2 = false, timing = false;
    uint32_t stage = 0;
#if SLNGX_HAVE_ID2
    if (p->id2 && p->surfaceCaps2)
    {
        VkPhysicalDeviceSurfaceInfo2KHR si{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR};
        si.surface = p->desc.surface;
        VkSurfaceCapabilitiesPresentId2KHR cid{VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR};
        VkSurfaceCapabilities2KHR c2{VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR};
        c2.pNext = &cid;
#if SLNGX_HAVE_TIMING
        VkPresentTimingSurfaceCapabilitiesEXT ct{VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
        if (p->timing)
            cid.pNext = &ct;
#endif
        if (p->surfaceCaps2(p->pd, &si, &c2) == VK_SUCCESS)
        {
            id2 = cid.presentId2Supported;
#if SLNGX_HAVE_TIMING
            // the stage at which the display starts to show the frame
            if (id2 && p->timing && ct.presentTimingSupported)
            {
                if (ct.presentStageQueries & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT)
                    stage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT;
                else if (ct.presentStageQueries & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT)
                    stage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT;
            }
            timing = stage != 0;
#endif
        }
    }
#endif

    (void)timing; // without the present timing headers
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
#if SLNGX_HAVE_LL2
    VkSwapchainLatencyCreateInfoNV latency{VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV};
    latency.latencyModeEnable = VK_TRUE;
    ci.pNext = p->lowLatency ? &latency : nullptr;
#endif
#if SLNGX_HAVE_ID2
    if (id2)
        ci.flags |= VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
#endif
#if SLNGX_HAVE_TIMING
    if (timing)
        ci.flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT;
#endif
    ci.surface = p->desc.surface;
    ci.minImageCount = count;
    ci.imageFormat = p->desc.format;
    ci.imageColorSpace = p->desc.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT; // checked against the surface at create
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
    ci.compositeAlpha = compositeAlpha(caps.supportedCompositeAlpha);
    // FIFO: the only mode where every present reaches the display in order
    // without tearing; the pacer's half-frame wait spaces the pair
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    {
        // the old swapchain is taken for itself (oldSwapchain is externally synchronised)
        std::lock_guard<std::mutex> l(p->swapLock);
        ci.oldSwapchain = p->swapchain;
        r = vkCreateSwapchainKHR(p->dev, &ci, nullptr, &sc);
    }
    if (r != VK_SUCCESS)
    {
        dropSwapchain(p);
        return r;
    }
    uint32_t n = 0;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> copied;
    r = vkGetSwapchainImagesKHR(p->dev, sc, &n, nullptr);
    if (r == VK_SUCCESS)
    {
        images.resize(n);
        r = vkGetSwapchainImagesKHR(p->dev, sc, &n, images.data());
    }
    for (uint32_t i = 0; i < n && r == VK_SUCCESS; i++)
    {
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore s = VK_NULL_HANDLE;
        r = vkCreateSemaphore(p->dev, &si, nullptr, &s);
        if (r == VK_SUCCESS)
            copied.push_back(s);
    }
    if (r != VK_SUCCESS)
    {
        destroySwapchainParts(p, sc, copied);
        dropSwapchain(p);
        return r;
    }
    VkSwapchainKHR old;
    std::vector<VkSemaphore> oldSems;
    uint64_t sleepValue;
    bool swapTiming = false;
    {
        std::lock_guard<std::mutex> l(p->swapLock);
        old = p->swapchain;
        oldSems.swap(p->copied);
        p->swapchain = sc;
        p->images = std::move(images);
        p->copied = std::move(copied);
        p->width = extent.width, p->height = extent.height;
#if SLNGX_HAVE_TIMING
        if (timing && p->setTimingQueue)
            swapTiming = p->setTimingQueue(p->dev, sc, kTimingQueue) == VK_SUCCESS;
        if (swapTiming)
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
            swapTiming = (d == VK_SUCCESS || d == VK_INCOMPLETE) && dp.timeDomainCount > 0;
            if (swapTiming)
                p->timeDomainId = ids[0];
        }
#endif
        sleepValue = p->sleepValue;
        applySleepMode(p);
        p->swapTiming = swapTiming;
        p->timingPending = 0;
    }
    p->swapId2 = id2;
    p->timingStage = stage;
    // lastRealId stays: a swapchain made between a frame's two presents
    // gets that frame's ids, so ids increase across swapchains too
    retire(p, old, oldSems, sleepValue);
    std::lock_guard<std::mutex> l(p->m);
    p->stats.swapchainsMade++;
    p->stats.width = extent.width, p->stats.height = extent.height;
    p->stats.presentTiming = swapTiming;
    PLOG_INFO("sl-ngx presenter: swapchain %ux%u, %u images, Reflex %s, present id2 %s, present timing %s",
        extent.width, extent.height, n, p->lowLatency ? "on" : "off", id2 ? "on" : "off", swapTiming ? "on" : "off");
    return VK_SUCCESS;
}

void readTimestamps(SlngxPresenter* p, uint32_t slot)
{
    Slot& s = p->slots[slot];
    if (!p->queries || !(s.timed[0] || s.timed[1]))
        return;
    double ms = 0;
    // each half separately: a frame without a generated image leaves the first unwritten
    for (int h = 0; h < 2; h++)
    {
        if (!s.timed[h])
            continue;
        uint64_t v[2][2]{};
        VkResult r = vkGetQueryPoolResults(p->dev, p->queries, slot * 4 + h * 2, 2, sizeof(v), v, sizeof(v[0]),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if ((r == VK_SUCCESS || r == VK_NOT_READY) && v[0][1] && v[1][1] && v[1][0] >= v[0][0])
            ms += (double)(v[1][0] - v[0][0]) * p->tsPeriodNs / 1e6;
    }
    s.timed[0] = s.timed[1] = false;
    std::lock_guard<std::mutex> l(p->m);
    p->copyAvgMs = p->copyAvgMs ? p->copyAvgMs * 0.9 + ms * 0.1 : ms;
    p->stats.copyGpuMs = (float)p->copyAvgMs;
}

// Counts the frames present timing reports shown, and frees their slots in
// the swapchain's timing queue (only complete reports free one). From the
// presenter's thread before each frame, and from slngxPresenterGetStats,
// so that the counts also come in while no frame follows.
void countShown(SlngxPresenter* p)
{
#if SLNGX_HAVE_TIMING
    if (!p->pastTiming)
        return;
    VkPastPresentationTimingEXT pt[kTimingQueue];
    VkPresentStageTimeEXT st[kTimingQueue];
    uint64_t generated = 0, real = 0;
    {
        // the swapchain is externally synchronised here too, and its
        // timing state is the presenter thread's and this caller's both
        std::lock_guard<std::mutex> l(p->swapLock);
        if (!p->swapchain || !p->swapTiming || !p->timingPending)
            return;
        uint32_t complete = 0;
        for (int round = 0; round < 4; round++)
        {
            for (uint32_t i = 0; i < kTimingQueue; i++)
            {
                pt[i] = VkPastPresentationTimingEXT{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT};
                pt[i].presentStageCount = 1;
                pt[i].pPresentStages = &st[i];
            }
            VkPastPresentationTimingInfoEXT info{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT};
            info.flags = VK_PAST_PRESENTATION_TIMING_ALLOW_OUT_OF_ORDER_RESULTS_BIT_EXT;
            info.swapchain = p->swapchain;
            VkPastPresentationTimingPropertiesEXT props{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT};
            props.presentationTimingCount = kTimingQueue;
            props.pPresentationTimings = pt;
            VkResult r = p->pastTiming(p->dev, &info, &props);
            if (r != VK_SUCCESS && r != VK_INCOMPLETE)
                break;
            for (uint32_t i = 0; i < props.presentationTimingCount; i++)
            {
                if (!pt[i].reportComplete)
                    continue;
                complete++;
                if (pt[i].presentStageCount && st[i].time)
                    (pt[i].presentId & 1 ? generated : real)++;
            }
            if (r == VK_SUCCESS)
                break;
        }
        p->timingPending -= std::min(complete, p->timingPending.load());
    }
    if (!generated && !real)
        return;
    std::lock_guard<std::mutex> l(p->m);
    p->stats.generatedShown += generated;
    p->stats.realShown += real;
#else
    (void)p;
#endif
}

// Under swapLock.
void marker(SlngxPresenter* p, int m, uint64_t id)
{
#if SLNGX_HAVE_LL2
    if (!p->lowLatency || !p->setMarker || !id)
        return;
    VkSetLatencyMarkerInfoNV info{VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV};
    info.presentID = id;
    info.marker = (VkLatencyMarkerNV)m;
    p->setMarker(p->dev, p->swapchain, &info);
#else
    (void)p, (void)m, (void)id;
#endif
}

void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess,
    VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    // no ownership transfer: the images belong to the presenter's queue
    // family or are shared (present.h)
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

// A suboptimal swapchain is remade where the surface changed size, or once
// otherwise: a platform that reports every swapchain suboptimal (a rotated
// output) must not get one made per frame.
void onSuboptimal(SlngxPresenter* p)
{
    VkSurfaceCapabilitiesKHR caps{};
    const bool resized = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p->pd, p->desc.surface, &caps) == VK_SUCCESS &&
        caps.currentExtent.width != 0xffffffffu &&
        (caps.currentExtent.width != p->width || caps.currentExtent.height != p->height);
    if (!resized && p->suboptimalRemade)
        return;
    p->suboptimalRemade = !resized;
    std::lock_guard<std::mutex> l(p->m);
    p->remake = true;
}

void fail(SlngxPresenter* p, VkResult r)
{
    std::lock_guard<std::mutex> l(p->m);
    if (r == VK_ERROR_DEVICE_LOST)
    {
        p->lost = true;
        p->stats.deviceLost = 1;
    }
    else if (p->failed == VK_SUCCESS)
        p->failed = r;
    p->stats.lastError = r;
}

// The results a frame may stop on with the presenter going on: a late
// image or report (VK_TIMEOUT), a minimised window (VK_NOT_READY), a
// swapchain to remake (VK_ERROR_OUT_OF_DATE_KHR, VK_SUBOPTIMAL_KHR). Any
// other error stops the presenter.
bool transient(VkResult r)
{
    return r >= 0 || r == VK_ERROR_OUT_OF_DATE_KHR;
}

// Copies one image into a swapchain image and presents it. `last`: the
// frame's final submit, which signals `released` and the slot's fence;
// `releaseSubmitted` tells whether that submit went in.
VkResult presentOne(SlngxPresenter* p, uint32_t slot, int h, const SlngxPresentImage& src, uint64_t id,
    uint64_t markerId, const SlngxPresentFrame* last, bool& releaseSubmitted)
{
    Slot& s = p->slots[slot];
    if (s.presentPending[h])
    {
        // this half's present fence, before anything is acquired: a late
        // one leaves nothing pending behind it
        VkResult w = vkWaitForFences(p->dev, 1, &s.presented[h], VK_TRUE, kReadyTimeoutNs);
        if (w != VK_SUCCESS)
            return w;
        s.presentPending[h] = false;
    }
    if (s.presented[h])
    {
        VkResult w = vkResetFences(p->dev, 1, &s.presented[h]);
        if (w != VK_SUCCESS)
        {
            fail(p, w);
            return w;
        }
    }
    uint32_t index = 0;
    VkResult r = VK_ERROR_OUT_OF_DATE_KHR;
    for (int attempt = 0; attempt < 2 && r == VK_ERROR_OUT_OF_DATE_KHR; attempt++)
    {
        if (attempt || !p->swapchain)
        {
            if (attempt)
                p->suboptimalRemade = false;
            VkResult w = vkQueueWaitIdle(p->queue);
            if (w != VK_SUCCESS)
                return w;
            VkResult m = makeSwapchain(p);
            if (m != VK_SUCCESS)
                return m;
        }
        r = acquire(p, s.acquired[h], index);
    }
    if (r == VK_SUBOPTIMAL_KHR)
        onSuboptimal(p);
    else if (r != VK_SUCCESS)
        return r;

    // from here on a failure leaves the acquired image and its semaphore
    // pending: the presenter cannot go on with this swapchain's sync objects
    VkCommandBuffer cb = s.cb[h];
    r = vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (r == VK_SUCCESS)
        r = vkBeginCommandBuffer(cb, &bi);
    if (r != VK_SUCCESS)
    {
        fail(p, r);
        return r;
    }
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
    // scaled: linear, which the source format must filter (present.h)
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
    r = vkEndCommandBuffer(cb);
    if (r != VK_SUCCESS)
    {
        fail(p, r);
        return r;
    }

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
        r = vkResetFences(p->dev, 1, &s.fence);
    if (r == VK_SUCCESS)
        r = vkQueueSubmit(p->queue, 1, &si, last ? s.fence : VK_NULL_HANDLE);
    if (r != VK_SUCCESS)
    {
        fail(p, r);
        return r;
    }
    if (last)
        releaseSubmitted = true;
    s.timed[h] = p->queries != VK_NULL_HANDLE;

    const void* chain = nullptr;
    bool timed = false;
#if SLNGX_HAVE_TIMING
    VkPresentTimingInfoEXT pti{VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT};
    pti.timeDomainId = p->timeDomainId;
    pti.presentStageQueries = p->timingStage;
    VkPresentTimingsInfoEXT ptis{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT};
    ptis.swapchainCount = 1;
    ptis.pTimingInfos = &pti;
    // a request takes a slot in the swapchain's timing queue until its
    // report completes; a full queue would fail the present
    if (p->swapTiming && id && p->timingPending < kTimingQueue)
        chain = &ptis, timed = true;
#endif
    bool idChained = false;
#if SLNGX_HAVE_ID2
    VkPresentId2KHR pid2{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR};
    pid2.swapchainCount = 1;
    pid2.pPresentIds = &id;
    if (p->swapId2)
        pid2.pNext = chain, chain = &pid2, idChained = true;
#endif
#if SLNGX_HAVE_LL2
    // VK_NV_low_latency2's markers match presents by VK_KHR_present_id's id
    VkPresentIdKHR pid1{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
    pid1.swapchainCount = 1;
    pid1.pPresentIds = &id;
    if (!idChained && p->lowLatency)
        pid1.pNext = chain, chain = &pid1;
#endif
    (void)idChained;
#if SLNGX_HAVE_M1
    VkSwapchainPresentFenceInfoEXT pf{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT};
    pf.swapchainCount = 1;
    pf.pFences = &s.presented[h];
    if (s.presented[h])
        pf.pNext = chain, chain = &pf;
#endif
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.pNext = chain;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &p->copied[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &p->swapchain;
    pi.pImageIndices = &index;
    {
        // A FIFO present can block until a display slot frees, and holds
        // swapLock meanwhile: Reflex calls on the application's thread wait
        // that long. The swapchain is externally synchronised; a second
        // lock order between the two threads would cost more.
        std::lock_guard<std::mutex> l(p->swapLock);
#if SLNGX_HAVE_LL2
        marker(p, VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV, markerId);
#endif
        r = vkQueuePresentKHR(p->queue, &pi);
#if SLNGX_HAVE_LL2
        marker(p, VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV, markerId);
#endif
        if ((r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) && timed)
            p->timingPending++;
    }
    (void)markerId;
    // a present the presentation engine rejected with these is enqueued all
    // the same (VK_KHR_swapchain_maintenance1), and signals its fence
    s.presentPending[h] = s.presented[h] &&
        (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR ||
            r == VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
    bool shown = r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR;
    if (r == VK_SUBOPTIMAL_KHR)
    {
        onSuboptimal(p);
        r = VK_SUCCESS;
    }
    else if (r == VK_ERROR_OUT_OF_DATE_KHR)
    {
        p->suboptimalRemade = false;
        std::lock_guard<std::mutex> l(p->m);
        p->remake = true;
        r = VK_SUCCESS; // dropped by the platform: done with all the same
    }
#if SLNGX_HAVE_TIMING
    else if (r == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT)
    {
        // not expected (requests stop short of the queue's size): a new
        // swapchain starts with an empty queue
        PLOG_WARN("sl-ngx presenter: present timing queue full, swapchain remade");
        std::lock_guard<std::mutex> l(p->m);
        p->remake = true;
        r = VK_SUCCESS;
    }
#endif
    if (shown)
    {
        std::lock_guard<std::mutex> l(p->m);
        (h ? p->stats.realPresented : p->stats.generatedPresented)++;
    }
    return r;
}

void stopOnError(SlngxPresenter* p, VkResult r, const SlngxPresentFrame& f)
{
    PLOG_ERROR("sl-ngx presenter: frame %u -> %d", f.frame, r);
    if (!transient(r))
    {
        fail(p, r); // a lost surface, no memory, a lost device: slngxPresent answers it
        return;
    }
    std::lock_guard<std::mutex> l(p->m);
    p->stats.lastError = r;
}

// The frame is not presented: release it (unless its final submit already
// does, or the device is lost) and count it skipped.
void skipFrame(SlngxPresenter* p, const SlngxPresentFrame& f, bool releaseSubmitted)
{
    bool lost;
    {
        std::lock_guard<std::mutex> l(p->m);
        lost = p->lost;
    }
    if (!lost && !releaseSubmitted)
    {
        // the frame before signals `released` on the GPU: a host signal
        // ahead of it would move the timeline backwards
        if (vkQueueWaitIdle(p->queue) == VK_ERROR_DEVICE_LOST)
            fail(p, VK_ERROR_DEVICE_LOST);
        else
            hostRelease(p, f);
    }
    std::lock_guard<std::mutex> l(p->m);
    p->stats.framesSkipped++;
}

void process(SlngxPresenter* p, const SlngxPresentFrame& f)
{
    {
        std::unique_lock<std::mutex> l(p->m);
        if (p->lost)
        {
            p->stats.framesSkipped++; // nothing can be released on a lost device
            return;
        }
        if (p->failed != VK_SUCCESS)
        {
            l.unlock();
            skipFrame(p, f, false);
            return;
        }
    }
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
        skipFrame(p, f, false);
        return;
    }
    const auto ready = Clock::now();
    double halfNs;
    bool remake;
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
        remake = p->remake;
    }
    // reports of earlier presents first: they free timing queue slots
    countShown(p);
    if (remake || !p->swapchain)
    {
        r = vkQueueWaitIdle(p->queue);
        if (r == VK_SUCCESS)
            r = makeSwapchain(p);
        if (r != VK_SUCCESS)
        {
            if (r != VK_NOT_READY)
                stopOnError(p, r, f);
            skipFrame(p, f, false);
            std::lock_guard<std::mutex> l(p->m);
            p->remake = true;
            return;
        }
    }
    const uint32_t slot = p->nextSlot;
    p->nextSlot = (p->nextSlot + 1) % kSlots;
    // the slot's last frame must be done before its command buffers and
    // acquire semaphores are used again
    r = vkWaitForFences(p->dev, 1, &p->slots[slot].fence, VK_TRUE, kReadyTimeoutNs);
    if (r == VK_TIMEOUT)
    {
        PLOG_WARN("sl-ngx presenter: copy of frame %u's slot still running after 1 s", f.frame);
        r = vkQueueWaitIdle(p->queue);
    }
    if (r != VK_SUCCESS)
    {
        stopOnError(p, r, f);
        skipFrame(p, f, false);
        return;
    }
    readTimestamps(p, slot);

    uint64_t realId = 2ull * f.frame + 2;
    if (realId <= p->lastRealId)
    {
        if (!p->idWarned)
            PLOG_WARN("sl-ngx presenter: frame number %u does not increase: presented without present ids", f.frame);
        p->idWarned = true;
        realId = 0;
    }
    else
        p->lastRealId = realId;
    const uint64_t generatedId = realId ? realId - 1 : 0;

    bool released = false;
    if (f.generated.image)
    {
        r = presentOne(p, slot, 0, f.generated, generatedId, realId, nullptr, released);
        if (r != VK_SUCCESS)
        {
            if (r != VK_NOT_READY)
                stopOnError(p, r, f);
            skipFrame(p, f, false);
            return;
        }
        std::this_thread::sleep_until(ready + std::chrono::nanoseconds((int64_t)halfNs));
    }
    r = presentOne(p, slot, 1, f.real, realId, realId, &f, released);
    if (r != VK_SUCCESS)
    {
        if (r != VK_NOT_READY)
            stopOnError(p, r, f);
        skipFrame(p, f, released);
    }
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
        // a caller waiting to hand over the next frame can now
        p->cv.notify_all();
        process(p, f);
        {
            std::lock_guard<std::mutex> l(p->m);
            p->busy = false;
        }
        p->cv.notify_all();
    }
}

// Waits for the presenter's queue (the copies of the frames it took, and
// their release, may still run there) and releases what is still queued;
// the presenter's thread is idle or gone. A failed wait stops the
// presenter, and the frames still queued stay unreleased.
VkResult drain(SlngxPresenter* p, std::unique_lock<std::mutex>& l)
{
    std::deque<SlngxPresentFrame> left;
    left.swap(p->queued);
    const bool lost = p->lost;
    l.unlock();
    VkResult r = VK_SUCCESS;
    if (!lost && p->queue)
    {
        r = vkQueueWaitIdle(p->queue);
        if (r != VK_SUCCESS)
            fail(p, r);
        else
            for (auto& f : left)
                hostRelease(p, f);
    }
    l.lock();
    p->stats.framesSkipped += left.size();
    return r;
}

// Stops the thread, releases what is queued and destroys the presenter's
// Vulkan objects; the struct stays. It is no longer gPresenter.
void teardown(SlngxPresenter* p)
{
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
    if (p->queue)
        vkQueueWaitIdle(p->queue);
    // the presentation engine's use of the present semaphores: known over
    // with present fences only (waitPresented). Where they do not signal,
    // the semaphores and fences are left, and the swapchain is destroyed
    // all the same, so that the surface is the application's again.
    const VkResult presented = waitPresented(p);
    const bool presentsOver = presented == VK_SUCCESS || presented == VK_ERROR_DEVICE_LOST;
    if (!presentsOver)
    {
        PLOG_WARN("sl-ngx presenter: presents still outstanding at the end (%d): their semaphores left", presented);
        p->copied.clear();
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
        for (int h = 0; h < 2; h++)
            if (s.presented[h] && (presentsOver || !s.presentPending[h]))
                vkDestroyFence(p->dev, s.presented[h], nullptr);
        s = Slot{};
    }
    if (p->queries)
        vkDestroyQueryPool(p->dev, p->queries, nullptr);
    if (p->pool)
        vkDestroyCommandPool(p->dev, p->pool, nullptr);
    if (p->sleepSem)
    {
        // every Reflex sleep is over (gSleepers), but one may have stopped
        // waiting before the driver signalled (the presenter was ended
        // meanwhile): the semaphore is destroyed once that signal came, or
        // left
        uint64_t reached = 0;
        VkResult r = vkGetSemaphoreCounterValue(p->dev, p->sleepSem, &reached);
        if (r == VK_SUCCESS && reached < p->sleepValue)
        {
            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.semaphoreCount = 1;
            wi.pSemaphores = &p->sleepSem;
            wi.pValues = &p->sleepValue;
            r = vkWaitSemaphores(p->dev, &wi, kReadyTimeoutNs);
        }
        if (r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST)
            vkDestroySemaphore(p->dev, p->sleepSem, nullptr);
        else
            PLOG_WARN("sl-ngx presenter: a Reflex sleep still pending at the end (%d): its semaphore left", r);
    }
    p->queries = VK_NULL_HANDLE;
    p->pool = VK_NULL_HANDLE;
    p->sleepSem = VK_NULL_HANDLE;
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
#if SLNGX_HAVE_LL2
    VkDevice dev;
    VkSemaphore sem;
    uint64_t value;
    SlngxPresenter* self;
    {
        std::lock_guard<std::mutex> l(gp);
        SlngxPresenter* p = self = gPresenter;
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
        value = p->sleepValue + 1;
        VkLatencySleepInfoNV info{VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV};
        info.signalSemaphore = p->sleepSem;
        info.value = value;
        if (p->latencySleep(p->dev, p->swapchain, &info) != VK_SUCCESS)
            return Result::eErrorReflexAPI;
        p->sleepValue = value; // only a value that will be signalled
        dev = p->dev;
        sem = p->sleepSem;
        gSleepers++;
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &sem;
    wi.pValues = &value;
    for (;;)
    {
        // a wakeup further off than one wait (a long frame limit) is waited
        // for: the semaphore and the swapchain stay while a signal is due.
        // A presenter ended meanwhile (slShutdown, a new device) is not
        // waited on: teardown checks the semaphore before destroying it,
        // and a replaced device's objects are left as they are. Nor is a
        // stopped one (a lost surface may never signal): the application
        // learns of it at its next frame and destroys it.
        VkResult r = vkWaitSemaphores(dev, &wi, kSleepTimeoutNs);
        if (r != VK_TIMEOUT)
            break;
        std::lock_guard<std::mutex> l(gp); // gp, then m (the lock order)
        if (gPresenter != self)
            break;
        std::lock_guard<std::mutex> m(self->m);
        if (self->lost || self->failed != VK_SUCCESS)
            break;
    }
    {
        std::lock_guard<std::mutex> l(gp);
        gSleepers--;
    }
    gpCv.notify_all();
#endif
    return Result::eOk;
}

sl::Result pclSetMarker(sl::PCLMarker m, const sl::FrameToken& frame)
{
#if SLNGX_HAVE_LL2
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
#else
    (void)m, (void)frame;
#endif
    return Result::eOk;
}

sl::Result reflexGetState(sl::ReflexState& state)
{
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    const bool device = currentDevice(inst, pd, dev);
#if SLNGX_HAVE_LL2
    state.lowLatencyAvailable = device && deviceHasExtension(pd, "VK_NV_low_latency2");
#else
    (void)device;
    state.lowLatencyAvailable = false;
#endif
    state.latencyReportAvailable = false;
    state.statsWindowMessage = 0;
    state.flashIndicatorDriverControlled = false;
    for (auto& r : state.frameReport)
        r = sl::ReflexReport{};
    if (state.structVersion >= sl::kStructVersion2)
        for (auto& r : state.frameReport2)
            r = sl::ReflexReport2{};
#if SLNGX_HAVE_LL2
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
    const uint32_t n = std::min(info.timingCount, (uint32_t)sl::kReflexFrameReportCount);
    state.latencyReportAvailable = n > 0;
    uint64_t prevEnd = 0;
    for (uint32_t i = 0; i < n; i++)
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
        o.gpuActiveRenderTimeUs = v.gpuRenderEndTimeUs > v.gpuRenderStartTimeUs
            ? (uint32_t)(v.gpuRenderEndTimeUs - v.gpuRenderStartTimeUs) : 0;
        o.gpuFrameTimeUs = prevEnd && v.gpuRenderEndTimeUs > prevEnd ? (uint32_t)(v.gpuRenderEndTimeUs - prevEnd) : 0;
        prevEnd = v.gpuRenderEndTimeUs;
    }
#endif
    return Result::eOk;
}

void endPresenter()
{
    SlngxPresenter* p;
    {
        std::lock_guard<std::mutex> l(gp);
        p = gPresenter;
        if (!p)
            return;
        gPresenter = nullptr;
        gEnded.insert(p);
    }
    PLOG_WARN("sl-ngx presenter: still running at the device's end: ending it");
    teardown(p);
}

void abandonPresenter()
{
    SlngxPresenter* p;
    {
        std::lock_guard<std::mutex> l(gp);
        p = gPresenter;
        if (!p)
            return;
        gPresenter = nullptr;
        gEnded.insert(p);
    }
    // no Vulkan call and no join: the old device may be lost, and its
    // thread stuck in a call on it; slngxPresenterDestroy joins it
    {
        std::lock_guard<std::mutex> l(p->m);
        p->stop = true;
        p->lost = true;
    }
    p->cv.notify_all();
    PLOG_WARN("sl-ngx presenter: still running on a replaced device: left there");
}

} // namespace slngx

extern "C" {

SL_NGX_EXPORT VkResult slngxPresenterCreate(const SlngxPresenterDesc* desc, SlngxPresenter** out)
{
    if (!desc || !out || !desc->surface)
        return VK_ERROR_INITIALIZATION_FAILED;
    *out = nullptr;
    if (desc->version != SLNGX_PRESENTER_VERSION)
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    {
        std::lock_guard<std::mutex> l(gp);
        if (gPresenter || gCreating)
        {
            PLOG_ERROR("sl-ngx presenter: one is running already");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        gCreating = true;
    }
    struct Creating
    {
        ~Creating()
        {
            std::lock_guard<std::mutex> l(gp);
            gCreating = false;
        }
    } creating;

    auto* p = new SlngxPresenter;
    auto refuse = [&](VkResult r, const char* why) {
        PLOG_ERROR("sl-ngx presenter: %s", why);
        delete p;
        return r;
    };
    if (!slngx::currentDevice(p->instance, p->pd, p->dev))
        return refuse(VK_ERROR_INITIALIZATION_FAILED, "no device yet (slSetVulkanInfo)");
    p->desc = *desc;
    p->wantW = desc->width, p->wantH = desc->height;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(p->pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(p->pd, &nq, qf.data());
    if (desc->queueFamily >= nq || !(qf[desc->queueFamily].queueFlags & VK_QUEUE_GRAPHICS_BIT) ||
        desc->queueIndex >= qf[desc->queueFamily].queueCount)
        return refuse(VK_ERROR_INITIALIZATION_FAILED, "no such graphics queue");
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(p->pd, desc->queueFamily, desc->surface, &present);
    if (!present)
        return refuse(VK_ERROR_INITIALIZATION_FAILED, "the queue family cannot present to the surface");
    VkSurfaceCapabilitiesKHR caps{};
    VkResult r = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p->pd, desc->surface, &caps);
    if (r != VK_SUCCESS)
        return refuse(r, "no surface capabilities");
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return refuse(VK_ERROR_FEATURE_NOT_PRESENT, "the surface's images cannot be copied to");
    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(p->pd, desc->surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(p->pd, desc->surface, &nf, formats.data());
    if (std::none_of(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& f) {
            return f.format == desc->format && f.colorSpace == desc->colorSpace;
        }))
        return refuse(VK_ERROR_FORMAT_NOT_SUPPORTED, "the surface has no such format and colour space");
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(p->pd, desc->format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
        return refuse(VK_ERROR_FORMAT_NOT_SUPPORTED, "the format takes no blit");
    vkGetDeviceQueue(p->dev, desc->queueFamily, desc->queueIndex, &p->queue);
#if SLNGX_HAVE_LL2
    p->lowLatency = desc->deviceBits & SLNGX_DEVICE_LOW_LATENCY2;
#endif
#if SLNGX_HAVE_ID2
    p->id2 = desc->deviceBits & SLNGX_DEVICE_PRESENT_ID2;
#endif
#if SLNGX_HAVE_TIMING
    p->timing = (desc->deviceBits & SLNGX_DEVICE_PRESENT_TIMING) && p->id2;
#endif
#if SLNGX_HAVE_M1
    p->maint1 = desc->deviceBits & SLNGX_DEVICE_SWAPCHAIN_MAINTENANCE1;
#endif
    if (p->id2)
    {
        // the device extensions need VK_KHR_get_surface_capabilities2 on the instance
        p->surfaceCaps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
            vkGetInstanceProcAddr(p->instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
        if (!p->surfaceCaps2)
            p->id2 = p->timing = false;
    }
#if SLNGX_HAVE_LL2
    if (p->lowLatency)
    {
        p->setSleepMode = devFn<PFN_vkSetLatencySleepModeNV>(p->dev, "vkSetLatencySleepModeNV");
        p->latencySleep = devFn<PFN_vkLatencySleepNV>(p->dev, "vkLatencySleepNV");
        p->setMarker = devFn<PFN_vkSetLatencyMarkerNV>(p->dev, "vkSetLatencyMarkerNV");
        p->getTimings = devFn<PFN_vkGetLatencyTimingsNV>(p->dev, "vkGetLatencyTimingsNV");
        p->notifyOob = devFn<PFN_vkQueueNotifyOutOfBandNV>(p->dev, "vkQueueNotifyOutOfBandNV");
        p->lowLatency = p->setSleepMode && p->latencySleep && p->setMarker && p->getTimings;
    }
#endif
#if SLNGX_HAVE_TIMING
    if (p->timing)
    {
        p->setTimingQueue = devFn<PFN_vkSetSwapchainPresentTimingQueueSizeEXT>(p->dev,
            "vkSetSwapchainPresentTimingQueueSizeEXT");
        p->pastTiming = devFn<PFN_vkGetPastPresentationTimingEXT>(p->dev, "vkGetPastPresentationTimingEXT");
        p->timeDomains = devFn<PFN_vkGetSwapchainTimeDomainPropertiesEXT>(p->dev,
            "vkGetSwapchainTimeDomainPropertiesEXT");
        p->timing = p->setTimingQueue && p->pastTiming && p->timeDomains;
    }
#endif

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = desc->queueFamily;
    r = vkCreateCommandPool(p->dev, &pci, nullptr, &p->pool);
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
        VkFenceCreateInfo pfci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        for (auto& f : s.presented)
            if (r == VK_SUCCESS && p->maint1)
                r = vkCreateFence(p->dev, &pfci, nullptr, &f);
    }
    VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo tsi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    tsi.pNext = &tci;
    if (r == VK_SUCCESS)
        r = vkCreateSemaphore(p->dev, &tsi, nullptr, &p->sleepSem);
    // GPU time of the copies, where the queue family keeps timestamps
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(p->pd, &props);
    if (r == VK_SUCCESS && qf[desc->queueFamily].timestampValidBits)
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
        teardown(p);
        delete p;
        return r;
    }
#if SLNGX_HAVE_LL2
    if (p->lowLatency && p->notifyOob)
    {
        VkOutOfBandQueueTypeInfoNV oob{VK_STRUCTURE_TYPE_OUT_OF_BAND_QUEUE_TYPE_INFO_NV};
        oob.queueType = VK_OUT_OF_BAND_QUEUE_TYPE_PRESENT_NV;
        p->notifyOob(p->queue, &oob);
    }
#endif
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

namespace
{

bool validImage(const SlngxPresentImage& i)
{
    // the presenter leaves the image in its layout: a barrier can go to
    // neither of these
    return i.image && i.width && i.height && i.layout != VK_IMAGE_LAYOUT_UNDEFINED &&
        i.layout != VK_IMAGE_LAYOUT_PREINITIALIZED;
}

} // namespace

SL_NGX_EXPORT VkResult slngxPresent(SlngxPresenter* p, const SlngxPresentFrame* f, uint64_t timeoutNs)
{
    if (!p || !f || f->version != SLNGX_PRESENTER_VERSION || !validImage(f->real) || !f->ready ||
        (f->generated.image && !validImage(f->generated)))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!live(p))
        return VK_ERROR_DEVICE_LOST;
    std::unique_lock<std::mutex> l(p->m);
    auto refused = [&]() -> VkResult {
        if (p->lost)
            return VK_ERROR_DEVICE_LOST;
        if (p->failed != VK_SUCCESS)
            return p->failed;
        if (p->paused)
            return VK_NOT_READY;
        return VK_SUCCESS;
    };
    if (VkResult r = refused(); r != VK_SUCCESS)
        return r;
    // one frame waits while one is presented: more would only add latency
    auto taken = [&] { return p->queued.empty() || p->paused || p->lost || p->failed != VK_SUCCESS; };
    if (timeoutNs >= (uint64_t)INT64_MAX / 2) // beyond what a signed duration holds: unbounded
        p->cv.wait(l, taken);
    else if (!p->cv.wait_for(l, std::chrono::nanoseconds((int64_t)timeoutNs), taken))
        return VK_TIMEOUT;
    if (VkResult r = refused(); r != VK_SUCCESS)
        return r;
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
    // always: the thread is done once it has submitted, not once the GPU is
    const VkResult r = drain(p, l);
    return p->lost ? VK_ERROR_DEVICE_LOST : r;
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
    bool running = false, ended = false;
    {
        std::lock_guard<std::mutex> l(gp);
        if (p && p == gPresenter)
        {
            running = true;
            gPresenter = nullptr;
        }
        else if (p)
            ended = gEnded.erase(p) > 0;
    }
    if (running)
    {
        teardown(p);
        delete p;
        PLOG_INFO("sl-ngx presenter: ended");
    }
    else if (ended)
    {
        // ended under the application: its Vulkan objects are destroyed
        // already, or were left on a replaced device; only its thread may
        // remain, and a Reflex sleep may still wait on that device
        {
            std::unique_lock<std::mutex> l(gp);
            gpCv.wait(l, [] { return gSleepers == 0; });
        }
        if (p->thread.joinable())
            p->thread.join();
        delete p;
    }
}

SL_NGX_EXPORT VkResult slngxPresenterGetStats(SlngxPresenter* p, SlngxPresenterStats* out)
{
    if (!p || !out || out->version != SLNGX_PRESENTER_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!live(p))
        return VK_ERROR_DEVICE_LOST;
    uint32_t reports = 0;
#if SLNGX_HAVE_LL2
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
#endif
    countShown(p);
    std::lock_guard<std::mutex> l(p->m);
    *out = p->stats;
    out->latencyReports = reports;
    return VK_SUCCESS;
}

} // extern "C"
