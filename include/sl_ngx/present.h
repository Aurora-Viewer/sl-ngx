// SPDX-License-Identifier: MIT
//
// sl-ngx's presenter: frame generation shown without swapchain hooks.
//
// Streamline presents generated frames by hooking the application's
// swapchain. sl-ngx has no hooks; instead the application hands the
// presenter a surface it does not present to itself, and once per rendered
// frame the generated frame and the rendered one. A thread of the presenter,
// on a queue of its own, copies each into its swapchain and presents the
// generated frame first and the rendered one half a frame later, with
// VK_NV_low_latency2 on that swapchain so that Reflex (slReflexSleep,
// slReflexSetOptions, slPCLSetMarker) works as it does with Streamline.
//
// These functions are sl-ngx's own, not Streamline's: find them with
// dlsym/GetProcAddress, so an older sl-ngx without them is told apart.

#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLNGX_PRESENTER_VERSION 1

typedef struct SlngxPresenter SlngxPresenter;

// What the application enabled on its device for the presenter: the
// extension and its feature both.
typedef enum SlngxPresenterDeviceBits
{
    SLNGX_DEVICE_LOW_LATENCY2 = 1 << 0,  // VK_NV_low_latency2 (and VK_KHR_present_id, presentId)
    SLNGX_DEVICE_PRESENT_ID2 = 1 << 1,   // VK_KHR_present_id2, presentId2
    SLNGX_DEVICE_PRESENT_WAIT2 = 1 << 2, // VK_KHR_present_wait2, presentWait2
    SLNGX_DEVICE_PRESENT_TIMING = 1 << 3 // VK_EXT_present_timing, presentTiming (and VK_KHR_swapchain_maintenance1)
} SlngxPresenterDeviceBits;

typedef struct SlngxPresenterDesc
{
    uint32_t version;           // SLNGX_PRESENTER_VERSION
    VkSurfaceKHR surface;       // the application's; no other swapchain on it while the presenter lives
    uint32_t queueFamily;       // a queue no one else submits to or presents on
    uint32_t queueIndex;
    uint32_t deviceBits;        // SlngxPresenterDeviceBits
    uint32_t width, height;     // the swapchain size, where the surface leaves it to the application
    VkFormat format;            // the swapchain format; it must support VK_FORMAT_FEATURE_BLIT_DST_BIT
    VkColorSpaceKHR colorSpace;
    uint32_t minImageCount;     // 0: three
} SlngxPresenterDesc;

typedef struct SlngxPresentImage
{
    VkImage image;              // VK_IMAGE_USAGE_TRANSFER_SRC_BIT, format with VK_FORMAT_FEATURE_BLIT_SRC_BIT
    VkImageLayout layout;       // its layout when `ready` is reached; the presenter leaves it in that layout
    uint32_t width, height;
} SlngxPresentImage;

// One rendered frame. Both images must stay untouched by the application
// from the moment `ready` is signalled until `released` is: they are read
// on the presenter's queue in between. Keep at least three sets in a ring.
typedef struct SlngxPresentFrame
{
    uint32_t version;           // SLNGX_PRESENTER_VERSION
    uint32_t frame;             // the frame token's value, as given to slReflexSleep and slPCLSetMarker
    SlngxPresentImage generated; // image VK_NULL_HANDLE: the rendered frame alone
    SlngxPresentImage real;
    VkSemaphore ready;          // a timeline the application signals once both images are written
    uint64_t readyValue;
    VkSemaphore released;       // a timeline the presenter signals, in frame order, once it has read both
    uint64_t releasedValue;
} SlngxPresentFrame;

typedef struct SlngxPresenterStats
{
    uint32_t version;           // SLNGX_PRESENTER_VERSION
    uint64_t framesAccepted;    // by slngxPresent
    uint64_t framesSkipped;     // released unread (their ready came late, or the presenter paused or stopped)
    uint64_t generatedPresented, realPresented;
    uint64_t generatedShown, realShown; // with present timing: the frames the display started to show
    uint32_t presentTiming;     // 1: the shown counts are measured
    uint32_t lowLatency;        // 1: VK_NV_low_latency2 is on the swapchain (Reflex can work)
    uint32_t latencyReports;    // frame reports vkGetLatencyTimingsNV returned at this call
    uint32_t swapchainsMade;
    uint32_t width, height;     // the swapchain's
    float copyGpuMs;            // the GPU time of the presenter's copies for one rendered frame, averaged
    float intervalMs;           // between rendered frames as they reached `ready`, averaged
    float presentHalfMs;        // the wait between the generated frame's present and the rendered one's
    int32_t lastError;          // the last VkResult that stopped a frame, VK_SUCCESS if none
    uint32_t deviceLost;        // 1: the presenter stopped on a lost device
} SlngxPresenterStats;

// Makes the swapchain and starts the presenter's thread. One presenter at
// a time, on the device given to slSetVulkanInfo.
VkResult slngxPresenterCreate(const SlngxPresenterDesc* desc, SlngxPresenter** presenter);

// Hands over one rendered frame and returns at once, or once the frame
// before it has been taken up (at most `timeoutNs`). VK_TIMEOUT: not
// taken, the images and the release stay the caller's. VK_NOT_READY: paused.
VkResult slngxPresent(SlngxPresenter* presenter, const SlngxPresentFrame* frame, uint64_t timeoutNs);

// The size the swapchain is made at next, where the surface leaves it to
// the application. The swapchain is remade before the next frame.
void slngxPresenterResize(SlngxPresenter* presenter, uint32_t width, uint32_t height);

// Returns once the presenter's queue is idle and every frame handed over
// is released; until slngxPresenterResume it submits nothing. Call it
// around anything that waits for the whole device (vkDeviceWaitIdle: a
// wgpu surface configured or dropped).
VkResult slngxPresenterPause(SlngxPresenter* presenter);
void slngxPresenterResume(SlngxPresenter* presenter);

// Stops the thread, releases every frame handed over, waits for its queue
// and destroys the swapchain. The surface stays the application's.
// slShutdown, and slSetVulkanInfo with another device, end a presenter
// still running; its handle then answers VK_ERROR_DEVICE_LOST and Destroy
// does nothing. Call those from the thread that calls the presenter.
void slngxPresenterDestroy(SlngxPresenter* presenter);

VkResult slngxPresenterGetStats(SlngxPresenter* presenter, SlngxPresenterStats* stats);

typedef VkResult (*PFN_slngxPresenterCreate)(const SlngxPresenterDesc*, SlngxPresenter**);
typedef VkResult (*PFN_slngxPresent)(SlngxPresenter*, const SlngxPresentFrame*, uint64_t);
typedef void (*PFN_slngxPresenterResize)(SlngxPresenter*, uint32_t, uint32_t);
typedef VkResult (*PFN_slngxPresenterPause)(SlngxPresenter*);
typedef void (*PFN_slngxPresenterResume)(SlngxPresenter*);
typedef void (*PFN_slngxPresenterDestroy)(SlngxPresenter*);
typedef VkResult (*PFN_slngxPresenterGetStats)(SlngxPresenter*, SlngxPresenterStats*);

#ifdef __cplusplus
}
#endif
