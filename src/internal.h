// SPDX-License-Identifier: MIT
//
// What sl_ngx.cpp and presenter.cpp share.

#pragma once

#include <vulkan/vulkan.h>

#include "sl/sl.h"
#include "sl/sl_reflex.h"

namespace slngx
{

// The device slSetVulkanInfo gave, or false when there is none.
bool currentDevice(VkInstance& instance, VkPhysicalDevice& physicalDevice, VkDevice& device);

void log(sl::LogType type, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

// Reflex and PCL, on the presenter's swapchain (no-ops without one).
sl::Result reflexGetState(sl::ReflexState& state);
sl::Result reflexSleep(const sl::FrameToken& frame);
sl::Result reflexSetOptions(const sl::ReflexOptions& options);
sl::Result pclSetMarker(sl::PCLMarker marker, const sl::FrameToken& frame);

// Whether the physical device has the extension.
bool deviceHasExtension(VkPhysicalDevice physicalDevice, const char* name);
bool instanceHasExtension(const char* name);

// A presenter left running at slShutdown: stopped and its Vulkan objects
// destroyed (the device still works).
void endPresenter();
// A presenter left running on a replaced device: stopped without a Vulkan
// call or a join (the old device may be lost); its objects leak there.
void abandonPresenter();

} // namespace slngx
