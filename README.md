# sl-ngx

NVIDIA DLSS, DLSS Ray Reconstruction and DLSS Frame Generation for Vulkan
on Linux, behind NVIDIA Streamline's public interface.

[Streamline](https://github.com/NVIDIA-RTX/Streamline) is NVIDIA's
integration layer for DLSS. Its public headers are MIT licensed, but NVIDIA
ships Streamline for Windows only. This library implements the part of that
interface an application needs for DLSS Super Resolution, Ray
Reconstruction and Frame Generation, on top of NVIDIA NGX, so a program
written against Streamline's headers can use DLSS on Linux.

## What is in this repository

- `include/sl/`: fourteen of Streamline's public headers (v2.14.16), unchanged,
  under NVIDIA's MIT licence (`include/sl/LICENSE-Streamline.txt`).
- `include/sl_ngx/present.h`: sl-ngx's own presenter for frame generation, MIT.
- `src/`: the implementation, MIT (`LICENSE`).
- `tests/`: a struct layout probe, export and source checks, a GPU smoke test
  and a presenter test.

Nothing of the NVIDIA DLSS SDK is in this repository. You build the library
against your own copy of the SDK, which you download from NVIDIA and use
under NVIDIA's licence. The built library contains NVIDIA's NGX core: keep
it for your own use and do not redistribute it unless NVIDIA's licence
allows it.

## Build

You need g++, make, the Vulkan headers and loader, and the
[NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS). The library builds
with older Vulkan headers too (`tests/present` needs current ones): the presenter's parts that need `VK_NV_low_latency2`,
`VK_KHR_present_id2` or `VK_EXT_present_timing` are left out where the
headers lack them (without `VK_NV_low_latency2`, no Reflex or PCL).

```bash
make DLSS_SDK=/path/to/DLSS
make check DLSS_SDK=/path/to/DLSS
```

CMake works too (`cmake -B build -DDLSS_SDK=/path/to/DLSS && cmake --build build`).
It also has a Windows (MSVC) recipe that builds `sl_ngx.dll` against the
SDK's `nvsdk_ngx_d.lib`; that build has not been tried yet.

Put `libsl_ngx.so` in a folder together with the SDK's
`lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*`. NGX loads the models from
the folders given in `Preferences::pathsToPlugins`, or from the library's own
folder when there are none. Recent NVIDIA drivers ship a Ray Reconstruction
model of their own (`libnvidia-ngx-builtin-dlssd.so`); the SDK's
`libnvidia-ngx-dlssd.so.*` in the same folder is used where they don't.
Frame generation needs the SDK's `libnvidia-ngx-dlssg.so.*` there too;
without it `slIsFeatureSupported(kFeatureDLSS_G)` says it is not supported.

To check it on your GPU:

```bash
make tests/smoke-build DLSS_SDK=/path/to/DLSS
tests/smoke /folder/with/the/models        # DLSS
tests/smoke /folder/with/the/models rr     # Ray Reconstruction
tests/smoke /folder/with/the/models abandon  # DLSS again on a second device
tests/smoke /folder/with/the/models fg     # Frame Generation
```

The presenter test needs SDL3 and opens a window; with the Vulkan
validation layer installed it runs under synchronisation and thread-safety
validation and fails on any error:

```bash
make tests/present-build DLSS_SDK=/path/to/DLSS
tests/present
```

## What is implemented

`slInit`, `slShutdown`, `slSetVulkanInfo`, `slGetFeatureRequirements`,
`slIsFeatureSupported`, `slGetFeatureVersion`, `slGetFeatureFunction`,
`slGetNewFrameToken`, `slSetConstants`, `slSetTagForFrame`,
`slAllocateResources`, `slFreeResources` and `slEvaluateFeature`, for
`kFeatureDLSS`, `kFeatureDLSS_RR` and `kFeatureDLSS_G` (frame generation,
below). Through `slGetFeatureFunction`:
`slDLSSGetOptimalSettings`, `slDLSSGetState`, `slDLSSSetOptions` and their
`slDLSSD*` counterparts, so the inline helpers in `sl_dlss.h` and
`sl_dlss_d.h` work as they do with Streamline. `kFeatureReflex`
(`slReflexSetOptions`, `slReflexSleep`, `slReflexGetState`) and
`kFeaturePCL` (`slPCLSetMarker`) work on the presenter's swapchain (below).

Differences from Streamline:

- Vulkan only, manual hooking only. Linux is tested; Windows is written but untested. There is no interposer, no swapchain or
  present hooks, no D3D, and no features beyond the ones above.
- Frame generation is evaluated by the application, as DLSS is: there is
  no swapchain hook to present the generated frame, and no
  `sl_dlss_g.h` or its options. Once per rendered frame, tag
  `kBufferTypeHUDLessColor` (the frame at display size, without the
  interface), depth and motion vectors (as for DLSS), then call
  `slEvaluateFeature(kFeatureDLSS_G, …)` with the viewport and, as
  `ResourceTag`s in its inputs, the output (`kBufferTypeScalingOutputColor`,
  a storage image) and optionally the frame as shown, with its interface
  (`kBufferTypeScalingInputColor`; NGX then finds the interface where the
  two frames differ and carries it into the generated frame as it is,
  instead of interpolating it). Present the output, then the rendered
  frame. One generated frame per rendered frame, colour buffers in LDR.
  The feature is made at the first evaluate and again when the sizes or
  format change; `slAllocateResources` does nothing for it.
- Instead of Streamline's swapchain hooks and pacer, sl-ngx has a presenter
  of its own (`include/sl_ngx/present.h`, found with `dlsym`): the
  application gives it a surface it does not present to itself and, once
  per rendered frame, the generated frame and the rendered one. On a queue
  of its own (`slGetFeatureRequirements(kFeatureDLSS_G)` asks for one more
  graphics queue) it copies the generated frame into its swapchain and
  presents it, then the rendered one half a frame later, in FIFO. The
  requirements name the device extensions it uses where the device has
  them; the application enables each with its feature
  (`presentId`, `presentId2`, `presentWait2`, `presentTiming`,
  `swapchainMaintenance1`) and says which in `SlngxPresenterDesc::deviceBits`.
  `presentId2` and `presentTiming` are used where the surface supports them;
  `presentWait2` is not used yet. The calling contract (queue families,
  formats, frame numbers, error codes) is in `present.h`.
- Reflex and PCL need the presenter: they use `VK_NV_low_latency2` on its
  swapchain, with the present id `2 × frame + 2` (the rendered frame's
  present), and do nothing without one. `slReflexSleep` returns at once
  while Reflex is off with no frame limit. `ReflexOptions::frameLimitUs`
  becomes the swapchain's minimum frame interval, which the driver applies
  per rendered frame (between Reflex sleeps), not per present:
  `tests/present` measures 100 ms rendered frames with both frames of each
  presented.
- `slSetVulkanInfo` may be called with an instance and physical device but no
  device. `slGetFeatureRequirements` then also reports the device extensions
  NGX needs on that device, so the application can create its device with
  them (Streamline makes a device of its own for this).
  A call without a device leaves NGX running on the current one.
- `slSetVulkanInfo` with a different device while NGX runs (a new device
  after a device loss) leaves NGX on the old device without calling it there
  and starts it on the new one. Keep the old `VkDevice` alive: NGX still
  holds objects on it. Call `slShutdown` instead to end NGX on a device that
  still works and is idle.
- Resources are never copied: a tag must be `eValidUntilEvaluate` or
  `eValidUntilPresent` and stay valid until the evaluate it is used in.
  `eOnlyValidNow` is refused.
- Tags need the Vulkan image, its view, size and format.
- `slFreeResources` releases the feature at once: the GPU must be done with
  the viewport's last evaluate. A feature replaced inside `slEvaluateFeature`
  (a mode or size change) is released four frame tokens later.

The mapping from Streamline's fields to NGX parameters follows Streamline's
own MIT DLSS and DLSS-D plugins.

## Licence

MIT for this repository's own code; Streamline's headers are MIT under
NVIDIA's copyright. NVIDIA, DLSS and Streamline are trademarks of NVIDIA
Corporation. This project is not affiliated with or endorsed by NVIDIA.
