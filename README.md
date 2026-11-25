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

- `include/sl/`: twelve of Streamline's public headers (v2.14.16), unchanged,
  under NVIDIA's MIT licence (`include/sl/LICENSE-Streamline.txt`).
- `src/sl_ngx.cpp`: the implementation, MIT (`LICENSE`).
- `tests/`: a struct layout probe, export and source checks, and a GPU smoke test.

Nothing of the NVIDIA DLSS SDK is in this repository. You build the library
against your own copy of the SDK, which you download from NVIDIA and use
under NVIDIA's licence. The built library contains NVIDIA's NGX core: keep
it for your own use and do not redistribute it unless NVIDIA's licence
allows it.

## Build

You need g++, make, the Vulkan headers and loader, and the
[NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS).

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

## What is implemented

`slInit`, `slShutdown`, `slSetVulkanInfo`, `slGetFeatureRequirements`,
`slIsFeatureSupported`, `slGetFeatureVersion`, `slGetFeatureFunction`,
`slGetNewFrameToken`, `slSetConstants`, `slSetTagForFrame`,
`slAllocateResources`, `slFreeResources` and `slEvaluateFeature`, for
`kFeatureDLSS`, `kFeatureDLSS_RR` and `kFeatureDLSS_G` (frame generation,
below). Through `slGetFeatureFunction`:
`slDLSSGetOptimalSettings`, `slDLSSGetState`, `slDLSSSetOptions` and their
`slDLSSD*` counterparts, so the inline helpers in `sl_dlss.h` and
`sl_dlss_d.h` work as they do with Streamline.

Differences from Streamline:

- Vulkan only, manual hooking only. Linux is tested; Windows is written but untested. There is no interposer, no swapchain or
  present hooks, no D3D, no Reflex or other features.
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
