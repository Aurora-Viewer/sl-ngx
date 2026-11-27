# Build: make DLSS_SDK=/path/to/the/NVIDIA/DLSS/SDK
# The SDK is NVIDIA's, under its own licence; nothing of it is in this repository.
DLSS_SDK ?= $(error set DLSS_SDK to the folder holding NVIDIA's DLSS SDK (include/, lib/))
CXX ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -fPIC -fvisibility=hidden -Wall -Wno-deprecated-declarations
NGX_LIB = $(DLSS_SDK)/lib/Linux_x86_64/libnvsdk_ngx.a

all: libsl_ngx.so

SOURCES = src/sl_ngx.cpp src/presenter.cpp

libsl_ngx.so: $(SOURCES) src/internal.h exports.map $(wildcard include/sl/*.h include/sl_ngx/*.h)
	$(CXX) $(CXXFLAGS) -shared -pthread -Iinclude -I$(DLSS_SDK)/include $(SOURCES) $(NGX_LIB) \
		-Wl,--exclude-libs,ALL -Wl,--version-script=exports.map -Wl,--no-undefined \
		-lvulkan -ldl -o $@

check: libsl_ngx.so
	tests/check-exports.sh libsl_ngx.so
	tests/check-no-ngx-definitions.sh
	$(CXX) -std=c++17 -Wno-deprecated-declarations -Iinclude tests/layout.cpp -o tests/layout && tests/layout | diff -u tests/layout.expected -

clean:
	rm -f libsl_ngx.so tests/layout tests/smoke tests/present

.PHONY: all check clean

tests/smoke-build: libsl_ngx.so
	$(CXX) -std=c++17 -Wall -Wno-deprecated-declarations -Iinclude tests/smoke.cpp -L. -lsl_ngx \
		-Wl,-rpath,'$$ORIGIN/..' -lvulkan -o tests/smoke

.PHONY: tests/smoke-build

# needs SDL3 (a window to present to)
tests/present-build: libsl_ngx.so
	$(CXX) -std=c++17 -Wall -Wno-deprecated-declarations -Iinclude tests/present.cpp -L. -lsl_ngx \
		-Wl,-rpath,'$$ORIGIN/..' $$(pkg-config --cflags --libs sdl3) -lvulkan -o tests/present

.PHONY: tests/present-build
