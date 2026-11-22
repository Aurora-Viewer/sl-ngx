# Build: make DLSS_SDK=/path/to/the/NVIDIA/DLSS/SDK
# The SDK is NVIDIA's, under its own licence; nothing of it is in this repository.
DLSS_SDK ?= $(error set DLSS_SDK to the folder holding NVIDIA's DLSS SDK (include/, lib/))
CXX ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -fPIC -fvisibility=hidden -Wall -Wno-deprecated-declarations
NGX_LIB = $(DLSS_SDK)/lib/Linux_x86_64/libnvsdk_ngx.a

all: libsl_ngx.so

libsl_ngx.so: src/sl_ngx.cpp exports.map $(wildcard include/sl/*.h)
	$(CXX) $(CXXFLAGS) -shared -Iinclude -I$(DLSS_SDK)/include src/sl_ngx.cpp $(NGX_LIB) \
		-Wl,--exclude-libs,ALL -Wl,--version-script=exports.map -Wl,--no-undefined \
		-lvulkan -ldl -o $@

check: libsl_ngx.so
	tests/check-exports.sh libsl_ngx.so
	tests/check-no-ngx-definitions.sh
	$(CXX) -std=c++17 -Iinclude tests/layout.cpp -o tests/layout && tests/layout | diff -u tests/layout.expected -

clean:
	rm -f libsl_ngx.so tests/layout tests/smoke

.PHONY: all check clean

tests/smoke-build: libsl_ngx.so
	$(CXX) -std=c++17 -Wall -Wno-deprecated-declarations -Iinclude tests/smoke.cpp -L. -lsl_ngx \
		-Wl,-rpath,'$$ORIGIN/..' -lvulkan -o tests/smoke

.PHONY: tests/smoke-build
