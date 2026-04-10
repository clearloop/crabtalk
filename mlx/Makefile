# Makefile for the CrabllmMlx Swift static library.
#
# Targets:
#   build  — swift build -c release
#   test   — build, then compile and run tests/smoke.c against the
#            static library to verify the C ABI round-trips.
#   clean  — remove SwiftPM build artifacts.
#
# The smoke test is the ground truth for the C<->Swift contract during
# Phase 2. Phase 3's Rust crate will exercise the same library through
# its own build.rs.

.PHONY: build test clean

SWIFT := swift
CC := clang++
SDK := $(shell xcrun --show-sdk-path)
SWIFT_LIB := /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx
SMOKE_BIN := .build/smoke

build:
	$(SWIFT) build -c release

test: build
	$(CC) -Wall -Wextra -Werror -x c -o $(SMOKE_BIN) tests/smoke.c \
		-L.build/release -lCrabllmMlx \
		-L$(SDK)/usr/lib/swift \
		-L$(SWIFT_LIB) \
		-lswiftCompatibility56 \
		-lswiftCompatibilityConcurrency \
		-lswiftCompatibilityPacks \
		-framework Foundation \
		-framework Metal \
		-framework MetalPerformanceShaders \
		-framework MetalPerformanceShadersGraph \
		-framework Accelerate \
		-framework CoreGraphics \
		-framework CoreImage
	$(SMOKE_BIN)

clean:
	rm -rf .build
