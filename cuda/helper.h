#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <sstream>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define FUNC_NAME "cupti_MMult"

struct ParsedArgs
{
    std::string profiler = "range"; // range, pm

    // range
    std::string rangeMode = "auto";
    std::string replayMode = "user";
    uint64_t maxRange = 20;

    // pm
    int queryBaseMetrics = 0;
    int queryMetricProperties = 0;
    std::string chipName;
    uint64_t samplingInterval = 100000; // 100us
    size_t hardwareBufferSize = 512 * 1024 * 1024; // 512MB
    uint64_t maxSamples = 10000;


    // common
    int deviceIndex = 0;
    std::vector<const char*> metrics =
    {
        "gpu__time_duration.avg",
        "sm__warps_active.avg.per_cycle_active", // warp occ 1
        //"sm__warps_active.avg.peak_sustained", // warp occ 2
        //"sm__warps_active.avg.pct_of_peak_sustained_active", // warp occ 3
        "sm__ctas_active.avg.per_cycle_active", // block occ 1, should not be 100%
        //"sm__ctas_active.avg.peak_sustained", // block occ 2
        //"sm__ctas_active.avg.pct_of_peak_sustained_active", // block occ 3
        "l1tex__data_pipe_lsu_wavefronts_mem_shared",
    };
};

void PrintHelp();
ParsedArgs parseArgs(int argc, char *argv[]);