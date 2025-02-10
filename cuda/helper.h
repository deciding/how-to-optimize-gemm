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
    int deviceIndex = 0;
    int queryBaseMetrics = 0;
    int queryMetricProperties = 0;
    std::string chipName;
    uint64_t samplingInterval = 100000; // 100us
    size_t hardwareBufferSize = 512 * 1024 * 1024; // 512MB
    uint64_t maxSamples = 10000;
    std::vector<const char*> metrics =
    {
        "gr__cycles_active.avg",                            // Active Cycles
        "gr__cycles_elapsed.max",                           // Elapsed Cycles
        "gpu__time_duration.sum",                           // Duration
        "sm__inst_executed_realtime.avg.per_cycle_active",  // Inst Executed per Active Cycle
        "sm__cycles_active.avg",                             // SM Active Cycles
        "dram__read_throughput.max.pct_of_peak_sustained_elapsed"                             // SM Active Cycles
    };
};

void PrintHelp();
ParsedArgs parseArgs(int argc, char *argv[]);