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



        // example 1:
        //"gpu__time_duration.sum",                   // total time
        //"dram__bytes_read.sum",                     // DRAM reads
        //"dram__bytes_write.sum",                    // DRAM writes
        //"lts__t_sectors_srcunit_tex_op_read.sum",   // L2 reads (sectors -- 32B)
        //"lts__t_sectors_srcunit_tex_op_write.sum",  // L2 writes (sectors -- 32B)
        //"sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", // % of peak tensor core utilization
        //"smsp__inst_executed.sum",                  // instructions

        // example 2: https://gist.github.com/getianao/1686c4d0dac02a0b91a2885e18d9c9a3
        //"smsp__thread_inst_executed_per_inst_executed", // the ratio of active threads that are not predicated off over the maximum number of threads per warp for each executed instruction
        //"l1tex__t_sector_hit_rate", // l1 cache hit rate, # of sector hits per sector
        //"lts__t_sector_hit_rate", // l2 cache hit rate
        //"sm__warps_active.avg.pct_of_peak_sustained_active", // achieved occupancy
        //"sm__maximum_warps_per_active_cycle_pct", // max occupancy
        //"smsp__maximum_warps_avg_per_active_cycle", // theoratical warp per scheduler
        //"smsp__warps_active", // active warps per scheduler
        //"smsp__warps_eligible", // eligible warps per scheduler
    };
};

void PrintHelp();
ParsedArgs parseArgs(int argc, char *argv[]);