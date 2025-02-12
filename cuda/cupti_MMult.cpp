#include <stdio.h>
// #include <malloc.h>
#include "parameters.h"
#include <stdlib.h>
#include <string.h>
#include <cassert>

#include "helper.h"

// CUDA runtime
#include <pm_sampling.h>
#include <range_profiling.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>
#include <thread>

#define DEDICATED_COUNTER_THREAD false
#define RECORD_CUDA_EVENT false

void REF_MMult(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult(cublasHandle_t, int, int, int, float *, int, float *, int,
              float *, int);
void copy_matrix(int, int, float *, int, float *, int);
void random_matrix(int, int, float *, int);
float compare_matrices(int, int, float *, int, float *, int);

double dclock();

int main(int argc, char* argv[]) {
  /* CUPTI Setup */
  ParsedArgs args = parseArgs(argc, argv);
  DRIVER_API_CALL(cuInit(0));
  std::string chipName = args.chipName;
  std::vector<uint8_t> counterAvailibilityImage;
  CUdevice cuDevice;
  CUcontext cuContext;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;

  CuptiPmProfilerHost pmSamplingHost;
  CuptiPmSampling cuptiPmSamplingTarget;
  CuptiProfilerHostPtr pCuptiProfilerHost;
  RangeProfilerTargetPtr pRangeProfilerTarget;

  assert (args.deviceIndex >= 0);

  printf("device index: %d\n", args.deviceIndex);
  DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));
  DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
  PmSamplingDeviceSupportStatus(cuDevice);

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cuDevice);
  std::cout << "Number of SMs: " << numSMs << std::endl;
  CuptiPmSampling::GetChipName(args.deviceIndex, chipName);
  CuptiPmSampling::GetCounterAvailabilityImage(args.deviceIndex, counterAvailibilityImage);
  printf("chip name: %s\n", chipName.c_str());
  //printf("Counter Availibility Image:\n");
  //for (auto v : counterAvailibilityImage){
  //  printf("%d, ", v);
  //}
  int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
  DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

  if (computeCapabilityMajor < 7)
  {
      std::cerr << "Range Profiling is supported only on devices with compute capability 7.0 and above" << std::endl;
      exit(EXIT_FAILURE);
  }

  //CuptiPmSampling::GetChipName(args.deviceIndex, chipName);

  if (args.profiler == "pm"){
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);
    CUPTI_API_CALL(pmSamplingHost.CreateConfigImage(args.metrics, configImage));

    cuptiPmSamplingTarget.SetUp(args.deviceIndex);

    // 1. Enable PM sampling and set config for the PM sampling data collection. create m_pmSamplerObject
    CUPTI_API_CALL(cuptiPmSamplingTarget.EnablePmSampling(args.deviceIndex));
    CUPTI_API_CALL(cuptiPmSamplingTarget.SetConfig(configImage, args.hardwareBufferSize, args.samplingInterval));
    // 2. Create counter data image
    std::cout << "Before create counter data: " << counterDataImage.size() << std::endl;
    CUPTI_API_CALL(cuptiPmSamplingTarget.CreateCounterDataImage(args.maxSamples, args.metrics, counterDataImage));
    std::cout << "After create counter data: " << counterDataImage.size() << std::endl;
    CUPTI_API_CALL(cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage));
    //printf("Counter Data Image:\n");
    //for (auto v : counterDataImage){
    //  printf("%d, ", v);
    //}
  }
  else if (args.profiler == "range"){
    RangeProfilerConfig config;
    config.maxNumOfRanges = args.maxRange;
    config.minNestingLevel = 1;
    config.numOfNestingLevel = args.rangeMode == "user" ? 2 : 1;
    pCuptiProfilerHost = std::make_shared<CuptiRangeProfilerHost>();

    pRangeProfilerTarget = std::make_shared<RangeProfilerTarget>(cuContext, config);

    pCuptiProfilerHost->SetUp(chipName, counterAvailibilityImage);
    CUPTI_API_CALL(pCuptiProfilerHost->CreateConfigImage(args.metrics, configImage));

    // Enable Range profiler
    CUPTI_API_CALL(pRangeProfilerTarget->EnableRangeProfiler());

    // Create CounterData Image
    CUPTI_API_CALL(pRangeProfilerTarget->CreateCounterDataImage(args.metrics, counterDataImage));

    // Set range profiler configuration
    printf("Range Mode: %s\n", args.rangeMode.c_str());
    printf("Replay Mode: %s\n", args.replayMode.c_str());
    CUPTI_API_CALL(pRangeProfilerTarget->SetConfig(
        args.rangeMode == "auto" ? CUPTI_AutoRange : CUPTI_UserRange,
        args.replayMode == "kernel" ? CUPTI_KernelReplay : CUPTI_UserReplay,
        configImage,
        counterDataImage
    ));

  }

  if (args.queryBaseMetrics || args.queryMetricProperties)
  {
      return PmSamplingQueryMetrics(chipName, counterAvailibilityImage, args.queryBaseMetrics, args.queryMetricProperties, args.metrics);
  }


  // 3. Launch the decode thread
  DecodeThread decodeThread;
  CUptiResult threadFuncResult;

  //// print gpu info
  //cudaDeviceProp deviceProp;
  //int devID = 0;
  //checkCudaErrors(cudaSetDevice(devID));
  //auto error = cudaGetDeviceProperties(&deviceProp, devID);
  //if (error != cudaSuccess) {
  //  printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
  //         __LINE__);
  //  exit(EXIT_FAILURE);
  //}
  //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
  //       deviceProp.name, deviceProp.major, deviceProp.minor);

  int p, m, n, k, rep;

  double dtime, dtime_best, gflops, diff;

  float *a, *b, *c, *cref, *cold;

  printf("MY_MMult = [\n");

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));
  // checkCudaErrors(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  /* Time the "optimized" implementation */
  cudaEvent_t start, stop;
  // Allocate CUDA events that we'll use for timing
  if (RECORD_CUDA_EVENT){
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
  }

  std::vector<uint8_t> counterDataImageCopy = counterDataImage;
  /* TEST LOOPS */
  for (p = PFIRST; p <= PLAST; p += PINC) {
    m = (SIZE_M == -1 ? p : SIZE_M);
    n = (SIZE_N == -1 ? p : SIZE_N);
    k = (SIZE_K == -1 ? p : SIZE_K);

    gflops = 2.0 * m * n * k * 1.0e-09;

    const int lda = k, ldb = n, ldc = n;

    /* Allocate CPU space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    a = (float *)malloc(mem_size_A);
    b = (float *)malloc(mem_size_B);
    c = (float *)malloc(mem_size_C);
    cold = (float *)malloc(mem_size_C);
    cref = (float *)malloc(mem_size_C);

    /* Generate random matrices A, B, Cold */
    random_matrix(m, k, a, m);
    random_matrix(k, n, b, k);
    random_matrix(m, n, cold, n);
    memset(cold, 0, mem_size_C);
    memset(cref, 0, mem_size_C);

    /* Init device matrix*/
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, a, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, b, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    /* Run the reference implementation so the answers can be compared */
    REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);


    float msecTotal = 0.0f;
    double nsecTotal = 0.0f;
    do {
      // 4. Start the PM sampling and launch the CUDA workload
      if (args.profiler == "pm"){
        CUPTI_API_CALL(cuptiPmSamplingTarget.StartPmSampling());
        if (DEDICATED_COUNTER_THREAD)
          decodeThread.start(std::ref(counterDataImage), 
                      std::ref(args.metrics), 
                      std::ref(cuptiPmSamplingTarget), 
                      std::ref(pmSamplingHost)
                      );
      }
      else if (args.profiler == "range"){
        CUPTI_API_CALL(pRangeProfilerTarget->StartRangeProfiler());
        std::cout << "pushed MMult range" << std::endl;
        CUPTI_API_CALL(pRangeProfilerTarget->PushRange("MMult"));
      }


      // Record the start event
      if (RECORD_CUDA_EVENT)
        checkCudaErrors(cudaEventRecord(start, NULL));

      for (rep = 0; rep < NREPEATS; rep++) {
        /* Time your implementation */
        MY_MMult(handle, m, n, k, d_A, k, d_B, n, d_C, n);
      }

      // Record the stop event
      if (RECORD_CUDA_EVENT){
        checkCudaErrors(cudaEventRecord(stop, NULL));
        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop)); // no need cudaDeviceSynchronize here
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
      }

      // 5. Stop the PM sampling and join the decode thread
      if (args.profiler == "pm"){
        CUPTI_API_CALL(cuptiPmSamplingTarget.StopPmSampling());
        if (DEDICATED_COUNTER_THREAD)
          decodeThread.join();
        else
          DecodeCounterDataSync(std::ref(counterDataImage), 
                      std::ref(args.metrics), 
                      std::ref(cuptiPmSamplingTarget), 
                      std::ref(pmSamplingHost),
                      std::ref(threadFuncResult)
          );
        std::cout << "counter not changed: " << (counterDataImage == counterDataImageCopy) << std::endl;
        std::cout << "counter eval error: " << (counterDataImage.data() == counterDataImageCopy.data()) << std::endl;
        // 6. Print the sample ranges for the collected metrics
        pmSamplingHost.PrintSampleRanges();
      }
      else if (args.profiler == "range"){
        CUPTI_API_CALL(pRangeProfilerTarget->PopRange());
        std::cout << "pop MMult range" << std::endl;
        CUPTI_API_CALL(pRangeProfilerTarget->StopRangeProfiler());
      }
    } while(args.profiler == "range" and !pRangeProfilerTarget->IsAllPassSubmitted()); // TODO: check all pass for pm also

    if (args.profiler == "pm"){
      // TODO: move DecodeCounterData here
    }
    if (args.profiler == "range"){
      // Get Profiler Data
      CUPTI_API_CALL(pRangeProfilerTarget->DecodeCounterData());

      // Evaluate the results
      size_t numRanges = 0;
      CUPTI_API_CALL(pCuptiProfilerHost->GetNumOfRanges(counterDataImage, numRanges));
      for (size_t rangeIndex = 0; rangeIndex < numRanges; ++rangeIndex)
      {
          CUPTI_API_CALL(pCuptiProfilerHost->EvaluateCounterData(rangeIndex, args.metrics, counterDataImage));
      }

      pCuptiProfilerHost->PrintProfilerRanges(nsecTotal);
      msecTotal = nsecTotal/1000000;

    }


    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / NREPEATS;
    double flopsPerMatrixMul = 2.0 * m * k * n;
    double gflops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(cold, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    diff = compare_matrices(m, n, cold, ldc, cref, ldc);
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    printf("%d %.2f %le \n", p, gflops, diff);


    /* Free Memory */
    free(a);
    free(b);
    free(c);
    free(cold);
    free(cref);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
  }

  if (args.profiler == "pm"){
    // 7. Clean up
    cuptiPmSamplingTarget.TearDown();
    pmSamplingHost.TearDown();
  }
  else if (args.profiler == "range"){
    CUPTI_API_CALL(pRangeProfilerTarget->DisableRangeProfiler());
    pCuptiProfilerHost->TearDown();
  }

  // Destroy the handle
  checkCudaErrors(cublasDestroy(handle));
  DRIVER_API_CALL(cuCtxDestroy(cuContext));

  printf("];\n");
  return 0;
}
