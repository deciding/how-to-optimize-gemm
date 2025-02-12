#include "pm_sampling.h"

void PmSamplingDeviceSupportStatus(CUdevice device)
{
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_PM_SAMPLING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << device << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }

        exit(EXIT_WAIVED);
    }
}

int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, int queryBaseMetrics, int queryMetricProperties, std::vector<const char*>& metrics)
{
    CuptiPmProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    if (queryBaseMetrics)
    {
        std::vector<std::string> baseMetrics;
        CUPTI_API_CALL(pmSamplingHost.GetSupportedBaseMetrics(baseMetrics));
        printf("Base Metrics:\n");
        for (const auto& metric : baseMetrics)
        {
            printf("  %s\n", metric.c_str());
        }
        return 0;
    }

    if (queryMetricProperties)
    {
        for (const auto& metricName : metrics)
        {
            std::vector<std::string> subMetrics;
            CUpti_MetricType metricType;
            CUPTI_API_CALL(pmSamplingHost.GetSubMetrics(metricName, subMetrics, metricType));
            const char* metricTypeName = metricType == CUPTI_METRIC_TYPE_COUNTER ? "Counter" : (metricType == CUPTI_METRIC_TYPE_RATIO) ? "Ratio" : "Throughput";
            printf("Sub Metrics for type %s metric %s:\n", metricTypeName, metricName);
            for (const auto& metric : subMetrics) {
                printf("  %s\n", metric.c_str());
            }

            std::string metricDescription;
            CUPTI_API_CALL(pmSamplingHost.GetMetricProperties(metricName, metricType,metricDescription));

            printf("Metric Description: %s\n", metricDescription.c_str());
            printf("\n");
        }
        return 0;
    }

    pmSamplingHost.TearDown();
    return 0;
}

void DecodeCounterData( std::vector<uint8_t>& counterDataImage,
                        std::vector<const char*> metricsList,
                        CuptiPmSampling& cuptiPmSamplingTarget,
                        CuptiPmProfilerHost& pmSamplingHost,
                        CUptiResult& result,
                        std::atomic<bool>& stopDecodeThread)
{
    while (!stopDecodeThread)
    {
        const char *errstr;
        result = cuptiPmSamplingTarget.DecodePmSamplingData(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "DecodePmSamplingData failed with error " << errstr << std::endl;
            return;
        }

        CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        counterDataInfo.pCounterDataImage = counterDataImage.data();
        counterDataInfo.counterDataImageSize = counterDataImage.size();
        result = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "cuptiPmSamplingGetCounterDataInfo failed with error " << errstr << std::endl;
            return;
        }

        for (size_t sampleIndex = 0; sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex)
        {
            pmSamplingHost.EvaluateCounterData(cuptiPmSamplingTarget.GetPmSamplerObject(), sampleIndex, metricsList, counterDataImage);
        }
        result = cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "ResetCounterDataImage failed with error " << errstr << std::endl;
            return;
        }
    }
}

void DecodeCounterDataSync( std::vector<uint8_t>& counterDataImage,
                        std::vector<const char*> metricsList,
                        CuptiPmSampling& cuptiPmSamplingTarget,
                        CuptiPmProfilerHost& pmSamplingHost,
                        CUptiResult& result
                        )
{
    const char *errstr;
    result = cuptiPmSamplingTarget.DecodePmSamplingData(counterDataImage);
    if (result != CUPTI_SUCCESS)
    {
        cuptiGetResultString(result, &errstr);
        std::cerr << "DecodePmSamplingData failed with error " << errstr << std::endl;
        return;
    }

    CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
    counterDataInfo.pCounterDataImage = counterDataImage.data();
    counterDataInfo.counterDataImageSize = counterDataImage.size();
    result = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
    if (result != CUPTI_SUCCESS)
    {
        cuptiGetResultString(result, &errstr);
        std::cerr << "cuptiPmSamplingGetCounterDataInfo failed with error " << errstr << std::endl;
        return;
    }

    for (size_t sampleIndex = 0; sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex)
    {
        pmSamplingHost.EvaluateCounterData(cuptiPmSamplingTarget.GetPmSamplerObject(), sampleIndex, metricsList, counterDataImage);
    }
    result = cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage);
    if (result != CUPTI_SUCCESS)
    {
        cuptiGetResultString(result, &errstr);
        std::cerr << "ResetCounterDataImage failed with error " << errstr << std::endl;
        return;
    }
}