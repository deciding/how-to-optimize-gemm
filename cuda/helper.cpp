#include "helper.h"

void PrintHelp()
{
    printf("Usage:\n");
    printf("  Query Metrics:\n");
    printf("    List Base Metrics : ./%s --device/-d <deviceIndex> --chip/-c <chipname> --queryBaseMetrics/-q\n", FUNC_NAME);
    printf("    List submetrics   : ./%s --device/-d <deviceIndex> --chip/-c <chipname> --metrics/-m <metric1,metric2,...> --queryMetricsProp/-p\n", FUNC_NAME);
    printf("  Note: when device index flag is passed, the chip name flag will be ignored.\n");
    printf("  PM Sampling:\n");
    printf("    Collection: ./%s --device/-d <deviceIndex> --samplingInterval/-i <samplingInterval> --maxsamples/-s <maxSamples in CounterDataImage> --hardwareBufferSize/-b <hardware buffer size> --metrics/-m <metric1,metric2,...>\n", FUNC_NAME);
}

ParsedArgs parseArgs(int argc, char *argv[])
{
    ParsedArgs args;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--device" || arg == "-d")
        {
            args.deviceIndex = std::stoi(argv[++i]);
        }
        else if (arg == "--samplingInterval" || arg == "-i")
        {
            args.samplingInterval = std::stoull(argv[++i]);
        }
        else if (arg == "--maxsamples" || arg == "-s")
        {
            args.maxSamples = std::stoull(argv[++i]);
        }
        else if (arg == "--hardwareBufferSize" || arg == "-b")
        {
            args.hardwareBufferSize = std::stoull(argv[++i]);
        }
        else if (arg == "--chip" || arg == "-c")
        {
            args.chipName = std::string(argv[++i]);
        }
        else if (arg == "--queryBaseMetrics" || arg == "-q")
        {
            args.queryBaseMetrics = 1;
        }
        else if (arg == "--queryMetricsProp" || arg == "-p")
        {
            args.queryMetricProperties = 1;
        }
        else if (arg == "--metrics" || arg == "-m")
        {
            std::stringstream ss(argv[++i]);
            std::string metric;
            args.metrics.clear();
            while (std::getline(ss, metric, ','))
            {
                args.metrics.push_back(strdup(metric.c_str()));
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        else
        {
            fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
            PrintHelp();
            exit(EXIT_FAILURE);
        }
    }
    return args;
}