#include <iostream>
#include <array>
#include <string_view>
#include "Config.h"

constexpr std::array<std::string_view, 2> HELP_COMMAND = { "--help", "-h" };
constexpr std::array<std::string_view, 2> UPSCALE_FACTOR_COMMAND = { "--factor", "-f" };
constexpr std::array<std::string_view, 2> INPUT_FILE_COMMAND = { "--input-file", "-i" };
constexpr std::array<std::string_view, 2> OUTPUT_FILE_COMMAND = { "--output-file", "-o" };
constexpr std::array<std::string_view, 2> MODELS_DIR_COMMAND = { "--models-dir", "-m" };
constexpr std::array<std::string_view, 2> PARALLEL_INSTANCES = { "--parallel-instances", "-p" };

bool Config::parseCommandLine(int argc, const char *const *argv)
{
    for (int i = 1; i < argc; i++)
    {
        std::string_view currentArg = argv[i];
        if (currentArg == HELP_COMMAND[0] || currentArg == HELP_COMMAND[1])
        {
            return false;
        }
        if (i == argc - 1) // last argument
        {
            break;
        }
        std::string_view nextArg = argv[i + 1];
        if (currentArg == UPSCALE_FACTOR_COMMAND[0] || currentArg == UPSCALE_FACTOR_COMMAND[1])
        {
            _upscaleFactor = std::stoi(std::string(nextArg));
        }
        else if (currentArg == INPUT_FILE_COMMAND[0] || currentArg == INPUT_FILE_COMMAND[1])
        {
            _inputFile = std::string(nextArg);
        }
        else if (currentArg == OUTPUT_FILE_COMMAND[0] || currentArg == OUTPUT_FILE_COMMAND[1])
        {
            _outputFile = std::string(nextArg);
        }
        else if (currentArg == MODELS_DIR_COMMAND[0] || currentArg == MODELS_DIR_COMMAND[1])
        {
            _modelsDirectoryPath = std::string(nextArg);
        }
        else if (currentArg == PARALLEL_INSTANCES[0] || currentArg == PARALLEL_INSTANCES[1])
        {
            _simultaneousInstances = std::stoi(std::string(nextArg));
        }
    }
    return !_inputFile.empty() && !_outputFile.empty() && !_modelsDirectoryPath.empty() && _upscaleFactor > 0 && _simultaneousInstances >= 0;
}

void Config::showHelp(std::string_view programPath)
{
    std::cout << "Usage:" << std::endl;
    std::cout << programPath << "";
    std::cout << " [-h | --help]";
    std::cout << " {-f | --factor} <upscaleFactor>";
    std::cout << " {-i | --input-file} <inputFilePath>";
    std::cout << " {-o | --output-file} <outputFilePath>";
    std::cout << " {-m | --models-dir} <modelsDirectoryPath>";
    std::cout << " [{-p | --parallel-instances} <simultaneousInstances>]";
    std::cout << std::endl;
}

const std::string &Config::getInputFile() const
{
    return _inputFile;
}

const std::string &Config::getOutputFile() const
{
    return _outputFile;
}

unsigned short Config::getUpscaleFactor() const
{
    return _upscaleFactor;
}

unsigned short Config::getSimultaneousInstances() const
{
    return _simultaneousInstances;
}

const std::string &Config::getModelsDirectoryPath() const
{
    return _modelsDirectoryPath;
}
