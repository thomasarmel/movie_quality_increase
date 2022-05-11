#include <string_view>
#include <utility>
#include <stdexcept>
#include <sys/stat.h>
#include "SuperRes.h"

constexpr std::pair<std::string_view, std::string_view> EDSR_NAME_AND_SUBPATH = {"edsr", "/EDSR/EDSR_x"};
constexpr std::pair<std::string_view, std::string_view> FSRCNN_NAME_AND_SUBPATH = {"fsrcnn", "/FSRCNN/FSRCNN_x"};
constexpr std::pair<std::string_view, std::string_view> FSRCNN_SMALL_NAME_AND_SUBPATH = {"fsrcnn", "/FSRCNN/FSRCNN-small_x"};
constexpr std::pair<std::string_view, std::string_view> LAPSRN_NAME_AND_SUBPATH = {"lapsrn", "/LapSRN/LapSRN_x"};
constexpr std::pair<std::string_view, std::string_view> ESPCN_NAME_AND_SUBPATH = {"espcn", "/ESPCN/ESPCN_x"};
constexpr std::string_view MODEl_FILE_EXTENSION = ".pb";

SuperRes::SuperRes(const std::string &modelFolderPath, Algo algo, unsigned short upscaleFactor)
{
    setModelFolderPath(modelFolderPath);
    setAlgoAndScale(algo, upscaleFactor);
}

void SuperRes::setModelFolderPath(const std::string &modelFolderPath)
{
    if (!PathExists(modelFolderPath))
    {
        throw std::invalid_argument("Cannot access " + modelFolderPath);
    }
    _modelsFolderPath = modelFolderPath;

    // Refresh model if already set
    if (_parametersSet)
    {
        setAlgoAndScale(_algo, _upscaleFactor);
    }
}

void SuperRes::setAlgoAndScale(Algo algo, unsigned short upscaleFactor)
{
    if (upscaleFactor < 2 || (upscaleFactor > 4 && algo != Algo::LapSRN) || (algo == Algo::LapSRN && (upscaleFactor > 8 || upscaleFactor % 2 != 0)))
    {
        throw std::invalid_argument("Undefined upscaleFactor");
    }
    std::string algoStr;
    switch (algo)
    {
    case Algo::EDSR:
        _inferenceModelPath = _modelsFolderPath + std::string(EDSR_NAME_AND_SUBPATH.second) + std::to_string(upscaleFactor) + std::string(MODEl_FILE_EXTENSION);
        algoStr = std::string(EDSR_NAME_AND_SUBPATH.first);
        break;
    case Algo::FSRCNN:
        _inferenceModelPath = _modelsFolderPath + std::string(FSRCNN_NAME_AND_SUBPATH.second) + std::to_string(upscaleFactor) + std::string(MODEl_FILE_EXTENSION);
        algoStr = std::string(FSRCNN_NAME_AND_SUBPATH.first);
        break;
    case Algo::FSRCNN_SMALL:
        _inferenceModelPath = _modelsFolderPath + std::string(FSRCNN_SMALL_NAME_AND_SUBPATH.second) + std::to_string(upscaleFactor) + std::string(MODEl_FILE_EXTENSION);
        algoStr = std::string(FSRCNN_SMALL_NAME_AND_SUBPATH.first);
        break;
    case Algo::LapSRN:
        _inferenceModelPath = _modelsFolderPath + std::string(LAPSRN_NAME_AND_SUBPATH.second) + std::to_string(upscaleFactor) + std::string(MODEl_FILE_EXTENSION);
        algoStr = std::string(LAPSRN_NAME_AND_SUBPATH.first);
        break;
    case Algo::ESPCN:
        _inferenceModelPath = _modelsFolderPath + std::string(ESPCN_NAME_AND_SUBPATH.second) + std::to_string(upscaleFactor) + std::string(MODEl_FILE_EXTENSION);
        algoStr = std::string(ESPCN_NAME_AND_SUBPATH.first);
        break;
    }
    _superresInferenceEngine.readModel(_inferenceModelPath);
    _superresInferenceEngine.setModel(algoStr, upscaleFactor);
    _algo = algo;
    _upscaleFactor = upscaleFactor;
    _superresInferenceEngine.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL); // Use GPU if available
    _parametersSet = true;
}

void SuperRes::upRes(const cv::Mat &input, cv::Mat &output)
{
    if (!_parametersSet)
    {
        throw std::logic_error("Need to define algo and scale");
    }
    _superresInferenceEngine.upsample(input, output);
}

SuperRes::Algo SuperRes::getAlgo() const
{
    return _algo;
}

unsigned short SuperRes::getScale() const
{
    return _upscaleFactor;
}

const std::string& SuperRes::getModelsFolderPath() const
{
    return _modelsFolderPath;
}

bool SuperRes::PathExists(const std::string &path)
{
    struct stat info{};
    return (stat(path.c_str(), &info) == 0) && (info.st_mode & S_IFDIR);
}