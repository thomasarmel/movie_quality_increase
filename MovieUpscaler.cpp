#include <stdexcept>
#include <thread>
#include <memory>
#include <opencv2/core/utils/logger.hpp>
#include "MovieUpscaler.h"

MovieUpscaler::MovieUpscaler(std::string_view inputVideoFilename, std::string_view outputVideoFilename,
                             unsigned short upscaleFactor, std::string_view modelsPath) : _inputVideoFilename(
        inputVideoFilename),
                                                                                          _outputVideoFilename(
                                                                                                  outputVideoFilename),
                                                                                          _upscaleFactor(upscaleFactor),
                                                                                          _modelsPath(modelsPath)
{
}

[[maybe_unused]] const std::string &MovieUpscaler::getInputVideoFilename() const
{
    return _inputVideoFilename;
}

[[maybe_unused]] void MovieUpscaler::setInputVideoFilename(std::string_view inputVideoFilename)
{
    _inputVideoFilename = inputVideoFilename;
}

[[maybe_unused]] const std::string &MovieUpscaler::getOutputVideoFilename() const
{
    return _outputVideoFilename;
}

[[maybe_unused]] void MovieUpscaler::setOutputVideoFilename(std::string_view outputVideoFilename)
{
    _outputVideoFilename = outputVideoFilename;
}

[[maybe_unused]] unsigned short MovieUpscaler::getUpscaleFactor() const
{
    return _upscaleFactor;
}

[[maybe_unused]] void MovieUpscaler::setUpscaleFactor(unsigned short upscaleFactor)
{
    _upscaleFactor = upscaleFactor;
}

[[maybe_unused]] const std::string &MovieUpscaler::getModelsPath() const
{
    return _modelsPath;
}

[[maybe_unused]] void MovieUpscaler::setModelsPath(std::string_view modelsPath)
{
    _modelsPath = modelsPath;
}

[[maybe_unused]] size_t MovieUpscaler::getSuperresInstancesNumber() const
{
    return _superresInstancesNumber;
}

[[maybe_unused]] void MovieUpscaler::setSuperresInstancesNumber(size_t superresInstancesNumber)
{
    _superresInstancesNumber = superresInstancesNumber;
}

[[maybe_unused]] void MovieUpscaler::run(const std::optional<std::function<bool(const size_t &)>> &progressCallback)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); // Avoid OpenCV logs
    if (!checkInitialized())
    {
        throw std::invalid_argument(
                "Uninitialized MovieUpscaler: input video, outpyt video, upscale factor and models path must be set");
    }
    if (!_inputVideoCapture.open(_inputVideoFilename, cv::CAP_FFMPEG))
    {
        throw std::invalid_argument("Could not open input video file: " + _inputVideoFilename);
    }

    VideoInformations inputVideoInformations = GetVideoInformations(_inputVideoCapture);

    initiateTaskAndIdQueues(); // Fill id queues and clear task queue

    std::vector<SuperRes> superResArray(_superresInstancesNumber);

    for (size_t i = 0; i < _superresInstancesNumber; i++)
    {
        superResArray[i].setModelFolderPath(_modelsPath);
        superResArray[i].setAlgoAndScale(DEFAULT_SUPERRES_ALGO, _upscaleFactor);
    }

    std::vector<cv::Mat> outputMats(_superresInstancesNumber); // Output frames

    if (!_outputVideoWriter.open(_outputVideoFilename, cv::VideoWriter::fourcc('A', 'V', 'C', '1'),
                                 inputVideoInformations.fps,
                                 cv::Size(inputVideoInformations.width * _upscaleFactor,
                                          inputVideoInformations.height * _upscaleFactor)))
    {
        throw std::invalid_argument("Could not open output video file: " + _outputVideoFilename);
    }

    // Write output frames to video
    std::thread consumerSuperresFuturesThread(&MovieUpscaler::consumeSuperresFuturesTask, this,
                                              std::ref(_outputVideoWriter), std::ref(outputMats));

    for (unsigned long long numFrame = 0;; ++numFrame)
    {
        bool callbackShouldContinue = true; // Callback requested stop ?
        if (progressCallback.has_value())
        {
            callbackShouldContinue = progressCallback.value()(numFrame);
        }
        std::shared_ptr<cv::Mat> framePtr = std::make_shared<cv::Mat>();
        if (!_inputVideoCapture.read(*framePtr) || !callbackShouldContinue)
        {
            _waitingSuperresTasks.push(std::nullopt); // Tell writer thread to stop
            break;
        }
        if (_waitingSuperresTasks.size() >= _superresInstancesNumber ||
            _vacantSuperresAndOutputIds.empty()) // Wait until there is an available output and superres task
        {
            std::unique_lock<std::mutex> lckQueueOverflow(_mtxQueueOverflow);
            _conditionVariableQueueOverflow.wait(lckQueueOverflow, [&]() -> bool {
                return _waitingSuperresTasks.size() < _superresInstancesNumber && !_vacantSuperresAndOutputIds.empty();
            });
        }
        size_t superresId = _vacantSuperresAndOutputIds.front(); // First available superres and output frame
        _vacantSuperresAndOutputIds.pop();
        std::unique_lock<std::mutex> lckQueueEmpty(_mtxQueueEmpty);
        _waitingSuperresTasks.emplace(std::optional<std::future<size_t>>(
                std::async(std::launch::async,
                           [&superResArray, &outputMats, superresId, framePtr]() -> size_t {
                               superResArray[superresId].upRes(*framePtr, outputMats[superresId]);
                               return superresId;
                           }))); // Add new task to superrres a frame
        _conditionVariableQueueEmpty.notify_one(); // If queue was empty, no longer empty
    }

    consumerSuperresFuturesThread.join(); // All frames are written to video before closing the video writer
    _inputVideoCapture.release();
    _outputVideoWriter.release();
}

bool MovieUpscaler::checkInitialized() const
{
    return !_inputVideoFilename.empty() && !_outputVideoFilename.empty() && _upscaleFactor != 0 && !_modelsPath.empty();
}

MovieUpscaler::VideoInformations MovieUpscaler::GetVideoInformations(const cv::VideoCapture &inputVideo)
{
    return VideoInformations{
            .width =  (unsigned short) inputVideo.get(cv::CAP_PROP_FRAME_WIDTH),
            .height =  (unsigned short) inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT),
            .fps =  inputVideo.get(cv::CAP_PROP_FPS)
    };
}

void MovieUpscaler::consumeSuperresFuturesTask(cv::VideoWriter &videoWriter, const std::vector<cv::Mat> &outputMats)
{
    bool videoFinished = false;
    while (!videoFinished) // Task != std::nullopt
    {
        if (_waitingSuperresTasks.empty()) // No superres taskOutputFrameId available
        {
            std::unique_lock<std::mutex> lckQueueEmpty(_mtxQueueEmpty);
            _conditionVariableQueueEmpty.wait(lckQueueEmpty, [&]() -> bool { return !_waitingSuperresTasks.empty(); });
        }
        std::optional<std::future<size_t>> &taskOutputFrameId = _waitingSuperresTasks.front(); // Task will return output frame id
        if (taskOutputFrameId.has_value())
        {
            size_t superresAndOutputId = taskOutputFrameId.value().get(); // Wait until taskOutputFrameId is finished
            videoWriter.write(outputMats[superresAndOutputId]);
            _vacantSuperresAndOutputIds.push(superresAndOutputId); // Can reuse output frame
        } else // No more taskOutputFrameId
        {
            videoFinished = true;
        }
        std::unique_lock<std::mutex> lckQueueOverflow(_mtxQueueOverflow);
        _waitingSuperresTasks.pop(); // Task is no longer waiting
        _conditionVariableQueueOverflow.notify_one();
    }
}

void MovieUpscaler::initiateTaskAndIdQueues()
{
    ClearQueue(_vacantSuperresAndOutputIds);
    ClearQueue(_waitingSuperresTasks);
    for (size_t i = 0; i < _superresInstancesNumber; ++i)
    {
        _vacantSuperresAndOutputIds.push(i);
    }
}
