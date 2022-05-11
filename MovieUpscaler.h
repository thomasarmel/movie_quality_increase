#ifndef MOVIE_QUALITY_INCREASE_MOVIEUPSCALER_H
#define MOVIE_QUALITY_INCREASE_MOVIEUPSCALER_H

#include <string>
#include <vector>
#include <queue>
#include <future>
#include <optional>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <opencv2/videoio.hpp>
#include "SuperRes.h"

class MovieUpscaler
{
public:
    MovieUpscaler() = default;

    MovieUpscaler(const std::string &inputVideoFilename, const std::string &outputVideoFilename,
                  unsigned short upscaleFactor, const std::string &modelsPath);

    MovieUpscaler(const MovieUpscaler &other) = delete;

    MovieUpscaler &operator=(const MovieUpscaler &other) = delete;

    ~MovieUpscaler() = default;

    [[maybe_unused]] [[nodiscard]] const std::string &getInputVideoFilename() const;

    [[maybe_unused]] void setInputVideoFilename(const std::string &inputVideoFilename);

    [[maybe_unused]] [[nodiscard]] const std::string &getOutputVideoFilename() const;

    [[maybe_unused]] void setOutputVideoFilename(const std::string &outputVideoFilename);

    [[maybe_unused]] [[nodiscard]] unsigned short getUpscaleFactor() const;

    [[maybe_unused]] void setUpscaleFactor(unsigned short upscaleFactor);

    [[maybe_unused]] [[nodiscard]] const std::string &getModelsPath() const;

    [[maybe_unused]] void setModelsPath(const std::string &modelsPath);

    [[maybe_unused]] void run(const std::optional<std::function<bool(const size_t&)>>& progressCallback = std::nullopt);

    static constexpr size_t DEFAULT_SUPERRES_INSTANCES_NUMBER = 8;

    static constexpr SuperRes::Algo DEFAULT_SUPERRES_ALGO = SuperRes::Algo::ESPCN; // Should not be changed

    [[maybe_unused]] [[nodiscard]] size_t getSuperresInstancesNumber() const;

    [[maybe_unused]] void setSuperresInstancesNumber(size_t superresInstancesNumber);

private:
    typedef struct {
        unsigned short width;
        unsigned short height;
        double fps;
    } VideoInformations;

    [[nodiscard]] bool checkInitialized() const;
    static VideoInformations GetVideoInformations(const cv::VideoCapture &inputVideo);
    void consumeSuperresFuturesTask(cv::VideoWriter& videoWriter, const std::vector<cv::Mat>& outputMats);

    template <typename T>
    void ClearQueue(std::queue<T> &queueToClear)
    {
        queueToClear = std::queue<T>();
    }

    std::string _inputVideoFilename;
    std::string _outputVideoFilename;
    unsigned short _upscaleFactor = 0;
    std::string _modelsPath;
    cv::VideoCapture _inputVideoCapture;
    cv::VideoWriter _outputVideoWriter;
    size_t _superresInstancesNumber = DEFAULT_SUPERRES_INSTANCES_NUMBER;
    std::queue<std::optional<std::future<size_t>>> _waitingSuperresTasks; // Task that are waiting for getting their results
    std::queue<size_t> _vacantSuperresAndOutputIds; // Ids of superres instances that are not currently working, also used for available output Mats
    std::mutex _mtxQueueOverflow, _mtxQueueEmpty;
    std::condition_variable _conditionVariableQueueOverflow, _conditionVariableQueueEmpty;
};


#endif //MOVIE_QUALITY_INCREASE_MOVIEUPSCALER_H
