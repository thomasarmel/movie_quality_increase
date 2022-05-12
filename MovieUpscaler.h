#ifndef MOVIE_QUALITY_INCREASE_MOVIEUPSCALER_H
#define MOVIE_QUALITY_INCREASE_MOVIEUPSCALER_H

#include <string>
#include <string_view>
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
    /**
     * @brief Construct a new Movie Upscaler object
     * @note Don't forget to initialize input / output video, upscale factor and models path
     */
    MovieUpscaler() = default;

    /**
     * @brief Initialize the MovieUpscaler object
     * @param inputVideoFilename Filename of the input video, must be readable by ffmpeg
     * @param outputVideoFilename Filename of the output video, encoded with AVC1 (file should be .mp4)
     * @param upscaleFactor Video upscale factor (1, 2, 3, 4, 8, depending on the model)
     * @param modelsPath Path to the models folder
     * @note Models folder must contain the following subfolders:
     * EDSR, ESPCN, FSRCNN, LapSRN.
     */
    MovieUpscaler(std::string_view inputVideoFilename, std::string_view outputVideoFilename,
                  unsigned short upscaleFactor, std::string_view modelsPath);

    MovieUpscaler(const MovieUpscaler &other) = delete; // Disallow copy

    MovieUpscaler &operator=(const MovieUpscaler &other) = delete; // Disallow copy

    /**
     * @brief Destroy the MovieUpscaler object
     */
    ~MovieUpscaler() = default;

    /**
     * @brief Get input video filename
     * @return Input video filename, as a string reference
     */
    [[maybe_unused]] [[nodiscard]] const std::string &getInputVideoFilename() const;

    /**
     * @brief Set the input video filename
     * @param inputVideoFilename Input video filename, must be readable by ffmpeg
     */
    [[maybe_unused]] void setInputVideoFilename(std::string_view inputVideoFilename);

    /**
     * @brief Get output video filename
     * @return Output video filename, as a string reference
     */
    [[maybe_unused]] [[nodiscard]] const std::string &getOutputVideoFilename() const;

    /**
     * @brief Set the output video filename
     * @param outputVideoFilename Output video filename
     * @note Output video filename will be encoded with AVC1 (file should be .mp4)
     */
    [[maybe_unused]] void setOutputVideoFilename(std::string_view outputVideoFilename);

    /**
     * @brief Get video upscale factor
     * @return Upscale factor
     */
    [[maybe_unused]] [[nodiscard]] unsigned short getUpscaleFactor() const;

    /**
     * @brief Set the video upscale factor
     * @param upscaleFactor Upscale factor for the video, depending on the model (1, 2, 3, 4, 8)
     */
    [[maybe_unused]] void setUpscaleFactor(unsigned short upscaleFactor);

    /**
     * @brief Get models path
     * @return Models path, as a string reference
     * @note Models folder must contain the following subfolders:
     * EDSR, ESPCN, FSRCNN, LapSRN.
     */
    [[maybe_unused]] [[nodiscard]] const std::string &getModelsPath() const;

    /**
     * @brief Set the models path
     * @param modelsPath Path to the models folder
     * @note Models folder must contain the following subfolders:
     * EDSR, ESPCN, FSRCNN, LapSRN.
     */
    [[maybe_unused]] void setModelsPath(std::string_view modelsPath);

    /**
     * @brief Run the MovieUpscaler
     * @param Optional callback function to be called after each frame is read and before it is written
     * @note Callback function takes as argument the current frame number
     * @note If callback function returns false, the MovieUpscaler will stop
     */
    [[maybe_unused]] void run(const std::optional<std::function<bool(const size_t&)>>& progressCallback = std::nullopt);

    static constexpr size_t DEFAULT_SUPERRES_INSTANCES_NUMBER = 8; // 8 simultaneous inference instances by default, reduce if you run out of memory

    static constexpr SuperRes::Algo DEFAULT_SUPERRES_ALGO = SuperRes::Algo::ESPCN; // Should not be changed, because it's the best for movies

    /**
     * @brief Get the number of concurrent inference instances
     * @return Simultaneous inference instances
     * @note You should reduce this value if you run out of memory
     */
    [[maybe_unused]] [[nodiscard]] size_t getSuperresInstancesNumber() const;

    /**
     * @brief Set the number of concurrent inference instances
     * @param superresInstancesNumber Number of concurrent inference instances
     * @note You should reduce this value if you run out of memory
     */
    [[maybe_unused]] void setSuperresInstancesNumber(size_t superresInstancesNumber);

private:
    typedef struct {
        unsigned short width;
        unsigned short height;
        double fps; // Frames per second, same in input and output
    } VideoInformations;

    [[nodiscard]] bool checkInitialized() const; // Check if all needed parameters are set
    static VideoInformations GetVideoInformations(const cv::VideoCapture &inputVideo);
    void consumeSuperresFuturesTask(cv::VideoWriter& videoWriter, const std::vector<cv::Mat>& outputMats);
    void initiateTaskAndIdQueues();

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
