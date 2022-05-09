#ifndef MOVIE_QUALITY_INCREASE_SUPERRES_H
#define MOVIE_QUALITY_INCREASE_SUPERRES_H

#include <vector>
#include <opencv2/dnn_superres.hpp>

class SuperRes
{
public:
    enum class Algo
    {
        /**
         * @brief EDSR superres model
         * @details This is the best performing model. However, it is also the biggest model and therefor has the biggest file size and slowest inference.
         * It's maybe the perfect one for pictures.
         * @note Available with superres x2, x3 and x4.
         */
        EDSR, // Perfect for pictures

        /**
         * @brief FSRCNN superres model
         * @details This is small model with fast and accurate inference. Can do real-time video upscaling.
         * @note Available with superres x2, x3 and x4.
         */
        FSRCNN,

        /**
         * @brief FSRCNN_SMALL superres model
         * @details This is small model with fast and accurate inference. Can do real-time video upscaling.
         * It's the same model than FSRCNN but with a smaller network. Thus it is faster but has lower performance.
         * @note Available with superres x2, x3 and x4.
         */
        FSRCNN_SMALL,

        /**
         * @brief LapSRN superres model
         * @details This is a medium sized model that can upscale by a factor as high as 8.
         * @note Available with superres x2, x4 and x8.
         */
        LapSRN,

        /**
         * @brief ESPCN superres model
         * @details This is a small model with fast and good inference. It can do real-time video upscaling (depending on image size).
         * It's maybe the perfect one for upscaling movies.
         * @note Available with superres x2, x3 and x4.
         */
        ESPCN // Perfect for movies
    };
    // Thanks to https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066 for models description

    /**
     * @brief Construct a new SuperRes object
     */
    SuperRes() = default;

    /**
     * @brief Construct a new SuperRes object
     * @param modelFolderPath Path to the models folder.
     * @param algo The superres algorithm to use
     * @param upscaleFactor The upscale to apply
     * @throw std::invalid_argument If the model folder doesn't exist or if the upscale factor is not supported by the algorithm
     * @note The scales available depend on the algorithm
     * @note Models folder must contain the following subfolders:
     * EDSR, ESPCN, FSRCNN, LapSRN.
     * @note These folders must contain the following files:
     * @note EDSR: EDSR_x2.pb EDSR_x3.pb EDSR_x4.pb
     * @note ESPCN: ESPCN_x2.pb ESPCN_x3.pb ESPCN_x4.pb
     * @note FSRCNN: FSRCNN_x2.pb FSRCNN_x3.pb FSRCNN_x4.pb FSRCNN-small_x2.pb FSRCNN-small_x3.pb FSRCNN-small_x4.pb
     * @note LapSRN: LapSRN_x2.pb LapSRN_x4.pb LapSRN_x8.pb
     */
    SuperRes(const std::string &modelFolderPath, Algo algo, unsigned short upscaleFactor);

    /**
     * @brief Destroy the SuperRes object
     */
    ~SuperRes() = default;

    /**
     * @brief Set the path containing the trained inference models
     * @param modelFolderPath Path to the models folder.
     * @throw std::invalid_argument If the folder doesn't exist
     * @note Models folder must contain the following subfolders:
     * EDSR, ESPCN, FSRCNN, LapSRN.
     * @note These folders must contain the following files:
     * @note EDSR: EDSR_x2.pb EDSR_x3.pb EDSR_x4.pb
     * @note ESPCN: ESPCN_x2.pb ESPCN_x3.pb ESPCN_x4.pb
     * @note FSRCNN: FSRCNN_x2.pb FSRCNN_x3.pb FSRCNN_x4.pb FSRCNN-small_x2.pb FSRCNN-small_x3.pb FSRCNN-small_x4.pb
     * @note LapSRN: LapSRN_x2.pb LapSRN_x4.pb LapSRN_x8.pb
     */
    void setModelFolderPath(const std::string &modelFolderPath);

    /**
     * @brief Set the superres algorithm to use and the upscale factor
     * @param algo The superres algorithm to use
     * @param upscaleFactor The upscale to apply
     * @note The scales available depend on the algorithm
     */
    void setAlgoAndScale(Algo algo, unsigned short upscaleFactor);

    /**
     * @brief Proceed the superres process on the input image
     * @param input Image to process
     * @param output Reference to the output image
     * @throw std::logic_error If the model folder, algo or upscale factor are not set
     * @note The output image will be larger than the input image.
     * @note Models folder path, algo and upscale factor must be set before calling this function.
     */
    void upRes(const cv::Mat &input, cv::Mat &output);

    /**
     * @brief Get path containing the trained inference models
     */
    [[nodiscard]] const std::string& getModelsFolderPath() const;

    /**
     * @brief Get the superres algorithm to use
     * @return The superres algorithm among the available ones (EDSR, ESPCN, FSRCNN, FSRCNN_SMALL, LapSRN)
     */
    [[nodiscard]] Algo getAlgo() const;

    /**
     * @brief Get the upscale factor
     * @note The upscale factor must be supported by the algorithm
     */
    [[nodiscard]] unsigned short getScale() const;

private:
    static bool PathExists(const std::string &path);

    cv::dnn_superres::DnnSuperResImpl _superresInferenceEngine;
    std::string _inferenceModelPath;
    std::string _modelsFolderPath;
    Algo _algo;
    unsigned short _upscaleFactor = 0;
    bool _parametersSet = false; // Can proceed inference only if parameters are set
};


#endif //MOVIE_QUALITY_INCREASE_SUPERRES_H
