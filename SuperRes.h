#ifndef MOVIE_QUALITY_INCREASE_SUPERRES_H
#define MOVIE_QUALITY_INCREASE_SUPERRES_H

#include <vector>
#include <opencv2/dnn_superres.hpp>
#include <exception>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>

class SuperRes
{
public:
    SuperRes() = default;
    ~SuperRes() = default;
    void setFolderPath(const std::string& folderPath);
    std::string getFolderPath();
    void setAlgoAndScale(std::string algo, unsigned short scale);
    void upRes(const cv::Mat &input, cv::Mat &output);
    [[nodiscard]] const std::string& getAlgo() const;
    [[nodiscard]] unsigned short getScale() const;

private:
    bool pathExists(const std::string& path);
    cv::dnn_superres::DnnSuperResImpl m_sr;
    std::string m_path;
    std::string m_folderPath;
    std::string m_algo;
    unsigned short m_scale = 0;
    bool m_parametersSet = false;
};


#endif //MOVIE_QUALITY_INCREASE_SUPERRES_H
