#include "SuperRes.h"

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

struct Hacker {
    dnn::Net net;
};

string SuperRes::getFolderPath()
{
    return m_folderPath;
}

void SuperRes::setFolderPath(const std::string& folderPath)
{
    if(!pathExists(folderPath))
    {
        throw invalid_argument("cannot access "+folderPath);
    }
    m_folderPath=folderPath;
}

void SuperRes::setAlgoAndScale(std::string algo, unsigned short scale)
{
    if(scale<2 || (scale>4 && algo!="lapsrn") || (algo=="lapsrn" && (scale>8 || scale%2!=0)))
    {
        throw invalid_argument("undefined scale");
    }
    if(algo=="edsr") /// photo
    {
        m_path=m_folderPath+"/EDSR/EDSR_x"+to_string(scale)+".pb";
    }
    else if(algo=="fsrcnn")
    {
        m_path=m_folderPath+"/FSRCNN/FSRCNN_x"+to_string(scale)+".pb";
    }
    else if(algo=="fsrcnn_small")
    {
        m_path=m_folderPath+"/FSRCNN/FSRCNN-small_x"+to_string(scale)+".pb";
        algo="fsrcnn";
    }
    else if(algo=="lapsrn")
    {
        m_path=m_folderPath+"/LapSRN/LapSRN_x"+to_string(scale)+".pb";
    }
    else if(algo=="espcn") /// film
    {
        m_path=m_folderPath+"/ESPCN/ESPCN_x"+to_string(scale)+".pb";
    }
    else
    {
        throw invalid_argument("undefined algo");
    }
    m_sr.readModel(m_path);
    m_sr.setModel(algo, scale);
    m_algo=algo;
    m_scale=scale;
    m_sr.setPreferableTarget(DNN_TARGET_OPENCL);
    m_parametersSet = true;
}

void SuperRes::upRes(const Mat &input, cv::Mat &output)
{
    if(!m_parametersSet)
    {
        throw logic_error("Need to define algo and scale");
    }
    m_sr.upsample(input, output);
}

const std::string& SuperRes::getAlgo() const
{
    return m_algo;
}

unsigned short SuperRes::getScale() const
{
    return m_scale;
}

bool SuperRes::pathExists(const std::string& path)
{
    struct stat info;
    if(stat(path.c_str(), &info) != 0)
    {
        return false;
    }
    else if(info.st_mode & S_IFDIR)
    {
        return true;
    }
    else
    {
        return false;
    }
}
