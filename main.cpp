#include <iostream>
#include <string_view>
#include <future>
#include <queue>
#include <thread>
#include <condition_variable>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include "SuperRes.h"

constexpr unsigned short BUFFER_SIZE = 10;
constexpr std::string_view FILENAME = "/storage/séries/Alerte Cobra/Alerte Cobra - S24/Alerte Cobra 24x01 Justice Pour Un Ami Disparu - By Willy Le Belge.avi";

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

std::queue<std::optional<std::future<Mat>>> waitingSuperresTasks;
std::queue<size_t> waitingSuperresIds;
std::mutex mtx;
std::condition_variable conditionVariableQueueOverflow, conditionVariableQueueEmpty;

void consumeSuperresTask(VideoWriter& videoWriter);

int main()
{
    for(size_t i = 0; i < BUFFER_SIZE; i++)
    {
        waitingSuperresIds.push(i);
    }
    cv::VideoCapture cap(std::string(FILENAME), cv::CAP_FFMPEG);

    if(!cap.isOpened())
    {
        std::cerr << "Unable to open file " << FILENAME << std::endl;
        return -1;
    }
    cv::Mat frame, converted;
    double fps=cap.get(CAP_PROP_FPS);
    unsigned short oldWidth=cap.get(CAP_PROP_FRAME_WIDTH);
    unsigned short oldHeight=cap.get(CAP_PROP_FRAME_HEIGHT);

    std::vector<SuperRes> superRes(BUFFER_SIZE);
    try
    {
        for(unsigned short i=0; i<BUFFER_SIZE; i++)
        {
            superRes[i].setFolderPath("/opt/dnn_superres_models");
            superRes[i].setAlgoAndScale("espcn", 4);
        }
    }
    catch(exception const &e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    VideoWriter video("/tmp/outcpp.avi", VideoWriter::fourcc('M','J','P','G'), fps, Size(oldWidth*superRes[0].getScale(), oldHeight*superRes[0].getScale()));
    unsigned long long numFrame(0);

    std::thread consumerThread(consumeSuperresTask, std::ref(video));

    while(1)
    {
        std::shared_ptr<Mat> framePtr = std::make_shared<cv::Mat>();
        if(!cap.read(*framePtr))
        {
            waitingSuperresTasks.push(std::nullopt);
            break;
        }
        if(waitingSuperresTasks.size() < BUFFER_SIZE || waitingSuperresIds.empty())
        {
            size_t superresId = waitingSuperresIds.front();
            waitingSuperresIds.pop();
            std::unique_lock<std::mutex> lck(mtx);
            waitingSuperresTasks.emplace(std::optional<std::future<Mat>>(std::async(std::launch::async, [&superRes, superresId, framePtr]() -> cv::Mat {
                cv::Mat newFrame = superRes[superresId].upRes(*framePtr);
                waitingSuperresIds.push(superresId);
                return newFrame;
            })));
            conditionVariableQueueEmpty.notify_one();
        }
        else
        {
            std::unique_lock<std::mutex> lck(mtx);
            conditionVariableQueueOverflow.wait(lck, [&]() -> bool { return waitingSuperresTasks.size() < BUFFER_SIZE && !waitingSuperresIds.empty(); });
        }
        cout << "\rFrame: " << numFrame << flush;
        ++numFrame;
    }
    consumerThread.join();

    /*while(1)
    {
        std::vector<std::future<Mat>> promises(BUFFER_SIZE);
        std::vector<Mat> frames(BUFFER_SIZE);
        for(int i = 0; i < BUFFER_SIZE; ++i)
        {
            cap >> frames[i];
            if(frames[i].empty())
            {
                goto stop; // berk
            }
            promises[i] = std::async(std::launch::async, [&superRes, i, &frames] {
                return superRes[i].upRes(frames[i]);
            });
        }
        for(int i = 0; i < BUFFER_SIZE; ++i)
        {
            video.write(promises[i].get());
        }
        cout << "\rFrame: " << numFrame << flush;
        numFrame+= BUFFER_SIZE;
    }

    stop:*/
    cap.release();
    video.release();
    return 0;
}


void consumeSuperresTask(VideoWriter& videoWriter)
{
    bool videoFinished = false;
    while(!videoFinished)
    {
        if(waitingSuperresTasks.empty())
        {
            std::unique_lock<std::mutex> lck1(mtx);
            conditionVariableQueueEmpty.wait(lck1, [&]() -> bool { return !waitingSuperresTasks.empty(); });
        }
        std::optional<std::future<Mat>> &task = waitingSuperresTasks.front();
        if(task.has_value())
        {
            videoWriter.write(task.value().get());
        }
        else
        {
            videoFinished = true;
        }
        std::unique_lock<std::mutex> lck2(mtx);
        waitingSuperresTasks.pop();
        conditionVariableQueueOverflow.notify_one();
    }
}