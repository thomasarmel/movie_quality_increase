#include <iostream>
#include <exception>
#include "MovieUpscaler.h"

constexpr std::string_view FILENAME = "/storage/séries/Alerte Cobra/Alerte Cobra - S24/Alerte Cobra 24x01 Justice Pour Un Ami Disparu - By Willy Le Belge.avi";

int main()
{
    MovieUpscaler movieUpscaler(std::string(FILENAME), std::string("/tmp/outcpp.mp4"), 4, std::string("/opt/dnn_superres_models"));
    try {
        movieUpscaler.run([](size_t frameID) -> bool {
            std::cout << "\rFrame: " << frameID << std::flush;
            return frameID <= 1000;
        });
    } catch (std::exception const &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}