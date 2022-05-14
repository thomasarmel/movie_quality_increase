#include <iostream>
#include <exception>
#include "MovieUpscaler.h"
#include "Config.h"

int main(int argc, char *argv[])
{
    Config config;
    if (!config.parseCommandLine(argc, argv))
    {
        config.showHelp(argv[0]);
        return 2;
    }
    MovieUpscaler movieUpscaler(config.getInputFile(), config.getOutputFile(), config.getUpscaleFactor(),
                                config.getModelsDirectoryPath());
    if (config.getSimultaneousInstances() > 0) // Simultaneous inference instances is set
    {
        movieUpscaler.setSuperresInstancesNumber(config.getSimultaneousInstances());
    }
    try
    {
        movieUpscaler.run([](size_t frameID) -> bool {
            std::cout << "\rFrame: " << frameID << std::flush;
            return true; // Continue until the end of the movie
        });
    } catch (std::exception const &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}