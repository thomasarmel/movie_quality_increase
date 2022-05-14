#ifndef MOVIE_QUALITY_INCREASE_CONFIG_H
#define MOVIE_QUALITY_INCREASE_CONFIG_H

#include <string>

class Config
{
public:
    /**
     * @brief Construct a new Config object
     */
    Config() = default;

    /**
     * @brief Destroy the Config object
     */
    ~Config() = default;

    /*
     * @brief Fill configuration with command line arguments
     * @param argc Number of arguments
     * @param argv Arguments
     * @return True if configuration is valid
     * @note If user requested help, configuration is not valid and help message will be displayed
     */
    bool parseCommandLine(int argc, const char *const argv[]);

    /**
     * @brief Print help message on stdout
     * @param programPath Path to call program : argv[0]
     */
    void showHelp(std::string_view programPath);

    /**
     * @brief Get path to input file
     * @return Path to input file
     * @note parseCommandLine() must be called before
     */
    [[nodiscard]] const std::string &getInputFile() const;

    /**
     * @brief Get path to output file
     * @return Path to output file
     * @note parseCommandLine() must be called before
     */
    [[nodiscard]] const std::string &getOutputFile() const;

    /**
     * @brief Get quality increase value
     * @return Quality increase value
     * @note parseCommandLine() must be called before
     * @note Quality increase value coould be 2, 3, 4 or 8
     */
    [[nodiscard]] unsigned short getUpscaleFactor() const;

    /**
     * @brief Get number of simultaneous inference instances
     * @return Number of simultaneous inference instances
     * @note parseCommandLine() must be called before
     * @note Consider reducing it if you run out of memory
     */
    [[nodiscard]] unsigned short getSimultaneousInstances() const;

    /**
     * @brief Get models directory path
     * @return Models directory path
     * @note parseCommandLine() must be called before
     * @note Models directory path must be a valid directory
     * @note Models directory path must contain the following directories :
     * EDSR, ESPCN, FSRCNN, LapSRN.
     */
    [[nodiscard]] const std::string &getModelsDirectoryPath() const;

private:
    std::string _inputFile;
    std::string _outputFile;
    unsigned short _upscaleFactor = 0;
    unsigned short _simultaneousInstances = 0; // Number of simultaneous instances of inference
    std::string _modelsDirectoryPath;
};


#endif //MOVIE_QUALITY_INCREASE_CONFIG_H
