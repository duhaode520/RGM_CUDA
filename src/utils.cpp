/**
 * @file utils.cu
 * @author Haode Du
 * @brief 
 * @version 1
 * @copyright Copyright (c) 2023
 * 
 */
#include "utils.h"

/**
 * @brief Generate Random N(0,1) Number using Box-Muller Method
 * 
 * @return random number [0-1] 
 */
float BoxMullerRandom() {
    float s1=1.0*rand()/RAND_MAX;
    float s2=1.0*(rand()+1U)/(RAND_MAX+1U); //To avoid zero s2
    float r=cos(2*PI*s1)*sqrt(-2*log(s2));
    return r;
}

float random01() {
    return (float)rand() / RAND_MAX;
}

void parseArgs(int argc, char *argv[])
{
    if (argc > 3) {
        std::cerr << "More than 2 arguments are not supported" << std::endl;
        throw std::invalid_argument("More than 2 arguments are not supported");
    } else if (argc == 3) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTION]" << std::endl;
            std::cout << "Run the program with default parameters" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h, --help\t\tDisplay this help and exit" << std::endl;
            std::cout << "  -t, --version\t\tOutput version information and exit" << std::endl;
            exit(0);
        } else if (arg == "-t" || arg == "--type") {
            std::string type = argv[2];
            if (type == "county") {
                std::cout << "Running with county data" << std::endl;
                dataConfig->load("./configs/county.ini");
                dataConfig->dataType="county";
            } else if (type == "city") {
                std::cout << "Running with city data" << std::endl;
                dataConfig->load("./configs/city.ini");
                dataConfig->dataType="city";
            } else {
                std::cerr << "Invalid type argument. Type argument must be county or city" << std::endl;
            }
        } else {
            std::cerr << "Invalid argument" << std::endl;
            exit(1);
        }
    } else {
        throw std::invalid_argument("Too few arguments");
    }
}
