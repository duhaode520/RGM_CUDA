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
double BoxMullerRandom() {
    double s1=1.0*rand()/RAND_MAX;
    double s2=1.0*(rand()+1U)/(RAND_MAX+1U); //To avoid zero s2
    double r=cos(2*PI*s1)*sqrt(-2*log(s2));
    return r;
}

void parseArgs(int argc, char *argv[]) {
    if (argc > 3) {
        std::cerr << "More than 2 arguments are not supported" << std::endl;
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
                dataConfig.load("configs/county.ini");
                dataConfig.dataType='county'
            } else if (type == "city") {
                dataConfig.load("configs/city.ini");
                dataConfig.dataType='city'
            } else {
                std::cerr << "Invalid type argument. Type argument must be county or city" << std::endl;
            }
        } else {
            std::cerr << "Invalid argument" << std::endl;
            exit(1);
        }
    }
}
