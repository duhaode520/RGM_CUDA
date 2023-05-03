#include "DataConfig.h"
#include <string>
#include <regex>
#include <cmath>


DataConfig* dataConfig;

DataConfig::DataConfig()
{
}

DataConfig::~DataConfig() {
    delete[] nodeNames;
}

void DataConfig::load(std::string filename) {
    std::map<std::string, std::string> config;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        throw std::runtime_error("Error opening file " + filename);
    }
    
    // 匹配 key-value
    parseConfig(infile, config);
    infile.close();

    // 读取配置
    nodeNum = std::stoi(config["nodeNum"]);
    dataType = config["dataType"];
    dataFile = config["dataFile"];
    dim = 2 * nodeNum + 1;
    cDim = std::stoi(config["cDim"]);

    PSwarmNum = int(ceil(1.0 * dim / cDim));
    flowNum = nodeNum * (nodeNum - 1) / 2;
    flowScale = std::stoi(config["flowScale"]);
    distScale = std::stoi(config["distScale"]);
    outputFile = config["outputFile"];

    // 读取节点名称列表
    nodeNames = new std::string[nodeNum];
    parseList(nodeNames, config["nodeNames"]);
       
}

void DataConfig::parseConfig(std::ifstream& infile, std::map<std::string, std::string>& config) {
    std::string line;
    std::string section;
    std::smatch match;
    std::regex keyValueRegex("(.*)=(.*)");
    while (std::getline(infile, line)) {
        if (std::regex_match(line, match, keyValueRegex)) {
            std::string key = match[1];
            std::string value = match[2];
            config[key] = value;
        }
    }
}

void DataConfig::parseList(std::string* resultList, std::string listStr) {
    // 匹配一对双引号中的任意字符，但不包括双引号本身
    std::regex pattern("\"([^\"]+)\"");

    std::smatch results;
    std::string::const_iterator iter = listStr.cbegin();
    std::string::const_iterator end = listStr.cend();

    int i = 0;
    while (std::regex_search(iter, end, results, pattern)) {
        // results[0] 匹配整个表达式，results[1] 匹配第一个子表达式
        resultList[i++] = results[1];
        // 更新迭代器，继续下一次匹配
        iter = results[0].second;
    }   
}
