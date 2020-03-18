#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::string line;
    std::ifstream testcase(argv[1]);
    std::vector<std::string> game_map;

    if (testcase.is_open()) {
        while (std::getline(testcase, line)) {
            game_map.emplace_back(line);
        }
        testcase.close();
    } else
        std::cerr << "Unable to open file.\n";

    return 0;
}
